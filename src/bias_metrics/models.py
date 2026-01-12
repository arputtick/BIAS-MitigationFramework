import os
import pickle
import numpy as np
import torch
import json
# import openai
# import dotenv
import string
from abc import ABC, abstractmethod
from transformers import BertTokenizer, AutoTokenizer

class EmbeddingModel(ABC):
    '''
    Abstract class for embedding models
    An instance of the class can be used to
     -load the model in RAM from storage (from formats: pickle, binary, txt, word2vec)
     -get the vector of a word
     -aggregate with the metric class (e.g. WEAT), to form an instance which allows to take a dataset and evaluate it
    '''

    LOADED_MODELS = {} # to keep track of loaded models, so we don't load them again. This is a class variable, so it is shared between all instances of the class. 
    LOADED_TOKENIZERS = {}

    def __init__(self, model_path, save_pickle=True, load_pickle=True): #model_path is the path to the model file, save_pickle and load_pickle to specify whether to save and load the model as a pickle file
        self.model_path = model_path #file with file ending e.g. "GoogleNews-vectors-negative300.bin" for word2vec
        self.model = None
        self.tokenizer = None
        self.save_pickle = save_pickle
        self.load_pickle = load_pickle
        self.language = 'en'
        self.device = None

    @abstractmethod #has to be implemented in subclasses. 
    def loading_model(self, language = 'en'):
        pass

    def check_vocab(self, word):
        '''Check if the given word is in the vocabulary of the model'''
        pass

    @abstractmethod
    def get_vector(self, word):
        pass

    def _save_as_pickle(self, filename):
        print(f"Saving model to {filename}")
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def _load_from_pickle(self, filename):
        print(f"Loading model from {filename}")
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)



class FasttextModel(EmbeddingModel):
    def loading_model(self, language):

        import fasttext # as imports are embedding-specific, we import them here
        import fasttext.util

        # Check to see if the model is already in the models folder.
        # If not, download the model in the specified language.
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            # Download model in given language if not already downloaded. Must run Install Certificates.command in Python folder to avoid SSL error.
            fasttext.util.download_model(language, if_exists='ignore')
            # Move downloaded model to models folder
            model_path = f"cc.{language}.300.bin"
            os.rename(model_path, self.model_path)

        if self.model_path in EmbeddingModel.LOADED_MODELS: # only load model if not already loaded
            print(f"Loading Fasttext model from cache")
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
        else: 
            if self.load_pickle: 
                print("cannot use pickle for fasttext model")
            print(f"Loading Fasttext model from {self.model_path}")
            self.model = fasttext.load_model(self.model_path)
            EmbeddingModel.LOADED_MODELS[self.model_path] = self.model # save model in cache
        
    def get_vector(self, sequence):
        return np.mean([self.model.get_word_vector(word) for word in sequence.split()], axis=0)


class Word2VecModel(EmbeddingModel):
    def loading_model(self, language):
        from gensim.models import KeyedVectors # as imports are embedding-specific, we import them here

        if self.model_path in EmbeddingModel.LOADED_MODELS: # only load model if not already loaded
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
        else:
            if self.load_pickle and os.path.exists(self.model_path + '.pkl'):
                self._load_from_pickle(self.model_path + '.pkl')
                EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
            else:
                print(f"Loading Word2Vec model from {self.model_path}")
                self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
                EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
                # Save model to pickle if specified
                if self.save_pickle:
                    self._save_as_pickle(self.model_path + '.pkl')
   
    def get_vector(self, sequence):
        # If the sequence is a single word, returns the vector of that word, otherwise returns the mean of all word vectors in the sequence.
        sequence = sequence.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation from sentence
        return np.mean([self.model[word] for word in sequence.split()], axis=0)


class GloveModel(EmbeddingModel):
    def loading_model(self, language):
        from gensim.models import KeyedVectors
        from gensim.scripts.glove2word2vec import glove2word2vec
        # Load model from cache
        if self.model_path in EmbeddingModel.LOADED_MODELS:
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
        else:
            if self.load_pickle and os.path.exists(self.model_path + '.word2vec.pkl'):
                print(f"Loading Glove pickle model from {self.model_path} .word2vec.pkl")
                self._load_from_pickle(self.model_path + '.word2vec.pkl')
                EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
            else:
                print(f"Loading Glove model from {self.model_path}")
                word2vec_output_file = self.model_path + '.word2vec'
                if not os.path.exists(word2vec_output_file): # convert glove to word2vec format if not already done
                    print("Converting Glove to Word2Vec format...")
                    glove2word2vec(self.model_path, word2vec_output_file)
                self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False, limit = 2196000) # limit not load the last couple of entries, as it messes up the model
                EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
                # Save model to pickle if specified
                if self.save_pickle:
                    self._save_as_pickle(word2vec_output_file + '.pkl')

    def get_vector(self, sequence):
        # If the sequence is a single word, returns the vector of that word, otherwise returns the mean of all word vectors in the sequence.
        sequence = sequence.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation from sentence
        return np.mean([self.model[word] for word in sequence.split()], axis=0)
        

class BertModel(EmbeddingModel):
    def __init__(self, model_path, save_pickle=False, load_pickle=False, save_model = True, embedding = '[CLS]'):
        '''
        model_path is the path to the model file, save_pickle and load_pickle aren't used for BERT models. 
        Instead use save_pretrained and from_pretrained methods.

        embedding = '[CLS]' , 'pooling' , 'first'  --  determines the embedding type for contextual word embeddings
        '''
        super().__init__(model_path, save_pickle, load_pickle)
        self.save_model = save_model
        self.embedding =  embedding

    def __call__(self, input_ids):
        return self.model.forward(input_ids)
      
    def loading_model(self, language = 'en', hidden_states = True):
        from transformers import BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel
        self.language = language

        if not hidden_states:
            self.model_path = self.model_path + '_logits'

        # Load model and tokenizer from cache
        if self.model_path in EmbeddingModel.LOADED_MODELS:
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
            self.tokenizer = EmbeddingModel.LOADED_TOKENIZERS[self.model_path]
            print(f"Using bert model from cache at {self.model_path}") 
            

        else:
            local = os.path.exists(self.model_path)
            if local:
                path = self.model_path
            else:
                path = self.model_path.split('models/')[-1].replace('_logits', '')
            print(f"Loading bert model from {path}")
            if self.language == 'en' or self.language == 'multi':
                self.model = BertForMaskedLM.from_pretrained(path, output_hidden_states = True)
                self.tokenizer = BertTokenizer.from_pretrained(path.replace('models/', '', 1))

            else:
                if hidden_states:
                    self.model = AutoModel.from_pretrained(path, trust_remote_code=True, output_hidden_states = True)
                    self.tokenizer = AutoTokenizer.from_pretrained(path.replace('models/', '', 1))
                else:
                    self.model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True, output_hidden_states = True)
                    path = self.model_path.split('models/')[-1].replace('_logits', '')
                    self.tokenizer = AutoTokenizer.from_pretrained(path.replace('models/', '', 1))
            
            # elif self.language == 'multi':
            #     self.model = BertModel.from_pretrained(path)
            #     self.tokenizer = BertTokenizer.from_pretrained(path.replace('models/', '', 1))
                    
            EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
            EmbeddingModel.LOADED_TOKENIZERS[self.model_path] = self.tokenizer

            # Save local copy of model
            if not local and self.save_model:
                if hidden_states:
                    self.model.save_pretrained(self.model_path)
                else:
                    self.model.save_pretrained(self.model_path + '_logits')

        ### E.g., Mac M2
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
        ### CUDA GPUs
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        except AttributeError:
            print("An error occured with torch device definition.")

        self.model.to(device)
        print("Model device:", next(self.model.parameters()).device)
        self.device = device
        self.model.eval()

    def get_vector(self, text, target_word = '[CLS]'):
        '''
        By default, returns the last hidden state correspond to the [CLS] token (as in May et al.).
        If you want to get the hidden state of a different word, specify it with the word argument.
        '''
        ## Roberta based tokenizers do not use CLS token. Instead, use <s>
        if 'Roberta' in self.tokenizer.__class__.__name__:
            target_word = '<s>'
        #print(target_word, text)
        #print(self.tokenizer.__class__)
        input_ids = torch.tensor(self.tokenizer.encode(text),device=self.device).unsqueeze(0)  # Batch size 1
        try:
            outputs = self.model(input_ids)
        except RuntimeError as e:
            print("Error during model forward pass:", e)

        embedding = self.embedding

        # Convert the word to (sub)tokens
        token = self.tokenizer.tokenize(target_word)

        if embedding == 'first':
            # If the token is broken into multiple subtokens, we take the first subtoken
            token = [token[0]]

        #print(target_word, token, text)
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        embeddings = []
        for token_id in token_ids:
            # Find index of the token in the input_ids
            token_index = input_ids[0].tolist().index(token_id)
            # Some models on HuggingFace don't return 'hidden_states' in the output.
            # In this case, we use 'last_hidden_state' instead.
            if 'hidden_states' in outputs.keys():
                last_hidden_states = outputs['hidden_states'][-1]
            else: 
                last_hidden_states = outputs['last_hidden_state']
            embedding = last_hidden_states[0][token_index].detach().cpu().numpy() # .cpu because of mps device type tensor cannot be converted
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        word_vector = np.mean(np.array(embeddings), axis = 0)
        return word_vector
    
    def check_vocab(self, word):
        '''
        Check if all words in the texts are in the vocabulary of the model.
            texts: list of words/sentences to check
        '''
        # Tokenize the word
        tokens = self.tokenizer.tokenize(word)

        # Check if the token was broken up into multiple subtokens
        if len(tokens) > 1:
            print(f"Word {word} was broken up into multiple subtokens: {tokens}")

        # Check for UNK tokens
        for token in tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == 100:
                print(f"Word {token} resulted in UNK token.")
        return



    
class GPTModel(EmbeddingModel):
    def __init__(self, model_path, save_pickle=False, load_pickle=False, save_model = True, embedding = 'last'):
        '''
        model_path is the path to the model file, save_pickle and load_pickle aren't used for GPT models. 
        Instead use save_pretrained and from_pretrained methods.

        The default embedding is the last hidden state of the last token in the sequence.
        Set embedding = 'target' to get the last hidden state of the first subtoken of target_word.
        Set embedding = 'target_pooled' to get the pooled last hidden states of all subtokens of target_word.
        '''
        super().__init__(model_path, save_pickle, load_pickle)
        self.save_model = save_model
        self.embedding =  embedding

    def __call__(self, input_ids):
        return self.model.forward(input_ids)
      
    def loading_model(self, language = 'en', hidden_states = True):
        from transformers import AutoTokenizer, AutoModel
        self.language = language

        # Load model and tokenizer from cache
        if self.model_path in EmbeddingModel.LOADED_MODELS:
            self.model = EmbeddingModel.LOADED_MODELS[self.model_path]
            self.tokenizer = EmbeddingModel.LOADED_TOKENIZERS[self.model_path]
            print(f"Using GPT model from cache at {self.model_path}") 

        else:
            local = os.path.exists(self.model_path)
            if local:
                path = self.model_path
                print(f"Loading local GPT model from {path}")
                self.model = AutoModel.from_pretrained(path, output_hidden_states = True)
                self.tokenizer = AutoTokenizer.from_pretrained(path.replace('models/', '', 1), add_prefix_space=True) # add_prefix_space=True ensures that all tokens are processed in the same way.
            else:
                path = self.model_path.replace('models/', '', 1)
                print(f"Downloading GPT model from {path}")
                self.model = AutoModel.from_pretrained(path, output_hidden_states = True)
                self.tokenizer = AutoTokenizer.from_pretrained(path)
            EmbeddingModel.LOADED_MODELS[self.model_path] = self.model
            EmbeddingModel.LOADED_TOKENIZERS[self.model_path] = self.tokenizer

            # Save local copy of model
            if not local and self.save_model:
                self.model.save_pretrained(self.model_path)

        ### E.g., Mac M2
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
        ### CUDA GPUs
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        except AttributeError:
            print("An error occured with torch device definition.")

        self.model.to(device)
        print("Model device:", next(self.model.parameters()).device)
        self.device = device
        self.model.eval()

    def get_vector(self, text, target_word = None):
        input_ids = torch.tensor(self.tokenizer.encode(text),device=self.device).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_state = outputs['last_hidden_state']
        embedding = self.embedding
        if target_word is None:
            target_word = text

        if embedding == 'first' or embedding == 'pooling':
            # Convert the word to (sub)tokens
            token = self.tokenizer.tokenize(target_word)
            if embedding == 'first':
                # If the token is broken into multiple subtokens, we take the first subtoken
                token = [token[0]]
            # print(target_word, token, text)
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            sentence_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            embeddings = []
            for token_id in token_ids:
                # Find index of the token in the input_ids
                # print(token_id, token, target_word, text, input_ids[0].tolist())
                token_index = input_ids[0].tolist().index(token_id)
                last_hidden_states = outputs['last_hidden_state']
                embedding = last_hidden_states[0][token_index].detach().cpu().numpy() # .cpu because of mps device type tensor cannot be converted
                embeddings.append(embedding)
            embeddings = np.array(embeddings)
            word_vector = np.mean(np.array(embeddings), axis = 0)

        elif embedding == 'last':
            word_vector = last_hidden_state[0][-1].detach().cpu().numpy()
    
        return word_vector


class OpenAiModel(EmbeddingModel):
    valid_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"] #class variable

    def __init__(self, model_name="text-embedding-3-small"):
        super().__init__(model_name) 
        self.model_name = model_name
        self.embeddings_cache = {}

        #imports here as they are only needed for the OpenAI embeddings
        from dotenv import load_dotenv
        from openai import OpenAI
        


        if model_name not in OpenAiModel.valid_models: #check if model name is valid
            raise ValueError(f"Invalid model_name. Choose from: {', '.join(OpenAiModel.valid_models)}")
        

        #initialize client
        # Load API key using the secure config loader
        try:
            from ..utils.config_loader import get_api_key
            api_key = get_api_key('openai')
        except (ImportError, ValueError):
            # Fallback to old method
            load_dotenv() # Load environment variables from .env file
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set it in config/credentials.json or as OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key) # Initialize OpenAI client with API key
        
        self.client = OpenAI(api_key=api_key) # Initialize OpenAI client with API key

    def loading_model(self, language = 'en'):
        pass
        # this method could be used to load embeddings which previously have been loaded already..?

    def get_vector(self, sequence):
        import time
        
        print(f"Getting embedding for sequence: {sequence}")
        print(f"Using model: {self.model_name}")
        start_time = time.time()


        # cache does not work atm. instance does not persist between calls. 
        if sequence in self.embeddings_cache:
            print(f"Embedding for sequence '{sequence}' found in cache")
            return self.embeddings_cache[sequence][:768]

        response = self.client.embeddings.create(
            input=sequence,
            model=self.model_name
        )

        embedding = response.data[0].embedding

        

        print("embedding datatype: ", type(embedding))

        print("rounding & slicing embedding")
        embedding = [embedding[:768]]

        print("embedding datatype: ", type(embedding))

        try:#print numpy array datatype
            print("before converting to numpy: embedding datatype: ", embedding.dtype)
        except:
            print("before converting to numpy: could not get numpy datatype")
        

        # chaning datatype to smaller float, to speed up calculation
        # reduce to float32
        embedding = np.array(embedding, dtype=np.float32)
        print("embedding datatype: ", type(embedding))

        try:#print numpy array datatype
            print("after converting to numpy: embedding datatype: ", embedding.dtype)
        except:
            print("after converting to numpy: could not get numpy datatype")


        embedding = embedding[0]
        

        self.embeddings_cache[sequence] = embedding


        end_time = time.time()
        response_time = end_time - start_time
        print(f"Response time: {response_time} seconds")
        #shape of the embedding
        print(f"Shape of the embedding: {embedding.shape}")
        #preview content of embedding:
        # print("first element of embedding: ", embedding[0])
        # print("second element of embedding: ", embedding[1])
        # print("third element of embedding: ", embedding[2])

        print(f"First five values of Embedding: {embedding[:5]}")
        print(f"Embedding length: {len(embedding)}")


        return embedding
 



class VoyageModel(EmbeddingModel):
    valid_models = ["voyage-multilingual-2"] #class variable

    def __init__(self, model_name="voyage-multilingual-2"):
        super().__init__(model_name) 



        #imports here as they are only needed here
        from dotenv import load_dotenv
        import voyageai
        

        if model_name not in VoyageModel.valid_models: #check if model name is valid
            raise ValueError(f"Invalid model_name. Choose from: {', '.join(VoyageModel.valid_models)}")
        

        #initialize client
        # Load API key using the secure config loader
        try:
            from ..utils.config_loader import get_api_key
            api_key = get_api_key('voyageai')
        except (ImportError, ValueError):
            # Fallback to old method
            load_dotenv() # Load environment variables from .env file
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VoyageAI API key not found. Please set it in config/credentials.json or as VOYAGE_API_KEY environment variable.")
        
        self.client = voyageai.Client(api_key=api_key) # Initialize VoyageAI client with API key


    def loading_model(self, language = 'en'):
        pass
        # not used atm
        # this method could be used to load embeddings which previously have been loaded already..?

    def get_vector(self, sequence):
        import time
        
        print(f"Getting embedding for sequence: {sequence}")
        print(f"Using model: {self.model_name}")
        start_time = time.time()


  

        response = self.client.embed(sequence, model=self.model_name)

        embedding = response.embeddings[0]

        

        print("embedding datatype: ", type(embedding))

        try:#print numpy array datatype
            print("before converting to numpy: embedding datatype: ", embedding.dtype)
        except:
            print("before converting to numpy: could not get numpy datatype")
        

        # changing datatype to np smaller float, to speed up calculation
        # reduce to float32
        embedding = np.array(embedding, dtype=np.float32)
        print("embedding datatype: ", type(embedding))

        try:#print numpy array datatype
            print("after converting to numpy: embedding datatype: ", embedding.dtype)
        except:
            print("after converting to numpy: could not get numpy datatype")


        end_time = time.time()
        response_time = end_time - start_time
        print(f"Response time: {response_time} seconds")
        #shape of the embedding
        print(f"Shape of the embedding: {embedding.shape}")
        #preview content of embedding:
        # print("first element of embedding: ", embedding[0])
        # print("second element of embedding: ", embedding[1])
        # print("third element of embedding: ", embedding[2])

        print(f"First five values of Embedding: {embedding[:5]}")
        print(f"Embedding length: {len(embedding)}")


        return embedding