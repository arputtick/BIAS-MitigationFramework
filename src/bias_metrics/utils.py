from .models import FasttextModel, Word2VecModel, GloveModel, BertModel, OpenAiModel, VoyageModel, GPTModel
from .metrics import WEAT, LPBS, CrowS_Pairs
import json
import pandas as pd
import os

from . import config

# Folder to the models
folderpath = config.FOLDER_PATH
fasttextfile = config.FASTTEXT_FILE
word2vecfile = config.WORD2VEC_FILE
glovefile = config.GLOVE_FILE
mbertfile = config.mBERT_FILE

### This method initializes the model based on the embedding type
def initialize_model(embedding, folderpath, modelname, language):
    # Initialize the model
    if embedding == "word2vec":
        model = Word2VecModel(folderpath+word2vecfile)
    elif embedding == "glove":
        model = GloveModel(folderpath+glovefile)
    elif embedding == "fasttext":
        model = FasttextModel(folderpath+fasttextfile.replace("XX", language))
    
    # Bert models
    elif embedding == "bert":
        model = BertModel(folderpath+modelname)
    elif embedding == "bert_pooling":
        model = BertModel(folderpath+modelname, embedding = 'pooling')
    elif embedding == "bert_first":
        model = BertModel(folderpath+modelname, embedding = 'first')
    elif embedding == "mBERT":
        model = BertModel(folderpath+mbertfile)

    # GPT models
    elif embedding == "gpt":
        model = GPTModel(folderpath+modelname)
    elif embedding == "gpt_pooling":
        model = GPTModel(folderpath+modelname, embedding = 'pooling')
    elif embedding == "gpt_first":
        model = GPTModel(folderpath+modelname, embedding = 'first')

    # API models
    elif embedding == "OpenAI_large":
        model = OpenAiModel(model_name = "text-embedding-3-large")
    elif embedding == "OpenAI_small":
        model = OpenAiModel(model_name = "text-embedding-3-small")
    elif embedding == "mVoyage":
        model = VoyageModel(model_name = "voyage-multilingual-2")
    return model


### This method checks if every word in the dataset is in the vocabulary of the model
def check_test_words(combination):
    '''Each combination has the form (test_name, metric, embedding, language, modelname)'''
    dataset, metric, embedding, language, modelname = combination

    # Initialize the model
    model = initialize_model(embedding, folderpath, modelname, language)

    print(f"\nChecking vocabulary for {dataset} with {metric} and {embedding} for language {language} with model {modelname}...\n")
    
    # Currently only implemented for BERT models
    if 'bert' not in embedding.lower():
        print(f'\nVocabulary check is not available for modeltype {embedding}.\n')
        return

    # LOAD THE MODEL
    # Originally this exception is because the default output for the model was [hidden_states], whereas for English is was [logits, hidden_states]
    # hidden_states = False loads with AutoModelForMaskedLM, which outputs [logits]
    # ALTERNATIVE: Use the same model loader for all languages and compute logits from hidden_states.
    if language in ("no","fr"):
        if metric == "LPBS" or metric == "CrowS_Pairs":
            model.loading_model(language, hidden_states = False)
        else:
            model.loading_model(language)
    else:
        model.loading_model(language=language)
    
    if metric == "WEAT":
        # Load the selected dataset from file
        testfile = open('data/wordlists/'+language+'/WEAT/'+dataset+'.txt', 'r')
        lines = testfile.read().split('\n')
        filtered_lines = [line for line in lines if not line.startswith('#')]
        words = filtered_lines[0].split(",") + filtered_lines[1].split(",") + filtered_lines[2].split(",") + filtered_lines[3].split(",")
        for word in words:
            word = word.replace(".","")
            model.check_vocab(word)

    elif metric == "SEAT":
        if "genSEAT" in dataset:
            f = open('data/wordlists/' + language + '/genSEAT/' + dataset + '.jsonl', 'r')
        else:
            f = open('data/wordlists/'+language+'/SEAT/'+dataset+'.jsonl', 'r')
        testfile = json.load(f)

        unique_words = set()
        for category in testfile.values():
            examples = category.get('examples', [])
            for sentence in examples:
                words = sentence.split()
                unique_words.update(words)
        for word in unique_words:
            word = word.replace(".","")
            model.check_vocab(word)
    return


###
### This method evaluates each experiment passed in a "combinations" structure.
###
def evaluate_combinations(combinations, calc_pvalue_bool, p_value_iterations, full_permut_bool):
    """Each combination has the form (dataset, metric, embedding, language, modelname)"""
    results = {}
    failed_tests = []
    
    for dataset, metric, embedding, language, modelname in combinations:
        print(f"\nEvaluating {dataset} with {metric} and {embedding} for language {language} with model {modelname}...\n")
        
        try:
            # Initialize the model
            model = initialize_model(embedding, folderpath, modelname, language)

            # LOAD THE MODEL
            # Originally this exception is because the default output for the model was [hidden_states], whereas for English is was [logits, hidden_states]
            # hidden_states = False loads with AutoModelForMaskedLM, which outputs [logits]
            # ALTERNATIVE: Use the same model loader for all languages and compute logits from hidden_states.
            if language != "en":
                if metric == "LPBS" or metric == "CrowS_Pairs":
                    model.loading_model(language, hidden_states = False)
                else:
                    model.loading_model(language)
            else:
                model.loading_model(language=language)

            ### Case of using WEAT ####
            if metric == "WEAT":
                # Initialize the WEAT tester
                if embedding == "bert_pooling" or embedding == "bert_first" or embedding == "gpt_pooling" or embedding == "gpt_first":
                    weat_tester = WEAT(model, enc = 'token-level')
                else:
                    weat_tester = WEAT(model)
                
                # Load the selected dataset from file
                testfile = open('../data/wordlists/'+language+'/WEAT/'+dataset+'.txt', 'r')
                lines = testfile.read().split('\n')
                filtered_lines = [line for line in lines if not line.startswith('#')]
                dataset_obj = {}
                dataset_obj["Target1"] = filtered_lines[0].split(",")
                dataset_obj["Target2"] = filtered_lines[1].split(",")
                dataset_obj["Attribute1"] = filtered_lines[2].split(",")
                dataset_obj["Attribute2"] = filtered_lines[3].split(",")
                #print("Dataset_Obj:",dataset_obj)
                # Evaluate and store the result
                result = weat_tester.evaluate(dataset_obj, calc_pvalue_bool, p_value_iterations, full_permut_bool)
                if modelname:
                    results[(dataset, metric, embedding, language, modelname)] = result
                else:
                    results[(dataset, metric, embedding, language)] = result

            ### Case of using SEAT ###
            elif metric == "SEAT":
                # Initialize the SEAT tester
                if embedding == "bert_pooling" or embedding == "bert_first" or embedding == "gpt_pooling" or embedding == "gpt_first":
                    seat_tester = WEAT(model, enc = 'token-level')
                else:
                    seat_tester = WEAT(model)

                ## special case genSEAT
                if "genSEAT" in dataset:
                    f = open('data/wordlists/' + language + '/genSEAT/' + dataset + '.jsonl', 'r')
                else:
                    f = open('data/wordlists/'+language+'/SEAT/'+dataset+'.jsonl', 'r')
                testfile = json.load(f)

                dataset_obj = {}
                dataset_obj["Target1"] = testfile["targ1"]["examples"]
                dataset_obj["Target2"] = testfile["targ2"]["examples"]
                dataset_obj["Attribute1"] = testfile["attr1"]["examples"]
                dataset_obj["Attribute2"] = testfile["attr2"]["examples"]
                #print("Dataset_Obj:",dataset_obj)

                # Evaluate and store the result
                result = seat_tester.evaluate(dataset_obj, calc_pvalue_bool, p_value_iterations, full_permut_bool)
                if modelname:
                    results[(dataset, metric, embedding, language, modelname)] = result
                else:
                    results[(dataset, metric, embedding, language)] = result

            ### Case of using LPBS ###
            elif metric == "LPBS":
                # Initialize the LPBS tester
                LPBS_tester = LPBS(model)

                f = open('../data/wordlists/'+language+'/LPBS/'+dataset+'.jsonl', 'r')
                testfile = json.load(f)

                dataset_obj = {}
                dataset_obj["Target1"] = testfile["targ1"]
                dataset_obj["Target2"] = testfile["targ2"]
                dataset_obj["Attribute1"] = testfile["attr1"]
                dataset_obj["Attribute2"] = testfile["attr2"]
                dataset_obj["Templates"] = testfile["templates"]
                #print("Dataset_Obj:",dataset_obj)

                # Evaluate and store the result
                result = LPBS_tester.evaluate(dataset_obj, calc_pvalue_bool, p_value_iterations, full_permut_bool)
                if modelname:
                    results[(dataset, metric, embedding, language, modelname)] = result
                else:
                    results[(dataset, metric, embedding, language)] = result

            ### Case of using CrowS Metric ###
            elif metric == "CrowS_Pairs":
                # Initialize the CrowS_Pairs tester
                crows_pairs_tester = CrowS_Pairs(model)

                if language == "fr":
                    df = pd.read_csv('data/wordlists/'+language+'/CROWS/'+dataset+'.csv', sep='\t', encoding='utf-8')
                else:
                    df = pd.read_csv('data/wordlists/'+language+'/CROWS/'+dataset+'.csv')

                result = crows_pairs_tester.evaluate(df)
                results[(dataset, metric, embedding, language, modelname)] = result
            
            print(f"✓ Successfully evaluated {dataset} with {metric}")
                
        except Exception as e:
            error_msg = f"Failed to evaluate {dataset} with {metric} and {embedding} for language {language}: {str(e)}"
            print(f"✗ {error_msg}")
            failed_tests.append(error_msg)
            # Continue with next test instead of failing completely

    # Print summary of results
    if failed_tests:
        print(f"\n⚠️  WARNING: {len(failed_tests)} test(s) failed:")
        for failure in failed_tests:
            print(f"  - {failure}")
        print(f"✓ Successfully completed {len(results)} test(s)")
    else:
        print(f"✓ All {len(results)} tests completed successfully")
    
    return results