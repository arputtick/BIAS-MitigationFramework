import torch
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer
)

# Model Configs #
model_configs = {
    "en": "bert-base-uncased",
    "de": "dbmdz/bert-base-german-cased", 
    "nb": "NbAiLab/nb-bert-base",
    "it": "dbmdz/bert-base-italian-uncased",
    "tk": "dbmdz/bert-base-turkish-uncased",
    "nl": "GroNLP/bert-base-dutch-cased",
    "is": "mideind/IceBERT" # This is a RoBERTa-base model
    }

# Model type mapping - specify which models are RoBERTa vs BERT
model_types = {
    "en": "bert",
    "de": "bert",
    "nb": "bert", 
    "it": "bert",
    "tk": "bert",
    "nl": "bert",
    "is": "roberta"  # IceBERT is RoBERTa-based
}

def load_model_and_tokenizer(language_code, occupations):
    model_name = model_configs.get(language_code, "bert-base-uncased") # default to English model
    model_type = model_types.get(language_code, "bert") # default to BERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
    print(f"Using device: {device}")
    
    if model_type == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(occupations), output_hidden_states=True)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(occupations), output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
    return model.to(device), tokenizer, model_name