### Script to create the sentence templates for Italian based on the existing word lists
#########################################

import json
import os
import re

test_path = 'datasets/it/WEAT/'
# Iterate through the text files in test_path
tests = []
for file in os.listdir(test_path):
    if file.endswith('.txt'):
      # if file.startswith('IT_5'):
        name = os.path.splitext(file)[0]
        tests.append(name)
print(tests)

# Templates for the sentences
templates_nouns_singular=[
"WORD è qui.",
"WORD è lì.",
]

templates_nouns_plural=[
"WORD sono qui.",
"WORD sono lì.",
]

templates_adjectives = [
"È WORD.",
"Era WORD.",
]

templates_verbs = [
"Questo può WORD.",
"Questa può WORD.",
]
      
# Load the spacy model
import spacy
nlp = spacy.load("it_core_news_sm")

# Words that are not correctly parsed
exceptions_nouns_f = ["elettrica", "meccanica", "informatica", "fisica", "veterinaria", "nonna", "ragazza", "zia", 
                      "razionalità", "sensibilità", "religiosità", "disponibilità", "società", "algebra", "pace","risata", "chimica"]
exceptions_nouns_m = ["dirigente", "management", "business", "CEO", "generale", "artigiano", "impiegato", "comesso", "mietitore", "conducente"]
exceptions_nouns_pl_f = ["bambine"]
exceptions_nouns_pl_m = ["parenti"]
exceptions_nouns = exceptions_nouns_f + exceptions_nouns_m + exceptions_nouns_pl_f + exceptions_nouns_pl_m
exceptions_nouns_sing = exceptions_nouns_f + exceptions_nouns_m
exceptions_nouns_pl = exceptions_nouns_pl_f + exceptions_nouns_pl_m

exceptions_prop_nouns = ["Simone", "Elisa", "Matteo", "Sunil", "Raju", "Manoj", "Anita", "Sunita", "Priyanka", "Alexandra", "Puja"]
exceptions_adj = ["male","trans","transgender", "etero", "cis", "cisgender", "queer"]

exceptions = exceptions_nouns + exceptions_prop_nouns + exceptions_adj

def get_article(word):
      '''
      Retrieves the correct article for a given noun.
      '''
      # print(word)
      article = ""
      doc = nlp(word)
      # print(doc,doc[0].pos_, doc[0].morph.get("Number"), doc[0].morph.get("Gender"))
      word_beg = word[:1]

      # Rules
      beginnings_1 = ["z", "x", "y"]
      beginnings_2 =  ["gn", "ps", "pn"]
      # reg_ex for "s" + consonant
      reg_ex = re.compile(r"s[^aeiou]")
      # reg_ex for "i" + vowel
      reg_ex2 = re.compile(r"i[aeiou]")
      # reg_ex for consonant
      reg_ex3 = re.compile(r"[^aeiou]")
      # reg_ex for vowel
      reg_ex4 = re.compile(r"[aeiou]")

      if doc[0].morph.get("Number") or word in exceptions_nouns:
            if doc[0].morph.get("Number") and doc[0].morph.get("Number")[0] == "Sing" or word in exceptions_nouns_sing:
                  if doc[0].morph.get("Gender") or word in exceptions_nouns:
                  # Masculine singular
                        if doc[0].morph.get("Gender") and doc[0].morph.get("Gender")[0] == "Masc" or word in exceptions_nouns_m:
                              if word[0] in beginnings_1 or word_beg in beginnings_2 or reg_ex.match(word_beg) or reg_ex2.match(word_beg):
                                    article = "uno "
                              else:
                                    article = "un "
                  # Feminine singular
                        elif doc[0].morph.get("Gender") and doc[0].morph.get("Gender")[0] == "Fem" or word in exceptions_nouns_f:
                              if reg_ex.match(word_beg) or reg_ex3.match(word_beg[0]):
                                    article = "una "
                              else:
                                    article = "un'"
                  else:
                        print('No gender found for word: '+word)

            elif doc[0].morph.get("Number")[0] == "Plur" or word in exceptions_nouns_pl:
            # Masculine plural
                  if doc[0].morph.get("Gender") and doc[0].morph.get("Gender")[0] == "Masc" or word in exceptions_nouns_pl_m:
                        if word[0] in beginnings_1 or word[0] == 'h' or word_beg in beginnings_2 or reg_ex.match(word_beg) or reg_ex2.match(word_beg) or reg_ex4.match(word_beg[0]):
                              article = "gli "
                        else:
                              article = "i "
            # Feminine plural
                  elif doc[0].morph.get("Gender") and doc[0].morph.get("Gender")[0] == "Fem" or word in exceptions_nouns_pl_f:
                        article = "le "
                  else:
                        print('No gender found for word: '+word)
      else:
            print('No number found for word:'+word)
      # print(article + word)
      return article + word


def check_parse(word):
      '''
      Checks the parsing of a word.
      '''
      doc = nlp(word)
      print("Parse:", doc,doc[0].pos_, doc[0].morph.get("Number"), doc[0].morph.get("Gender"))

# Create the new SEAT files
for testname in tests:
      new_data = {
            "line1": {
                  "category": "TODO",
                  "examples": []},
            "line2": {
                  "category": "TODO",
                  "examples": []},

            "line3": {
                  "category": "TODO",
                  "examples": []},
            "line4": {
                  "category": "TODO",
                  "examples": []}
      }
      path = "datasets/it/WEAT/" + testname + ".txt"
      testfile = open(path, 'r')
      lines = testfile.read().split('\n')
      filtered_lines = [line for line in lines if not line.startswith('#')]
      counter = 0
      for line in filtered_lines:
            counter = counter + 1
            examples = []
            words_base = line.split(",")
            # Remove all exceptions from words
            words = [word for word in words_base if word not in exceptions]

            # Build templates for the remaining words
            print('Building templates for the following words:', words)
            for word in words:
                  doc = nlp(word)
                  # Ensure the word is correctly parsed, if not, add to exceptions
                  check_parse(word)
                  if doc[0].pos_ == "NOUN":
                        word = get_article(word)
                        if doc[0].morph.get("Gender"):
                              if doc[0].morph.get("Number")[0] == "Sing":
                                    for template in templates_nouns_singular:
                                          examples.append(template.replace("WORD",word))
                              else:
                                    for template in templates_nouns_plural:
                                          examples.append(template.replace("WORD",word))
                        else:
                              print("No doc[0].morph.get(Gender) for word: "+word)
                  if doc[0].pos_ == "ADJ":
                        for template in templates_adjectives:
                              examples.append(template.replace("WORD", word))
                  if doc[0].pos_ == "PROPN":
                        for template in templates_nouns_singular:
                              examples.append(template.replace("WORD", word))
                  if doc[0].pos_ == "VERB":
                        for template in templates_verbs:
                              examples.append(template.replace("WORD", word))

            # Build templates for exceptions that are in words_base
            test_exceptions = [word for word in words_base if word in exceptions]
            print('Building templates for the following exceptions:', test_exceptions)
            for word in test_exceptions:
                  if word in exceptions_nouns:
                        word_art = get_article(word)
                        if word in exceptions_nouns_sing:
                              for template in templates_nouns_singular:
                                    examples.append(template.replace("WORD",word_art))
                        else:
                              for template in templates_nouns_plural:
                                    examples.append(template.replace("WORD",word_art))
                  elif word in exceptions_prop_nouns:
                        for template in templates_nouns_singular:
                              examples.append(template.replace("WORD", word))
                  elif word in exceptions_adj:
                        for template in templates_adjectives:
                              examples.append(template.replace("WORD", word))
                  else:
                        print("Word not found in exceptions: "+word)
                  

            new_data["line"+str(counter)]["examples"]=examples
      
      # Uppercase the first letter of the sentences
      # print(new_data)
      for i in range(1,5):
            new_data["line"+str(i)]["examples"] = [example.capitalize() for example in new_data["line"+str(i)]["examples"]]


      new_data["targ1"] = new_data['line1']
      del new_data['line1']
      new_data["targ2"] = new_data['line2']
      del new_data['line2']
      new_data["attr1"] = new_data['line3']
      del new_data['line3']
      new_data["attr2"] = new_data['line4']
      del new_data['line4']

      # Writing the new content to a JSONL file
      with open("datasets/it/SEAT/SEAT_"+testname+".jsonl", 'w', encoding='utf-8') as file:
            json.dump(new_data, file, indent=4, ensure_ascii=False)