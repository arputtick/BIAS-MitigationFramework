import os
import json

# Compile LPBS lists from WEAT lists

path = "../data/wordlists/"
languages = ["de", "it", "nl", "is", "tk"]

for lang in languages:
        print(f"Processing language: {lang}")
        path_lang = path + lang + "/"
        weat_path = path_lang+"WEAT/"
        lpbs_path = path_lang+"LPBS/"
        templates = ['TARGET ATTRIBUTE']

        # Go through all files in the directory
        for test in os.listdir(weat_path):
                # Only take text files
                if not test.endswith(".txt"):
                        continue
                new_data = {
                        "targ1" : {
                                "singular" : [],
                                "plural" : [],
                        },
                        "targ2" : {
                                "singular" : [],
                                "plural" : [],
                        },
                        "attr1" : [],
                        "attr2" : [],
                        "templates" : {
                                "singular" : [],
                                "plural" : [],
                        }
                }
                print(test)
                testfile = open(weat_path+test)
                lines = testfile.read().split('\n')
                filtered_lines = [line for line in lines if not line.startswith('#')]
                dataset_obj = {}
                dataset_obj["Target1"] = filtered_lines[0].split(",")
                dataset_obj["Target2"] = filtered_lines[1].split(",")
                dataset_obj["Attribute1"] = filtered_lines[2].split(",")
                dataset_obj["Attribute2"] = filtered_lines[3].split(",")

                new_data["targ1"]["singular"] = dataset_obj["Target1"]
                new_data["targ2"]["singular"] = dataset_obj["Target2"]
                new_data["attr1"] = dataset_obj["Attribute1"]
                new_data["attr2"] = dataset_obj["Attribute2"]
                new_data["templates"]["singular"] = templates

                # Writing the new content to a JSONL file
                # Ensure the LPBS directory exists
                os.makedirs(lpbs_path, exist_ok=True)
                with open(lpbs_path+"LPBS_"+test.replace(".txt", ".jsonl"), 'w', encoding='utf-8') as file:
                        json.dump(new_data, file, indent=4, ensure_ascii=False)