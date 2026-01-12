import json

# Converts a list of traits into a list where each line consists of
# either a single trait if the word doesn't change according to grammatical gender.
# Otherwise, the line consists of a list of two traits, the masculine form followed by the feminine form.
# The list is then saved to a file.
path = 'datasets/it/FISE/'
# lists = ['traitlist_gender_bal.txt', 'occupationlist_gender_bal.txt']
test = 'FISE_IT1'

### Gender balancing for test lists ###
# Load the test data
with open(path + test + '.json', 'r') as f:
    test_data = json.load(f)

# Extract the words for 'class' and 'race'
rich = test_data['class']['rich']
poor = test_data['class']['poor']
white = test_data['race']['white']
black = test_data['race']['black']
words = [rich, poor, white, black]

gg_words = []
for wordlist in words:
    gg_wordlist = []
    # Find all pairs where the difference is either 'o' vs. 'a' or 'e'/'he' vs. 'i'
    wordlist = [word.strip() for word in wordlist]
    for word in wordlist:
        print(word)
        if word[-1] == 'o':
            gg_wordlist.append([word, word[:-1]+'a'])
        elif word[-1] == 'a':
            pass
        elif word[-1] == 'e':
            gg_wordlist.append([word, word[:-1]+'i'])
        elif word[-2:] == 'he':
            gg_wordlist.append([word, word[:-2]+'i'])
        elif word[-1] == 'i':
            pass
        else:
            gg_wordlist.append(word)
    print(gg_wordlist)
    gg_words.append(gg_wordlist)

# Replace the original words in the test_data with gg_words
test_data['class']['rich'] = gg_words[0]
test_data['class']['poor'] = gg_words[1]
test_data['race']['white'] = gg_words[2]
test_data['race']['black'] = gg_words[3]

# Save the test data as a new json file 'GG_FISE_IT1.json'
with open(path + test + '_GG.json', 'w') as f:
    json.dump(test_data, f, indent=4)

# ### Gender balancing the trait lists ###
# for list in lists:
#     list = path + list
#     # Load the trait list
#     with open(list, 'r') as f:
#         traits = f.readlines()

#     # Find all pairs where the difference is either 'o' vs. 'a' or 'ore' vs. 'rice'
#     traitlist = []
#     for trait in traits:
#         trait = trait.strip()
#         if trait[-1] == 'o':
#             traitlist.append([trait, trait[:-1]+'a'])
#         elif trait[-1] == 'a':
#             pass
#         elif trait[-3:] == 'ore':
#             traitlist.append([trait, trait[:-3]+'rice'])
#         elif trait[-4:] == 'rice':
#             pass
#         else:
#             traitlist.append(trait)
#     print(traitlist)
#     # Save the trait list
#     with open(list.replace('.txt', '_gg.txt'), 'w') as f:
#         for trait in traitlist:
#             f.write(str(trait) + '\n')

# ### Checking gender balancing of stim lists ###
# # Sort ingressivity and valance stim alphabetically
# stim_lists = ['ingressivitystim_gender_bal.csv', 'valencestim_gender_bal.csv']
# import pandas as pd

# # Load as dataframes, with the first row as the header
# ingressivity = pd.read_csv(path + stim_lists[0], header=0)
# valence = pd.read_csv(path + stim_lists[1], header=0)

# # Remove all spaces from every value in the columns
# ingressivity = ingressivity.applymap(lambda x: x.strip())
# valence = valence.applymap(lambda x: x.strip())

# # Separately sort columns alphabetically
# ingressive = ingressivity['Ingressivo'].sort_values().reset_index(drop=True)
# congressive = ingressivity['Congressivo'].sort_values().reset_index(drop=True)
# positive = valence['Positivo'].sort_values().reset_index(drop=True)
# negative = valence['Negativo'].sort_values().reset_index(drop=True)

# # Reassign the sorted columns
# ingressivity['Ingressivo'] = ingressive
# ingressivity['Congressivo'] = congressive
# valence['Positivo'] = positive
# valence['Negativo'] = negative

# # Save the sorted dataframes
# ingressivity.to_csv(path + 'ingressivitystim_gender_bal.csv', index=False)
# valence.to_csv(path + 'valencestim_gender_bal.csv', index=False)
