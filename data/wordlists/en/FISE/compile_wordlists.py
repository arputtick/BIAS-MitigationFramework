import json

path = 'datasets/en/FISE/'

## Class words
rich = ['affluent', 'advantaged','rich','wealthy','moneyed','prosperous','privileged','elite','elites',
                    'aristocrat','aristocrats', 'nobility']

poor = ['poor','destitute', 'needy','disadvantaged','beggar','threadbare','beggars', 'paupers',
                    'impoverished','penniless','unprosperous','homeless']

## Race words
white = ['white', 'whites', 'british','caucasian', 'caucasians',
                     'light-skinned','american', 'americans',
                     'european','europeans','englishman', 'englishmen']

black = ['black', 'blacks', 'negro', 'negroes', 'dark-skinned','negroid',
                     'african', 'africans', 'ethiopian', 'ethiopians', 
                     'african-american','afro-american']

## Gender words
men =  ['man',  'son', 'father', 'brother', 'grandfather','nephew',
                   'male','boy','sir','uncle','gentleman','king']

women = ['woman', 'daughter',  'mother', 'sister', 'grandmother',  'niece', 
                     'female','girl','madam','aunt','maiden','queen']

dict = {}
dict["class"] = {"rich" : rich, "poor": poor}
dict["race"] = {"white": white, "black" : black}
dict['gender'] = {"men" : men, "women": women}

# Save dictionary as json file
with open(path + 'FISE_1/FISE_1.json', 'w') as json_file:
    json.dump(dict, json_file, indent=4)

