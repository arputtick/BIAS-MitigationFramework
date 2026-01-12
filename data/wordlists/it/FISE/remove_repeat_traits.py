# load traitlists from FISE and remove repeated traits

path = 'datasets/it/FISE/'
traitlist = 'traitlist_gender_bal.txt'
occupationlist = 'occupationlist.txt'

# with open(path + traitlist, 'r') as f:
#     traits = f.readlines()
#     traits = [t.strip() for t in traits]
#     # remove repeated traits
#     traits = list(set(traits))
#     # sort alphabetically
#     traits.sort()

with open(path + occupationlist, 'r') as f:
    occupations = f.readlines()
    occupations = [o.strip() for o in occupations]
    # remove repeated occupations
    occupations = list(set(occupations))
    # sort alphabetically
    occupations.sort()

# delete old content and overwrite the traitlists
# with open(path + traitlist, 'w') as f:
#     for t in traits:
#         f.write(t + '\n')

with open(path + occupationlist, 'w') as f:
    for o in occupations:
        f.write(o + '\n')