from enum import Enum


EntType = Enum('Entity Type', '<party_size> <location> <cuisine> <rest_type> <non_ent>')

party_sizes = ['1', '2', '3', '4', '5', '6', '7', '8', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
locations = ['bangkok', 'beijing', 'bombay', 'hanoi', 'paris', 'rome', 'london', 'madrid', 'seoul', 'tokyo'] 
cuisines = ['british', 'cantonese', 'french', 'indian', 'italian', 'japanese', 'korean', 'spanish', 'thai', 'vietnamese']
rest_types = ['cheap', 'expensive', 'moderate']


def ent_type(word):
    if ent in party_sizes:
        return EntType['party_size'].name
    elif ent in locations:
        return EntType['location'].name
    elif ent in cuisines:
        return EntType['cuisine'].name
    elif ent in rest_types:
        return EntType['rest_type'].name
    else:
        return word


def read_content():
    return ' '.join([' '.join(row) for row in read_dialogs()])

def read_dialogs():

    def rm_index(row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    with open('data/dialog-babi-task1-API-calls-trn.txt') as f:
        return [ rm_index(row.split('\t')) for row in  f.read().split('\n') ]


def get_entities():

    def filter_(items):
        return sorted(list(set([ item for item in items if item and '_' not in item ])))

    with open('data/dialog-babi-kb-all.txt') as f:
        return filter_([item.split('\t')[-1] for item in f.read().split('\n') ])

#def ent_extraction(utterance):
#    return ' '.join(

if __name__ == '__main__':
    op = read_file()
    print(op[:5])
    #print(get_entities())

