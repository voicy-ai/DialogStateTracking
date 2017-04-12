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
