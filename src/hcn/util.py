def read_content():
    return ' '.join(get_utterances())

def read_dialogs(with_indices=False):

    def rm_index(row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    def filter_(dialogs):
        filtered_ = []
        for row in dialogs:
            if row[0][:6] != 'resto_':
                filtered_.append(row)
        return filtered_

    with open('data/dialog-babi-task5-full-dialogs-trn.txt') as f:
        dialogs = filter_([ rm_index(row.split('\t')) for row in  f.read().split('\n') ])
        # organize dialogs -> dialog_indices
        prev_idx = -1
        n = 1
        dialog_indices = []
        updated_dialogs = []
        for i, dialog in enumerate(dialogs):
            if not dialogs[i][0]:
                dialog_indices.append({
                    'start' : prev_idx + 1,
                    'end' : i - n + 1
                })
                prev_idx = i-n
                n += 1
            else:
                updated_dialogs.append(dialog)        

        if with_indices:
            return updated_dialogs, dialog_indices[:-1]

        return updated_dialogs


def get_utterances(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs()
    return [ row[0] for row in dialogs ]

def get_responses(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs()
    return [ row[1] for row in dialogs ] 


def get_entities():

    def filter_(items):
        return sorted(list(set([ item for item in items if item and '_' not in item ])))

    with open('data/dialog-babi-kb-all.txt') as f:
        return filter_([item.split('\t')[-1] for item in f.read().split('\n') ])
