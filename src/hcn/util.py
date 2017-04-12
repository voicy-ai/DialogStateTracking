def read_content():
    return ' '.join(get_utterances())

def read_dialogs():

    def rm_index(row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    def filter_(dialogs):
        filtered_ = []
        for row in dialogs:
            if row[0][:6] != 'resto_' and row[0]:
                filtered_.append(row)
        return filtered_

    with open('data/dialog-babi-task5-full-dialogs-trn.txt') as f:
        return filter_([ rm_index(row.split('\t')) for row in  f.read().split('\n') ])


def get_utterances():
    dialogs = read_dialogs()
    return [ row[0] for row in dialogs ]

def get_responses():
    dialogs = read_dialogs()
    return [ row[1] for row in dialogs ] 


def get_entities():

    def filter_(items):
        return sorted(list(set([ item for item in items if item and '_' not in item ])))

    with open('data/dialog-babi-kb-all.txt') as f:
        return filter_([item.split('\t')[-1] for item in f.read().split('\n') ])


def action_templates():
    ent_tracker_ = ent_tracker.EntityTracker()
    responses = list(set([ ent_tracker_.extract_entities(response, update=False) 
        for response in get_responses() ]))

    def extract_(response):
        template = []
        for word in response.split(' '):
            if 'resto_' in word:
                word = '<restaurant>'
            template.append(word)
        return ' '.join(template)

    # extract restaurant entities
    return sorted(set([ extract_(response) for response in responses ])) + ['doh! no choices available that fit your criteria.']
