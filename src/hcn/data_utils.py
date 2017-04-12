import util


'''
    Train

    1. Prepare training examples
        1.1 Format 'utterance \t action_template_id\n'
    2. Prepare dev set
    3. Organize trainset as list of dialogues
'''

class Data():

    def __init__(self, entity_tracker, action_tracker):

        self.action_templates = action_tracker.get_action_templates()
        self.et = entity_tracker
        # prepare data
        self.trainset = self.prepare_data()

    def prepare_data(self):
        # get utterances
        utterances = util.get_utterances()
        # get responses
        responses = util.get_responses()
        responses = [ self.get_template_id(response) for response in responses ]

        trainset = []
        for u,r in zip(utterances, responses):
            trainset.append((u,r))

        return trainset


    def get_template_id(self, response):

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'resto_' in word: 
                    if 'phone' in word:
                        template.append('<info_phone>')
                    elif 'address' in word:
                        template.append('<info_address>')
                    else:
                        template.append('<restaurant>')
                else:
                    template.append(word)
            return ' '.join(template)

        return self.action_templates.index(
                extract_(self.et.extract_entities(response, update=False))
                )
