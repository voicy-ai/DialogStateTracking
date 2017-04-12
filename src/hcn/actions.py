import util
import numpy as np

'''
    Action Templates

    1. 'any preference on a type of cuisine',
    2. 'api_call <party_size> <rest_type>',
    3. 'great let me do the reservation',
    4. 'hello what can i help you with today',
    5. 'here it is ',
    6. 'how many people would be in your party',
    7. "i'm on it",
    8. 'is there anything i can help you with',
    9. 'ok let me look into some options for you',
    10. 'sure is there anything else to update',
    11. 'sure let me find an other option for you',
    12. 'what do you think of this option: ',
    13. 'where should it be',
    14. 'which price range are looking for',
    15. "you're welcome",
    16. 'No choices available that fit your criteria' (custom template)

    [1] : cuisine
    [2] : location
    [3] : party_size
    [4] : rest_type

'''
class ActionTracker():

    def __init__(self, ent_tracker):
        # maintain an instance of EntityTracker
        self.et = ent_tracker
        # get a list of action templates
        self.action_templates = self.action_templates()
        self.action_size = len(self.action_templates)
        # action mask
        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        self.am_dict = {
                '0000' : [ 1,4,6,8,13,14,16 ],
                '0001' : [ 1,4,6,8,13,16 ],
                '0010' : [ 1,4,8,13,14,16 ],
                '0011' : [ 1,4,8,13,16 ],
                '0100' : [ 1,4,6,8,14,16 ],
                '0101' : [ 1,4,6,8,16 ],
                '0110' : [ 1,4,8,14,16 ],
                '0111' : [ 1,4,8,16 ],
                '1000' : [ 4,6,8,13,14,16 ],
                '1001' : [ 4,6,8,13,16 ],
                '1010' : [ 4,8,13,14,16 ],
                '1011' : [ 4,8,13,16 ],
                '1100' : [ 4,6,8,14,16 ],
                '1101' : [ 4,6,8,16 ],
                '1110' : [ 4,8,14,16 ],
                '1111' : [ 2,3,5,7,9,10,11,12,15,16 ]
                }


    def action_mask(self):
        # get context features as string of ints (0/1)
        ctxt_f = ''.join([ str(flag) for flag in self.et.context_features().astype(np.int32) ])

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index-1] = 1.
            return self.am
    
        return construct_mask(ctxt_f)

    def action_templates(self):
        responses = list(set([ self.et.extract_entities(response, update=False) 
            for response in util.get_responses() ]))

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'resto_' in word:
                    word = '<restaurant>'
                template.append(word)
            return ' '.join(template)

        # extract restaurant entities
        return sorted(set([ extract_(response) for response in responses ])) + ['doh! no choices available that fit your criteria.']
