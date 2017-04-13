from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from  modules.lstm_net import LSTM_net
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data
import modules.util as util

import numpy as np
import sys


class InteractiveSession():

    def __init__(self):

        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()
        at = ActionTracker(et)

        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128

        self.net = LSTM_net(obs_size=obs_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)


    def interact(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        # begin interaction loop
        while True:

            # get input from user
            u = input(':: ')

            # check if user wants to begin new session
            if u == 'clear' or u == 'reset' or u == 'restart':
                self.net.reset_state()
                et = EntityTracker()
                at = ActionTracker(et)
                print('')

            # check for exit command
            elif u == 'exit' or u == 'stop' or u == 'quit' or u == 'q':
                break

            else:
                # ENTER press : silence
                if not u:
                    u = '<SILENCE>'

                # encode
                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # get action mask
                action_mask = at.action_mask()

                # forward
                prediction = self.net.forward(features, action_mask)
                print('>>', self.action_templates[prediction])


if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
    isess.interact()
