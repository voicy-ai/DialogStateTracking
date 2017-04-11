from gensim.models import Word2Vec
import numpy as np


class UtteranceEmbed():

    def __init__(self, fname='text8.model'):
        # load saved model
        self.model = Word2Vec.load(fname)

    def encode(self, utterance):
        embs = [ self.model[word] for word in utterance.split(' ') if word ]
        # average of embeddings
        return np.mean(embs, axis=0)
