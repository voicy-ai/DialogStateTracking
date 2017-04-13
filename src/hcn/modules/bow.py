import modules.util as util
import numpy as np


'''
    Bag of Words
    
    1. Get a list of words from training set
    2. Get vocabulary, vocab_size
    3. Build bag of words for each utterance
        3.1 iterate through words
        3.2 pick id from vocab
        3.3 set index as 1 in ndarray
'''

class BoW_encoder():

    def __init__(self):
        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab)

    def get_vocab(self):
        content = util.read_content()
        vocab = sorted(set(content.split(' ')))
        # remove empty strings
        return [ item for item in vocab if item ]

    def encode(self, utterance):
        bow = np.zeros([self.vocab_size], dtype=np.int32)
        for word in utterance.split(' '):
            if word in self.vocab:
                idx = self.vocab.index(word)
                bow[idx] += 1
        return bow
