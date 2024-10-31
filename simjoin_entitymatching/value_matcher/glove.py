import numpy as np
from scipy import spatial
import gensim.downloader as api

""" 
The glove is under development
Do not include it in current project
This is the very old version, not guarenteed to work
"""

class Glove:
    '''
    Pre-trained GloVe model
    '''

    def __init__(self):
        self.model = ""


    def _load_model(self):
        # choose from multiple models https://github.com/RaRe-Technologies/gensim-data
        self.model = api.load("glove-wiki-gigaword-50") 


    def _semantic_compare(self, s1, s2):
        ls1 = [i.lower() for i in s1.split()]
        ls2 = [i.lower() for i in s2.split()]

        vs1 = np.sum(np.array([self.model[i] for i in ls1]), axis=0)
        vs2 = np.sum(np.array([self.model[i] for i in ls2]), axis=0)

        sim = spatial.distance.cosine(vs1, vs2)
        
        print(s1, s2, sim)

        if(sim >= 0.5):
            return 1
        return 0

    
    def label_pairs(self, setences1, setences2):
        # load pre-trained model
        self._load_model()

        length = len(setences1)
        Y = np.zeros((length,), dtype=int)

        # label
        for i in range(length):
            s1 = setences1[i]
            s2 = setences2[i]
            Y[i] = self._semantic_compare(s1, s2)

        return Y
