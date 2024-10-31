from numpy import dot
from numpy.linalg import norm
import gensim.models
from gensim import utils
import joblib
import pandas as pd
from gensim.test.utils import get_tmpfile
from collections import defaultdict
import numpy as np
import py_entitymatching as em
import time
import subprocess
import multiprocessing
import re
from utils import DSU

# debug
from py_entitymatching.catalog.catalog import Catalog
import py_entitymatching.catalog.catalog_manager as cm

""" 
The word2vec is under development
Do not include it in current project
This is the very old version, not guarenteed to work
"""


class Word2Vec:
    '''
    Word2Vec for attribute: str_eq_1w or numeric
    But do numeric really needs normalization?
    '''

    def __init__(self, inmemory_):
        self.model = ""
        self.blk_res = ""
        self.match_res = ""
        self.setences = []
        # if the table could fit in memory & small
        # then train on whole table
        self.inmemory = inmemory_


    def _preprocess(self, blk_attr, rawtable, rawtable2=None):
        rawdata = []

        # if the table is small enough in memory
        # we train on whole table
        if self.inmemory == 0:
            lattr = 'ltable_' + blk_attr
            rattr = 'rtable_' + blk_attr

            for _, row in self.blk_res.iterrows():
                lstr = row[lattr]
                rstr = row[rattr]
                if pd.isnull(lstr) == False:
                    rawdata.append(lstr)
                if pd.isnull(rstr) == False:
                    rawdata.append(rstr)

        elif self.inmemory == 1:
            for _, row in rawtable.iterrows():
                if pd.isnull(row[blk_attr]) == False:
                    rawdata.append(row[blk_attr])

            if rawtable2 is not None:
                for _, row in rawtable2.iterrows():
                    if pd.isnull(row[blk_attr]) == False:
                        rawdata.append(row[blk_attr])

        # build corup
        for _, line in enumerate(rawdata):
            sentence = utils.simple_preprocess(line)
            self.setences.append(sentence)

    
    # apis: 1. train & save 2. apply 3. group
    def train_and_save(self, blk_attr, rawtable, rawtable2=None):
        self._preprocess(blk_attr, rawtable, rawtable2)

        # train
        self.model = gensim.models.word2vec.Word2Vec(vector_size=50, min_count=2, epochs=40)
        self.model.build_vocab(self.setences)
        self.model.train(self.setences, total_examples=self.model.corpus_count, 
                         epochs=self.model.epochs)

        # save
        joblib.dump(self.model, 'training/model/word2vec.joblib')


    def train_all_and_save(self, attrs, rawtable, rawtable2=None):
        '''
        Train model for all attributes except id
        attrs: attributes that could use word2vec
        '''

        for attr in attrs:
            print(f"trainging word2vec on {attr} ...")
            self._preprocess(attr, rawtable, rawtable2)
            self.model = gensim.models.word2vec.Word2Vec(vector_size=50, min_count=2, epochs=40)
            self.model.build_vocab(self.setences)
            self.model.train(self.setences, total_examples=self.model.corpus_count, 
                            epochs=self.model.epochs)
            model_name = 'training/model/word2vec_' + attr + '.joblib'
            joblib.dump(self.model, model_name)

    
    def apply_sample(self, blk_attr, tau):
        '''
        Apply Word2Vec for sampling and labeling cand
        '''

        # apply
        poscnt = 0
        row_index = list(self.blk_res.index)
        lattr = 'ltable_' + blk_attr
        rattr = 'rtable_' + blk_attr

        for row in row_index:
            if pd.isnull(self.blk_res.loc[row, lattr]) == True or \
               pd.isnull(self.blk_res.loc[row, rattr]) == True:
                continue
            lstr = utils.simple_preprocess(self.blk_res.loc[row, lattr])
            rstr = utils.simple_preprocess(self.blk_res.loc[row, rattr])
            lvec = self.model.infer_vector(lstr)
            rvec = self.model.infer_vector(rstr)
            cos_sim = dot(lvec, rvec) / (norm(lvec) * norm(rvec))
            
            if cos_sim >= tau:
                poscnt += 1
                self.blk_res.loc[row, 'label'] = 1

        # flush
        print(poscnt, len(row_index) - poscnt)
        # self.blk_res.to_parquet('buffer/cpp_blk_res.parquet', engine='fastparquet', index=False)
        self.blk_res.to_csv('buffer/cpp_blk_res.csv', index=False)


    def group_interchangeable(self, blk_attr, tau):
        '''
        Apply Word2Vec for grouping interchangeable value in blocking result
        '''

        # apply
        lattr = 'ltable_' + blk_attr
        rattr = 'rtable_' + blk_attr

        # dsu
        words_list, pair_list = [], []

        for _, row in self.blk_res.iterrows():
            if pd.isnull(row[lattr]) == True or pd.isnull(row[rattr]) == True:
                continue
            lstr = utils.simple_preprocess(row[lattr])
            rstr = utils.simple_preprocess(row[rattr])
            words_list.append(tuple(lstr))
            words_list.append(tuple(rstr))
            pair_list.append((lstr, rstr))

        bag_of_words = set(words_list)
        totins = len(bag_of_words)
        cluster = DSU(totins)
        id2word = {}
        for idx, word in enumerate(bag_of_words):
            id2word[word] = idx
        
        # clustering
        for (lstr, rstr) in pair_list:
            lvec = self.model.infer_vector(lstr)
            rvec = self.model.infer_vector(rstr)
            cos_sim = dot(lvec, rvec) / (norm(lvec) * norm(rvec))

            if cos_sim >= tau:
                lid = id2word[tuple(lstr)]
                rid = id2word[tuple(rstr)]
                cluster.union(lid, rid)

        # group
        group = defaultdict(list)
        for doc in words_list:
            fa = cluster.find(id2word[doc])
            group[words_list[fa]].append(doc)

        # report
        with open('buffer/interchangeable.txt', 'w') as interfile:
            for val in group.itervalues():
                if(len(val)) <= 1:
                    continue
                interfile.writelines(val)
        interfile.close()


    # io
    def load_blk_res(self, usage):
        '''
        Avoid multi-io
        usage: 0 for labeler & training and 1 for value matcher
        only use this method when training using a sample
        '''
        self.blk_res = pd.read_csv('buffer/cpp_blk_res.csv') if usage == 0 else pd.read_csv('output/blk_res.csv')

    def load_match_res(self):
        self.match_res = pd.read_csv('output/match_res.csv')
    
    def load_model(self):
        self.model = joblib.load('training/model/word2vec.joblib')