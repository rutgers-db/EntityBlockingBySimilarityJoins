# author: Yunqi Li
# contact: liyunqixa@gmail.com
import gensim.models
from gensim import utils
import joblib
import pandas as pd
import pathlib
from collections import defaultdict
import numpy as np
import py_entitymatching as em
import time
import subprocess
import multiprocessing
import re
from simjoin_entitymatching.value_matcher.utils import DSU, run_cosine_exe

# debug
from py_entitymatching.catalog.catalog import Catalog
import py_entitymatching.catalog.catalog_manager as cm
        

class Doc2Vec:
    '''
    Dov2Vec for attribute: str_bt_1w_5w, str_bt_5w_10w and str_gt_10w
    Generally the value matcher has two usages:
        1. usage 0, label the sample result. 
        2. usage 1, group interchangeable values in match result.
        
        For usage 0, the dataframe is stored at "sample_res"
        For usage 1, the dataframe is stored at "match_res"
        
        The "inmemory" indicates whether we could directly train doc2vec by using 
        the whole table(s) in memory. It will be set as True when the table(s) are
        usually not to large.
        
        Both two usages will share "training_set" as training set if the "inmemory" is False.
    '''

    def __init__(self, inmemory_):
        self.model = None
        self.sample_res = None     # usage 0: label sample res
        self.match_res = None      # usage 1: group ic values in match res
        self.training_set = None   # training set if the inmemory is False
        self.setences = []
        # if the table could fit in memory & small
        # then train on whole table
        self.inmemory = inmemory_
        
        self.cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())


    def _preprocess(self, blk_attr, rawtable, rawtable2=None):
        rawdata = []

        # if the table is small enough in memory
        # we train on whole table
        if self.inmemory == 0:
            lattr = 'ltable_' + blk_attr
            rattr = 'rtable_' + blk_attr

            for _, row in self.sample_res.iterrows():
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
        for i, line in enumerate(rawdata):
            # print(line)
            toks = utils.simple_preprocess(line)
            corpu = gensim.models.doc2vec.TaggedDocument(toks, [i])
            self.setences.append(corpu)


    # sample to get training set for usage 0
    def sample_sample(self, sample_size=5000):
        if len(self.sample_res) > sample_size:
            self.training_set = em.sample_table(self.sample_res, sample_size)
        else:
            print(f"no need for sampling: {len(self.sample_res)}")
            self.training_set = self.sample_res


    # sample to get training set for usage 1
    def sample_match(self, sample_size=5000):
        if len(self.match_res) > sample_size:
            self.sample_res = em.sample_table(self.match_res, sample_size)
        else:
            print(f"no need for sampling: {len(self.match_res)}")
            self.sample_res = self.match_res


    def train_and_save(self, blk_attr, rawtable, rawtable2=None, 
                       default_output_dir=""):
        if self.inmemory == 0:
            self.sample_sample()

        self._preprocess(blk_attr, rawtable, rawtable2)
        
        if default_output_dir == "":
            model_path = "/".join([self.cur_parent_dir, "model", "doc2vec.joblib"])
        else:
            default_output_dir = default_output_dir[ : -1] if default_output_dir[-1] == '/' \
                                                           else default_output_dir
            model_path = "/".join([default_output_dir, "dov2vec.joblib"])

        # train
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
        self.model.build_vocab(self.setences)
        self.model.train(self.setences, total_examples=self.model.corpus_count, 
                         epochs=self.model.epochs)

        # save
        joblib.dump(self.model, model_path)
        print("training for labeling value matcher done")


    def train_all_and_save(self, attrs, rawtable, rawtable2=None, 
                           default_output_dir=""):
        '''
        Train model for all attributes except id
        attrs: attributes that could use word2vec
        '''

        if self.inmemory == 0:
            self.sample_match()

        for attr in attrs:
            print(f"trainging doc2vec on {attr} ...")
            self._preprocess(attr, rawtable, rawtable2)
            self.model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
            self.model.build_vocab(self.setences)
            self.model.train(self.setences, total_examples=self.model.corpus_count, 
                            epochs=self.model.epochs)
            
            if default_output_dir == "":
                model_path = "/".join([self.cur_parent_dir, "model", "doc2vec_" + attr + ".joblib"])
            else:
                default_output_dir = default_output_dir[ : -1] if default_output_dir[-1] == '/' \
                                                               else default_output_dir
                model_path = "/".join([default_output_dir, "doc2vec_" + attr + ".joblib"])
            
            joblib.dump(self.model, model_path)


    def apply_sample(self, blk_attr, tau, ground_truth_label=False, default_icv_dir="", 
                     default_gold_dir="", default_sample_res_dir=""):
        '''
        Apply Doc2Vec for sampling and labeling cand
        '''
        time_start = time.time()

        # apply
        poscnt = 0
        if ground_truth_label is False:
            row_index = list(self.sample_res.index)
            lattr = 'ltable_' + blk_attr
            rattr = 'rtable_' + blk_attr
            
            if default_icv_dir == "":
                vec_path = "/".join([self.cur_parent_dir, "ic_values", "vec_sample.txt"])
                vec_label_path = "/".join([self.cur_parent_dir, "ic_values", "vec_sample_label.txt"])
            else:
                default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                         else default_icv_dir
                vec_path = "/".join([default_icv_dir, "vec_sample.txt"])
                vec_label_path = "/".join([default_icv_dir, "vec_sample_label.txt"])

            with open(vec_path, "w") as vecfile:
                stat = [str(len(row_index)), '\n']
                vecfile.writelines(stat)

                for row in row_index:
                    if pd.isnull(self.sample_res.loc[row, lattr]) == True or \
                    pd.isnull(self.sample_res.loc[row, rattr]) == True:
                        continue

                    lstr = utils.simple_preprocess(self.sample_res.loc[row, lattr])
                    rstr = utils.simple_preprocess(self.sample_res.loc[row, rattr])
                    lvec = self.model.infer_vector(lstr)
                    rvec = self.model.infer_vector(rstr)
                    wlvec = [str(e) + ' ' for e in lvec]
                    wrvec = [str(e) + ' ' for e in rvec]
                    wlvec.insert(0, str(len(lvec)) + ' ')
                    wrvec.insert(0, str(len(rvec)) + ' ')
                    wlvec.append('\n')
                    wrvec.append('\n')
                    vecfile.writelines(wlvec)
                    vecfile.writelines(wrvec)
                    
            # label
            run_cosine_exe(vec_path, vec_label_path, tau)

            # read
            with open(vec_label_path, "r") as labelfile:
                labels = labelfile.readlines()
                if(len(labels) != len(row_index)):
                    raise ValueError(f"Error in vec label file: {len(labels)}, {len(row_index)}")
                for idx, row in enumerate(row_index):
                    if labels[idx] == '1\n':
                        poscnt += 1
                        self.sample_res.loc[row, 'label'] = 1
        else:
            if default_gold_dir == "":
                ground_truth_path = "/".join([self.cur_parent_dir, "..", "..", "output", "buffer", "gold.csv"])
            else:
                default_gold_dir = default_gold_dir[ : -1] if default_gold_dir[-1] == '/' \
                                                           else default_gold_dir
                ground_truth_path = "/".join([default_gold_dir, "gold.csv"])
            ground_truth = pd.read_csv(ground_truth_path)
            gold = set()
            for idx, row in ground_truth.iterrows():
                lid = int(row["id1"])
                rid = int(row["id2"])
                gold.add((lid, rid))
            
            row_index = list(self.sample_res.index)
            for row in row_index:
                lid = int(self.sample_res.loc[row, "ltable_id"])
                rid = int(self.sample_res.loc[row, "rtable_id"])
                label = 1 if (lid, rid) in gold else 0
                self.sample_res.loc[row, "label"] = label
                poscnt = poscnt + 1 if (lid, rid) in gold else poscnt

        # flush
        print(poscnt, len(row_index) - poscnt)
        if default_sample_res_dir == "":
            sample_res_path = "/".join([self.cur_parent_dir, "..", "..", "output", "buffer", "sample_res.csv"])
        else:
            default_sample_res_dir = default_sample_res_dir[ : -1] if default_sample_res_dir[-1] == '/' \
                                                                   else default_sample_res_dir
            sample_res_path = "/".join([default_sample_res_dir, "sample_res.csv"])
        self.sample_res.to_csv(sample_res_path, index=False)

        time_end = time.time()
        apply_time = time_end - time_start
        print(f"apply doc2vec on labeling time: {apply_time}s")


    def _label_and_group(self, tau, cluster, bag_of_words, pair_list, word2id, 
                         default_icv_dir=""):
        '''
        helper function for "group_interchangeable"
        1. call executable to label vecs
        2. group similar docs
        3. report in the desired file

        Args:
            cluster: the dsu;
            bag_of_words: the doc set
            pair_list: matching result
        '''

        # label
        if default_icv_dir == "":
            vec_path = "/".join([self.cur_parent_dir, "ic_values", "vec_interchangeable.txt"])
            vec_label_path = "/".join([self.cur_parent_dir, "ic_values", "vec_interchangeable_label.txt"])
        else:
            default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                     else default_icv_dir
            vec_path = "/".join([default_icv_dir, "vec_interchangeable.txt"])
            vec_label_path = "/".join([default_icv_dir, "vec_interchangeable_label.txt"])
        run_cosine_exe(vec_path, vec_label_path, tau)

        # read
        with open(vec_label_path, 'r') as labelfile:
            labels = labelfile.readlines()
            if(len(labels) != len(pair_list)):
                raise ValueError(f"Error in vec label file: {len(labels)}, {len(pair_list)}")
            for idx, row in enumerate(pair_list):
                if labels[idx] == '1\n':
                    lid = word2id[tuple(row[0])]
                    rid = word2id[tuple(row[1])]
                    cluster.union(lid, rid)

        # group
        group = defaultdict(set)
        bag_of_words = list(bag_of_words)
        for doc in bag_of_words:
            fa = cluster.find(word2id[doc])
            group[bag_of_words[fa]].add(doc)

        return group
    

    def _flush_group_and_cluster(self, blk_attr, ori_grp, ori_clt, 
                                 default_icv_dir=""):
        if default_icv_dir == "":
            grp_path = "/".join([self.cur_parent_dir, "ic_values", "interchangeable_grp_" + blk_attr + ".txt"])
            clt_path = "/".join([self.cur_parent_dir, "ic_values", "interchangeable_clt_" + blk_attr + ".txt"])
        else:
            default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                     else default_icv_dir
            grp_path = "/".join([default_icv_dir, "interchangeable_grp_" + blk_attr + ".txt"])
            clt_path = "/".join([default_icv_dir, "interchangeable_clt_" + blk_attr + ".txt"])

        with open(grp_path, "w") as grp_file:
            print(len(ori_grp), file=grp_file)
            for key, val in ori_grp.items():
                print(key, len(val), file=grp_file)
                if len(val) <= 1:
                    continue
                for v in val:
                    print(v, file=grp_file)

        with open(clt_path, "w") as clt_file:
            print(len(ori_clt), file=clt_file)
            for key, val in ori_clt.items():
                print(key, file=clt_file)
                print(val, file=clt_file)


    def group_interchangeable(self, blk_attr, tau, 
                              default_icv_dir=""):
        '''
        Apply Doc2Vec for grouping interchangeable value in matching result
        '''

        # apply
        lattr = 'ltable_' + blk_attr
        rattr = 'rtable_' + blk_attr
        words_list, pair_list = [], [], []
        ori_doc2pre_doc = defaultdict(set)

        for _, row in self.match_res.iterrows():
            ori_lstr, ori_lid = row[lattr], row["ltable_id"]
            ori_rstr, ori_rid = row[rattr], row["rtable_id"]
            if pd.isnull(ori_lstr) == True or pd.isnull(ori_rstr) == True:
                continue

            lstr = utils.simple_preprocess(ori_lstr)
            rstr = utils.simple_preprocess(ori_rstr)
            ori_doc2pre_doc[tuple(lstr)].add((ori_lid, ori_lstr))
            ori_doc2pre_doc[tuple(rstr)].add((ori_rid, ori_rstr))
            # convert into tuple to hash
            words_list.append(tuple(lstr))
            words_list.append(tuple(rstr))
            pair_list.append((lstr, rstr))

        bag_of_words = set(words_list)
        totins = len(bag_of_words)
        cluster = DSU(totins)
        word2id = { word: idx for idx, word in enumerate(bag_of_words)}
        
        # clustering
        if default_icv_dir == "":
            vec_path = "/".join([self.cur_parent_dir, "ic_values", "vec_interchangeable.txt"])
        else:
            default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                     else default_icv_dir
            vec_path = "/".join([default_icv_dir, "vec_interchangeable.txt"])

        with open(vec_path, "w") as vecfile:
            stat = [str(len(pair_list)), '\n']
            vecfile.writelines(stat)

            for (lstr, rstr) in pair_list:
                lvec = self.model.infer_vector(lstr)
                rvec = self.model.infer_vector(rstr)
                wlvec = [str(e) + ' ' for e in lvec]
                wrvec = [str(e) + ' ' for e in rvec]
                wlvec.insert(0, str(len(lvec)) + ' ')
                wrvec.insert(0, str(len(rvec)) + ' ')
                wlvec.append('\n')
                wrvec.append('\n')
                vecfile.writelines(wlvec)
                vecfile.writelines(wrvec)
        
        group = self._label_and_group(tau, cluster, bag_of_words, pair_list, word2id)

        ori_group = defaultdict(set)
        ori_cluster, print_cluster = {}, {}
        ori_key = 0
        for k, v in group.items():
            # first add key
            ori_set = ori_doc2pre_doc[k]
            for ori_kid, ori_kdoc in ori_set:
                print_cluster[ori_kid] = ori_key
                ori_cluster[ori_kdoc] = ori_key
                ori_kdoc = re.sub(r'[\n\r]', ' ', ori_kdoc)
                ori_group[ori_key].add(ori_kdoc)

            # second add val
            ori_vsets = [ori_doc2pre_doc[vdoc] for vdoc in v]
            for ori_vs in ori_vsets:
                for ori_vid, ori_vdoc in ori_vs:
                    print_cluster[ori_vid] = ori_key
                    ori_cluster[ori_vdoc] = ori_key
                    ori_vdoc = re.sub(r'[\n\r]', ' ', ori_vdoc)
                    ori_group[ori_key].add(ori_vdoc)

            ori_key += 1

        self._flush_group_and_cluster(blk_attr, ori_group, ori_cluster)
        return ori_group, ori_cluster


    def _group_interchangeable(self, blk_attr, tableid, return_dict, 
                               default_match_res_dir=""):
        '''
        worker for grouping each chunk of match result
        '''

        # apply
        lattr = "ltable_" + blk_attr
        rattr = "rtable_" + blk_attr
        words_list, pair_list, vec_list = [], [], []
        ori_doc2pre_doc = defaultdict(set)

        if default_match_res_dir == "":
            partial_name = "/".join([self.cur_parent_dir, "..", "..", "output", "match_res", "match_res" + str(tableid) + ".csv"])
        else:
            default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
                                                                 else default_match_res_dir
            partial_name = "/".join([default_match_res_dir, "match_res" + str(tableid) + ".csv"])    
        partial_match_res = pd.read_csv(partial_name)

        # print(partial_match_res.columns)
        for _, row in partial_match_res.iterrows():
            ori_lstr, ori_lid = row[lattr], row["ltable_id"]
            ori_rstr, ori_rid = row[rattr], row["rtable_id"]
            if pd.isnull(ori_lstr) == True or pd.isnull(ori_rstr) == True:
                continue

            lstr = utils.simple_preprocess(ori_lstr)
            rstr = utils.simple_preprocess(ori_rstr)
            ori_doc2pre_doc[tuple(lstr)].add((ori_lid, ori_lstr))
            ori_doc2pre_doc[tuple(rstr)].add((ori_rid, ori_rstr))
            # convert into tuple to hash
            words_list.append(tuple(lstr))
            words_list.append(tuple(rstr))
            pair_list.append((lstr, rstr))
            # infer vecs
            lvec = self.model.infer_vector(lstr)
            rvec = self.model.infer_vector(rstr)
            vec_list.append((lvec, rvec))

        bag_of_words = set(words_list)

        return_dict[tableid] = (bag_of_words, pair_list, vec_list, ori_doc2pre_doc)


    def group_interchangeable_parallel(self, blk_attr, tau, tottable=100, 
                                       default_icv_dir="", default_match_res_dir=""):
        # parallel
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []
        for i in range(tottable):
            pgroup = multiprocessing.Process(target=self._group_interchangeable, 
                                             args=(blk_attr, i, return_dict, default_match_res_dir))
            processes.append(pgroup)
            pgroup.start()
        # wait
        for pgroup in processes:
            pgroup.join()
            if pgroup.exitcode > 0:
                raise ValueError(f"error in worker: {pgroup.exitcode}")
        # collect result
        bag_of_words = set()
        pair_list, vec_list = [], []
        ori_doc2pre_doc = defaultdict(set)
        for val in return_dict.values():
            bag_of_words = bag_of_words.union(val[0])
            pair_list.extend(val[1])
            vec_list.extend(val[2])
            ori_doc2pre_doc.update(val[3])
        print(f"parallel part done: {len(bag_of_words)}, {len(pair_list)}, {len(vec_list)}")

        # dsu
        totins = len(bag_of_words)
        cluster = DSU(totins)
        word2id = { word: idx for idx, word in enumerate(bag_of_words) }

        # clustering
        if default_icv_dir == "":
            vec_path = "/".join([self.cur_parent_dir, "ic_values", "vec_interchangeable.txt"])
        else:
            default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                     else default_icv_dir
            vec_path = "/".join([default_icv_dir, "vec_interchangeable.txt"])
            
        with open(vec_path, "w") as vecfile:
            stat = [str(len(pair_list)), '\n']
            vecfile.writelines(stat)

            for (lvec, rvec) in vec_list:
                wlvec = [str(e) + ' ' for e in lvec]
                wrvec = [str(e) + ' ' for e in rvec]
                wlvec.insert(0, str(len(lvec)) + ' ')
                wrvec.insert(0, str(len(rvec)) + ' ')
                wlvec.append('\n')
                wrvec.append('\n')
                vecfile.writelines(wlvec)
                vecfile.writelines(wrvec)

        group = self._label_and_group(tau, cluster, bag_of_words, pair_list, word2id)

        ori_group = defaultdict(set)
        ori_cluster, print_cluster = {}, {}
        ori_key = 0
        for k, v in group.items():
            # first add key
            ori_set = ori_doc2pre_doc[k]
            for ori_kid, ori_kdoc in ori_set:
                # if ori_kid in print_cluster:
                #     raise ValueError(f"{ori_kid} has been updated")
                print_cluster[ori_kid] = ori_key
                ori_cluster[ori_kdoc] = ori_key
                ori_kdoc = re.sub(r'[\n\r]', ' ', ori_kdoc)
                ori_group[ori_key].add(ori_kdoc)

            # second add val
            ori_vsets = [ori_doc2pre_doc[vdoc] for vdoc in v]
            for ori_vs in ori_vsets:
                for ori_vid, ori_vdoc in ori_vs:
                    # if ori_vid in print_cluster and print_cluster[ori_vid] != ori_key:
                    #     raise ValueError(f"{ori_vid} has been updated")
                    print_cluster[ori_vid] = ori_key
                    ori_cluster[ori_vdoc] = ori_key
                    ori_vdoc = re.sub(r'[\n\r]', ' ', ori_vdoc)
                    ori_group[ori_key].add(ori_vdoc)

            ori_key += 1

        self._flush_group_and_cluster(blk_attr, ori_group, ori_cluster)
        return ori_group, ori_cluster

    # io
    def load_sample_res(self, tableA, tableB, default_sample_res_dir=""):
        if default_sample_res_dir == "":
            sample_res_path = "/".join([self.cur_parent_dir, "..", "..", "output", "buffer", "sample_res.csv"])
        else:
            default_sample_res_dir = default_sample_res_dir[ : -1] if default_sample_res_dir[-1] == '/' \
                                                                   else default_sample_res_dir
            sample_res_path = "/".join([default_sample_res_dir, "sample_res.csv"])
        self.sample_res = em.read_csv_metadata(file_path=sample_res_path,
                                               key='_id',
                                               ltable=tableA, rtable=tableB, 
                                               fk_ltable='ltable_id', fk_rtable='rtable_id')


    def load_match_res(self, tableA, tableB, default_match_res_dir=""):
        if default_match_res_dir == "":
            match_res_path = "/".join([self.cur_parent_dir, "..", "..", "output", "match_res", "match_res.csv"])
        else:
            default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
                                                                 else default_match_res_dir
            match_res_path = "/".join([default_match_res_dir, "match_res.csv"])
        self.match_res = em.read_csv_metadata(file_path=match_res_path,
                                              key='_id',
                                              ltable=tableA, rtable=tableB, 
                                              fk_ltable='ltable_id', fk_rtable='rtable_id')


    def load_model(self, usage, attr, default_model_dir=""):
        '''
        usage: 0 for labeler and 1 for value matcher
        '''
        if default_model_dir == "":
            model_path = "/".join([self.cur_parent_dir, "model", "doc2vec.joblib"])
            attr_model_path = "/".join([self.cur_parent_dir, "model", "doc2vec_" + attr + ".joblib"])
        else:
            default_model_dir = default_model_dir[ : -1] if default_model_dir[-1] == '/' \
                                                         else default_model_dir
            model_path = "/".join([default_model_dir, "dov2vec.joblib"])
            attr_model_path = "/".join([default_model_dir, "dov2vec_" + attr+ ".joblib"])
            
        if usage == 0:
            self.model = joblib.load(model_path)
        elif usage == 1:
            self.model = joblib.load(attr_model_path)