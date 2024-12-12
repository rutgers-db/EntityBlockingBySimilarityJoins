# author: Yunqi Li
# contact: liyunqixa@gmail.com
'''
perform k-nearest-neighbors search for match results
based on semantic similarity
'''
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Literal
import faiss
import time
import gensim.utils

from simjoin_entitymatching.value_matcher.doc2vec import Doc2Vec
import simjoin_entitymatching.utils.path_helper as ph


'''
the match table ("attr" is the representative):
_id    ltable_id   rtable_id   ltable_attr    rtable_attr   ......

Query set Q: ltable_id
Index set I: rtable_id
'''


def _get_word_embeddings(match_tab, attr, default_vmatcher_dir=""):
    doc2vec = Doc2Vec(inmemory_=0)
    doc2vec.load_model(usage=1, attr=attr, default_model_dir=default_vmatcher_dir)
    vec_dict = doc2vec.infer_vectors_for_df(attr, match_tab)
    
    l_schema = "ltable_" + attr
    r_schema = "rtable_" + attr
    
    # bucket
    buck = defaultdict(list)
    rev_buck = defaultdict(list)
    query_vec = defaultdict()
    rev_query_vec = defaultdict()
    
    prev_dim = 0
    
    for idx, row in match_tab.iterrows():
        lid, lstr = row["ltable_id"], row[l_schema]
        rid, rstr = row["rtable_id"], row[r_schema]
        
        # check
        if lstr not in vec_dict.keys():
            raise KeyError(f"the dictionary does not contain : {lstr}")
        if rstr not in vec_dict.keys():
            raise KeyError(f"the dictionary does not contain : {rstr}")
        
        # vec dimension check
        l_vec = vec_dict[lstr]
        r_vec = vec_dict[rstr]
        l_dim = l_vec.shape[0]
        r_dim = r_vec.shape[0]
        
        if idx != 0 and (l_dim != r_dim or l_dim != prev_dim):   
            raise ValueError(f"error in vector's dimension : {l_dim}, {r_dim}, {prev_dim}")
        prev_dim = l_dim
        
        # table a -> table b
        buck[lid].append((rid, r_vec))
        query_vec[lid] = l_vec
        
        # table b -> table a
        rev_buck[rid].append((lid, l_vec))
        rev_query_vec[rid] = r_vec
        
    return buck, query_vec, rev_buck, rev_query_vec, prev_dim, doc2vec.model


def _search_KNN(dimension, K, buck, query_vec, search_strategy=Literal["exact", "approximate"]):
    nn_res = defaultdict(set)
    nn_dis = defaultdict(list)
    
    for q_id, indices in buck.items():
        q_vec = query_vec[q_id]
        i_id, i_vec = [], []
        for index in indices:
            i_id.append(index[0])
            i_vec.append(index[1])
            
        # prepare
        if len(i_id) <= K:
            nn_res[q_id] = set(i_id)
            nn_dis[q_id] = [1.0 for i in range(len(i_id))]
        
        index_data = np.array(i_vec).astype('float32')
        query_data = np.array([q_vec.flatten().tolist()]).astype('float32')
    
        # normalize
        faiss.normalize_L2(index_data)
        faiss.normalize_L2(query_data)
    
        # index
        if search_strategy == "exact":
            search_index = faiss.IndexFlatIP(dimension)
        else:
            search_index = faiss.index_factory(dimension, "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
        search_index.add(index_data)
    
        # query
        distances, indices = search_index.search(query_data, K)
    
        nn_res[q_id] = set([i_id[i] for i in indices.flat])
        nn_dis[q_id] = [d for d in distances.flat]
        
    return nn_res, nn_dis


def _slim_match_tab(match_tab, nn_res, model, rep_attr, threshold, is_rev=False):
    row_index = match_tab.index
    l_schema = "ltable_" + rep_attr
    r_schema = "rtable_" + rep_attr
    
    if not is_rev:
        for ridx in row_index:
            lid = match_tab.loc[ridx, "ltable_id"]
            rid = match_tab.loc[ridx, "rtable_id"]
            
            # check original similarity score
            l_title = match_tab.loc[ridx, l_schema]
            r_title = match_tab.loc[ridx, r_schema]
            l_docs = gensim.utils.simple_preprocess(l_title)
            r_docs = gensim.utils.simple_preprocess(r_title)
            l_vec = model.infer_vector(l_docs)
            r_vec = model.infer_vector(r_docs)
            cos_sim = np.dot(l_vec, r_vec) / (np.linalg.norm(l_vec) * np.linalg.norm(r_vec))
            
            if rid not in nn_res[lid]:
                match_tab.loc[ridx, "predicted"] = 0
    else:
        for ridx in row_index:
            lid = match_tab.loc[ridx, "ltable_id"]
            rid = match_tab.loc[ridx, "rtable_id"]
            
            # check original similarity score
            l_title = match_tab.loc[ridx, l_schema]
            r_title = match_tab.loc[ridx, r_schema]
            l_docs = gensim.utils.simple_preprocess(l_title)
            r_docs = gensim.utils.simple_preprocess(r_title)
            l_vec = model.infer_vector(l_docs)
            r_vec = model.infer_vector(r_docs)
            cos_sim = np.dot(l_vec, r_vec) / (np.linalg.norm(l_vec) * np.linalg.norm(r_vec))
            
            if lid not in nn_res[rid]:
                match_tab.loc[ridx, "predicted"] = 0
        
        # concat with negative predictions after the reverse slim     
        neg_match_tab = pd.read_csv("output/exp/neg_match_res0.csv")
        match_tab = pd.concat([match_tab, neg_match_tab], ignore_index=True)

    return match_tab


def _check_similarity(nn_dis, default_icv_dir=""):
    path_dis = ph.get_nearest_neighbors_vec_path(default_icv_dir)
    with open(path_dis, "w") as nn_dis_file:
        for k, v in nn_dis.items():
            print(f"ltable id : {k}, nearest neighbors : {v}", file=nn_dis_file)


def filter_match_res_memory(match_tab, attr, K, threshold, search_strategy=Literal["exact", "approximate"], 
                            default_vmatcher_dir="", default_icv_dir=""):
    time_st1 = time.time()
    
    # get word embeddings
    buck, query_vec, rev_buck, rev_query_vec, dim, value_matcher = _get_word_embeddings(match_tab, attr, default_vmatcher_dir)
    
    # similarity search
    time_st2 = time.time()
    nn_res, nn_dis = _search_KNN(dim, K, buck, query_vec, search_strategy)
    rev_nn_res, rev_nn_dis = _search_KNN(dim, K, rev_buck, rev_query_vec, search_strategy)
    time_end2 = time.time()
    
    # filter
    match_tab = _slim_match_tab(match_tab, nn_res, value_matcher, attr, threshold)
    match_tab = _slim_match_tab(match_tab, rev_nn_res, value_matcher, attr, threshold, True)
    
    _check_similarity(nn_dis, default_icv_dir)
    
    time_end1 = time.time()
    print(f"total filtering time : {time_end1 - time_st1}")
    print(f"similarity search time : {time_end2 - time_st2}")
    
    return match_tab


def filter_match_res_disk(attr, K, threshold, search_strategy=Literal["exact", "approximate"], 
                          default_vmatcher_dir="", default_icv_dir="", 
                          default_match_res_dir=""):
    time_st1 = time.time()
    
    # get the matching result
    path_match_res = ph.get_match_res_path(default_match_res_dir)
    match_tab = pd.read_csv(path_match_res)
    
    # get word embeddings
    buck, query_vec, rev_buck, rev_query_vec, dim, value_matcher = _get_word_embeddings(match_tab, attr, default_vmatcher_dir)
    
    # similarity search
    time_st2 = time.time()
    nn_res, nn_dis = _search_KNN(dim, K, buck, query_vec, search_strategy)
    rev_nn_res, rev_nn_dis = _search_KNN(dim, K, rev_buck, rev_query_vec, search_strategy)
    time_end2 = time.time()
    
    # filter
    match_tab = _slim_match_tab(match_tab, nn_res, value_matcher, attr, threshold)
    match_tab = _slim_match_tab(match_tab, rev_nn_res, value_matcher, attr, threshold, True)
    
    _check_similarity(nn_dis, default_icv_dir)
    
    time_end1 = time.time()
    print(f"total filtering time : {time_end1 - time_st1}")
    print(f"similarity search time : {time_end2 - time_st2}")
    
    return match_tab