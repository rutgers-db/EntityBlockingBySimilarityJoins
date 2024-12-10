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

from simjoin_entitymatching.value_matcher.doc2vec import Doc2Vec
import simjoin_entitymatching.utils.path_helper as ph


'''
the match table ("attr" is the representative):
_id    ltable_id   rtable_id   ltable_attr    rtable_attr   ......

Query set Q: ltable_id
Index set I: rtable_id
'''


def _get_word_embeddings(attr, group_tau, default_vmatcher_dir="", default_icv_dir="", default_match_res_dir=""):
    doc2vec = Doc2Vec(inmemory_=0)
    doc2vec.load_model(usage=1, attr=attr, default_model_dir=default_vmatcher_dir)
    vec_dict = doc2vec._group_interchangeable_parallel(attr, group_tau, 1, default_icv_dir, 
                                                       default_match_res_dir)
    
    path_match_res = ph.get_match_res_path(default_match_res_dir)
    match_tab = pd.read_csv(path_match_res)
    
    l_schema = "ltable_" + attr
    r_schema = "rtable_" + attr
    
    # bucket
    index_buck = set()
    query_buck = set()
    index_vecs = defaultdict()
    query_vecs = defaultdict()
    
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
        
        index_buck.add(rid)
        query_buck.add(lid)
        index_vecs[rid] = r_vec
        query_vecs[lid] = l_vec
        
    return index_buck, index_vecs, query_vecs, prev_dim


def _search_KNN(dimension, K, index_buck, index_vecs, query_vecs, search_strategy=Literal["exact", "approximate"]):
    # prepare the embeddings for faiss
    index_data = np.array(list(index_vecs.values())).astype('float-32')
    
    # normalize
    faiss.normalize_L2(index_data)
    
    # index
    if search_strategy == "exact":
        search_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
    else:
        search_index = faiss.index_factory(dimension, "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    search_index.add(index_data)
    
    # query
    nn_res = defaultdict(list)
    for k, v in query_vecs.items():
        faiss.normalize_L2(v)
        _, idx = search_index.search(v, K)
        nn_id = index_buck[idx]
        nn_res[k] = nn_id
        
    return nn_res


def filter_match_res(attr, group_tau, K, search_strategy=Literal["exact", "approximate"], 
                     default_vmatcher_dir="", default_icv_dir="", default_match_res_dir=""):
    # get word embeddings
    index_buck, index_vecs, query_vecs, dim = _get_word_embeddings(attr, group_tau, default_vmatcher_dir, 
                                                                   default_icv_dir, default_match_res_dir)
    
    # similarity search
    nn_res = _search_KNN(dim, K, index_buck, index_vecs, query_vecs, search_strategy)
    
    # filter