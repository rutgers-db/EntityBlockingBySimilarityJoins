import gensim.models
from gensim import utils
import pathlib
import joblib
import pandas as pd
from collections import defaultdict
import time
from typing import Literal

import simjoin_entitymatching.utils.path_helper as ph

""" 
word2vec model for word embedding;
the semantic similarity of a tuple pairs will be calculated by "coherent group";
"Seeping Semantics: Linking Datasets using Word Embeddings for Data Discovery", ICDE'23

WARNING: remain tested
"""

_cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())


def _train_on_raw_table(attrs, num_data=Literal[1, 2], default_buffer_dir="", 
                        default_output_dir=""):
    '''
    train the word2vec multiple attributes
    concat two raw tables to form training corpus
    '''
    
    # read
    path_tab_A, path_tab_B = ph.get_raw_tables_path(default_buffer_dir)
    tab_A = pd.read_csv(path_tab_A)
    tab_B = None if num_data == 1 else pd.read_csv(path_tab_B)
    
    for attr in attrs:
        print(f"training word2vec on : {attr} ...")
        
        # process
        raw_text_A = tab_A[attr].tolist()
        raw_text_B = [] if num_data == 1 else tab_B[attr].tolist()
        raw_text = raw_text_A + raw_text_B 
        raw_text = [doc for doc in raw_text if not pd.isnull(doc)]
        
        corpus = []
        for i, line in enumerate(raw_text):
            toks = utils.simple_preprocess(line)
            corpu = gensim.models.doc2vec.TaggedDocument(toks, [i])
            corpus.append(corpu)
            
        # train
        st = time.time()
        word2vec = gensim.models.word2vec.Word2Vec(vector_size=50, min_count=1, epochs=40)
        word2vec.build_vocab(corpus)
        word2vec.train(corpus, total_examples=word2vec.corpus_count, epochs=word2vec.epochs)
        print(f"corpus size : {len(corpus)}, training time : {time.time() - st}")
        
        # dump
        model_path = ph.get_value_matcher_path(_cur_parent_dir, attr, default_output_dir)    
        joblib.dump(word2vec, model_path)
        
        
def group_interchangeable_external(target_attr, word2vec, default_match_res_dir="", 
                                   default_icv_dir=""):
    # apply
    lattr = "ltable_" + target_attr
    rattr = "rtable_" + target_attr
    vec_dict = defaultdict()
    vec_pair = []

    # read
    partial_name, _ = ph.get_chunked_match_res_path(0, default_match_res_dir)   
    partial_match_res = pd.read_csv(partial_name)
    
    # infer
    for _, row in partial_match_res.iterrows():
        ori_lstr = row[lattr]
        ori_rstr = row[rattr]
        if pd.isnull(ori_lstr) == True or pd.isnull(ori_rstr) == True:
            continue
        # process to model
        lstr = utils.simple_preprocess(ori_lstr)
        rstr = utils.simple_preprocess(ori_rstr)
        # # infer vecs
        # lvec = word2vec.infer_vector(lstr)
        # rvec = word2vec.infer_vector(rstr)
        # # add
        # vec_dict[ori_lstr] = lvec
        # vec_dict[ori_rstr] = rvec
        # vec_pair.append((ori_lstr, ori_rstr))

    # report
    vec_path, pair_path = ph.get_icval_vec_input_path(_cur_parent_dir, default_icv_dir)
    
    # vec
    with open(vec_path, "w") as vecfile:
        stat = [str(len(vec_dict)), '\n']
        vecfile.writelines(stat)
        for k, v in vec_dict.items():
            # str
            vecfile.writelines([k + "\n"])
            # vectors
            v = [str(e) + ' ' for e in v]
            v.insert(0, str(len(v)) + ' ')
            v.append('\n')
            vecfile.writelines(v)
            
    # candidare pair
    with open(pair_path, "w") as pairfile:
        stat = [str(len(vec_pair)), "\n"]
        pairfile.writelines(stat)
        for pair in vec_pair:
            pairfile.writelines([pair[0] + "\n"])
            pairfile.writelines([pair[1] + "\n"])
            
    return vec_dict