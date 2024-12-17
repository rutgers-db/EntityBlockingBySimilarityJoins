# author: Yunqi Li
# contact: liyunqixa@gmail.com
import gensim.models
import joblib
import pandas as pd
import pathlib
from collections import defaultdict
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer

import simjoin_entitymatching.utils.path_helper as ph

""" 
fasttexts model for word embedding;
the semantic similarity of a tuple pairs will be calculated by "coherent group";
"Seeping Semantics: Linking Datasets using Word Embeddings for Data Discovery", ICDE'23
"""

_cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())


def _load_wiki_pre_trained_model(default_model_dir=""):
    model_dir = ph.get_fasttext_pre_trained_dir(_cur_parent_dir, default_model_dir)
    path_vec_bin = "/".join([model_dir, "wiki.en.bin"])
    model = gensim.models.fasttext.load_facebook_vectors(path_vec_bin)
    return model


def _dump_model(model, default_model_dir=""):
    model_dir = ph.get_fasttext_pre_trained_dir(_cur_parent_dir, default_model_dir)
    path_model = "/".join([model_dir, "fasttexts_wiki.joblib"])
    joblib.dump(model)
    
    
def _if_some_alphanumeric(s):
    for i in s:
        if i.isalnum():
            return True
    return False
    
    
def group_interchangeable_external_exp(target_attr, fasttext, default_match_res_dir="", 
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
    tokenizer = DelimiterTokenizer([' ', '\'', '\"', ',', '\\', '\t', '\r', '\n'])
    
    for _, row in partial_match_res.iterrows():
        ori_lstr = row[lattr]
        ori_rstr = row[rattr]
        
        # check missing
        if pd.isnull(ori_lstr) or pd.isnull(ori_rstr):
            raise ValueError(f"error in target attribute : {target_attr} with : {ori_lstr}, {ori_rstr}")
        
        # tokenize
        l_toks = tokenizer.tokenize(ori_lstr)
        r_toks = tokenizer.tokenize(ori_rstr)
        l_toks = [tok for tok in l_toks if _if_some_alphanumeric(tok)]
        r_toks = [tok for tok in r_toks if _if_some_alphanumeric(tok)]
        
        # vectors
        l_vecs = [fasttext.mv[tok] for tok in l_toks]
        r_vecs = [fasttext.mv[tok] for tok in r_toks]
        
        vec_dict[ori_lstr] = l_vecs
        vec_dict[ori_rstr] = r_vecs
        vec_pair.append((ori_lstr, ori_rstr))

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
            vecfile.writelines([str(len(v)) + "\n"])
            for sub_v in v:
                sub_v2 = [str(e) + ' ' for e in sub_v]
                sub_v2.insert(0, str(len(v)) + ' ')
                sub_v2.append('\n')
                vecfile.writelines(sub_v2)
            
    # candidare pair
    with open(pair_path, "w") as pairfile:
        stat = [str(len(vec_pair)), "\n"]
        pairfile.writelines(stat)
        for pair in vec_pair:
            pairfile.writelines([pair[0] + "\n"])
            pairfile.writelines([pair[1] + "\n"])
            
    return vec_dict
    