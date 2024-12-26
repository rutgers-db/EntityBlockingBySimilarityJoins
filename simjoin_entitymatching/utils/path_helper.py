# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pathlib


def get_raw_tables_path(default_buffer_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_buffer_dir == "":
        buffer_dir = '/'.join([cur_parent_dir, "..", "..", "output", "buffer"])
    else:
        buffer_dir = default_buffer_dir[ : -1] if default_buffer_dir[-1] == '/' \
                                               else default_buffer_dir
    path_tab_A = '/'.join([buffer_dir, "clean_A.csv"])
    path_tab_B = '/'.join([buffer_dir, "clean_B.csv"])
    return path_tab_A, path_tab_B


def get_chunked_blk_res_path(table_id, default_blk_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        blk_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "blk_res", "blk_res" + str(table_id) + ".csv"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        blk_res_path = '/'.join([default_blk_res_dir, "blk_res" + str(table_id) + ".csv"])
    return blk_res_path


def get_chunked_fea_vec_path(table_id, default_blk_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        fea_vec_path = '/'.join([cur_parent_dir, "..", "..", "output", "blk_res", "feature_vec" + str(table_id) + ".csv"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        fea_vec_path = '/'.join([default_blk_res_dir, "feature_vec" + str(table_id) + ".csv"])
    return fea_vec_path


def get_deep_matcher_input_path(default_blk_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        blk_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "blk_res", "blk_res_dm.csv"])
        root_dir = '/'.join([cur_parent_dir, "..", "..", "output", "blk_res"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        blk_res_path = '/'.join([default_blk_res_dir, "blk_res_dm.csv"])
        root_dir = default_blk_res_dir
        
    # subtables
    path_dm_train = '/'.join([root_dir, "train.csv"])
    path_dm_validation = '/'.join([root_dir, "validation.csv"])
    path_dm_test = '/'.join([root_dir, "test.csv"])
        
    return blk_res_path, root_dir, path_dm_train, path_dm_validation, path_dm_test


def get_chunked_match_res_path(table_id, default_match_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_match_res_dir == "":
        match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "match_res" + str(table_id) + ".csv"])
        neg_match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "neg_match_res" + str(table_id) + ".csv"])
    else:
        default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
                                                             else default_match_res_dir 
        match_res_path = '/'.join([default_match_res_dir, "match_res" + str(table_id) + ".csv"])
        neg_match_res_path = '/'.join([default_match_res_dir, "neg_match_res" + str(table_id) + ".csv"])
    return match_res_path, neg_match_res_path


def get_match_res_path(default_match_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_match_res_dir == "":
        match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "match_res.csv"])
    else:
        default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
                                                             else default_match_res_dir 
        match_res_path = '/'.join([default_match_res_dir, "match_res.csv"])
    return match_res_path


def get_blk_res_stat_path(default_blk_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        path_block_stat = "/".join([cur_parent_dir, "..", "..", "output", "blk_res", "stat.txt"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        path_block_stat = "/".join([default_blk_res_dir, "stat.txt"])
    return path_block_stat


def get_match_res_stat_path(default_match_res_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_match_res_dir == "":
        path_match_stat = "/".join([cur_parent_dir, "..", "..", "output", "match_res", "stat.txt"])
    else:
        default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
                                                             else default_match_res_dir
        path_match_stat = "/".join([default_match_res_dir, "stat.txt"])
    return path_match_stat


def get_icval_vec_input_path(cur_parent_dir, default_icv_dir):
    if default_icv_dir == "":
        vec_path = "/".join([cur_parent_dir, "ic_values", "vec_interchangeable.txt"])
        pair_path = "/".join([cur_parent_dir, "ic_values", "pair_interchangeable.txt"])
    else:
        default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                    else default_icv_dir
        vec_path = "/".join([default_icv_dir, "vec_interchangeable.txt"])
        pair_path = "/".join([default_icv_dir, "pair_interchangeable.txt"])
    return vec_path, pair_path


def get_neg_icval_vec_input_path(cur_parent_dir, default_icv_dir):
    if default_icv_dir == "":
        vec_path = "/".join([cur_parent_dir, "ic_values", "vec_interchangeable_neg.txt"])
        pair_path = "/".join([cur_parent_dir, "ic_values", "pair_interchangeable_neg.txt"])
    else:
        default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                    else default_icv_dir
        vec_path = "/".join([default_icv_dir, "vec_interchangeable_neg.txt"])
        pair_path = "/".join([default_icv_dir, "pair_interchangeable_neg.txt"])
    return vec_path, pair_path


def get_icval_vec_path(cur_parent_dir, default_icv_dir):
    if default_icv_dir == "":
        vec_path = "/".join([cur_parent_dir, "ic_values", "vec_interchangeable.txt"])
        vec_label_path = "/".join([cur_parent_dir, "ic_values", "vec_interchangeable_label.txt"])
    else:
        default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                    else default_icv_dir
        vec_path = "/".join([default_icv_dir, "vec_interchangeable.txt"])
        vec_label_path = "/".join([default_icv_dir, "vec_interchangeable_label.txt"])
    return vec_path, vec_label_path


def get_nearest_neighbors_vec_path(default_icv_dir):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_icv_dir == "":
        vec_path = "/".join([cur_parent_dir, "..", "value_matcher", "ic_values", "nn_dis.txt"])
    else:
        default_icv_dir = default_icv_dir[ : -1] if default_icv_dir[-1] == '/' \
                                                    else default_icv_dir
        vec_path = "/".join([default_icv_dir, "nn_dis.txt"])
    return vec_path


def get_value_matcher_path(cur_parent_dir, attr, default_output_dir):
    if default_output_dir == "":
        path_model = "/".join([cur_parent_dir, "model", "doc2vec_" + attr + ".joblib"])
    else:
        default_output_dir = default_output_dir[ : -1] if default_output_dir[-1] == '/' \
                                                        else default_output_dir
        path_model = "/".join([default_output_dir, "doc2vec_" + attr + ".joblib"])
    return path_model


def get_fasttext_pre_trained_dir(cur_parent_dir, default_model_dir):
    if default_model_dir == "":
        path_dir = "/".join([cur_parent_dir, "model"])
    else:
        default_model_dir = default_model_dir[ : -1] if default_model_dir[-1] == '/' \
                                                     else default_model_dir
        path_dir = "/".join([default_model_dir])
    return path_dir