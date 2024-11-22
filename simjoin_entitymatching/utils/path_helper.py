# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pathlib


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