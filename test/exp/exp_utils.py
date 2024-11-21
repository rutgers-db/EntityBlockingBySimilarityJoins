# experiments utils
import pandas as pd
from os import system
from typing import Literal
import networkx as nx
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table, select_representative_attr


def get_paths(data_name, data_type, exp_turn):
    dir_path = "../datasets/tables/megallen/amazon-google-structured"
    path_tableA = "/".join([dir_path, "table_a.csv"])
    path_tableB = "/".join([dir_path, "table_b.csv"])
    path_gold = "/".join([dir_path, "gold.csv"])
    
    rule_name = "rules_" + data_name + "_" + data_type + str(exp_turn) + ".txt"
    range_name = "ranges_" + data_name + "_" + data_type + str(exp_turn) + ".txt"
    tree_name = "trees_" + data_name + "_" + data_type + str(exp_turn) + ".txt"
    model_name = "rf_" + data_name + "_" + data_type + str(exp_turn) + ".joblib"
    
    path_rule = "/".join(["simjoin_entitymatching/blocker/rules", rule_name])
    path_range = "/".join(["simjoin_entitymatching/matcher/model/ranges", range_name])
    path_tree = "/".join(["simjoin_entitymatching/matcher/model/trees", tree_name])
    path_rf = "/".join(["simjoin_entitymatching/matcher/model", model_name])

    return path_tableA, path_tableB, path_gold, path_rule, path_range, path_tree, path_rf


def read_tables(path_tableA, path_tableB, path_gold):
    gold_graph = nx.Graph()
    tableA = read_csv_table(path_tableA)
    tableB = read_csv_table(path_tableB)
    gold = read_csv_golds(path_gold, gold_graph)
    return tableA, tableB, gold, gold_graph


def dump_tables(tableA, tableB, gold):
    dump_table(tableA, "output/buffer/clean_A.csv")
    dump_table(tableB, "output/buffer/clean_B.csv")
    dump_table(gold, "output/buffer/gold.csv")
    
    
def get_representative_attr(tableA, tableB):
    representativeA = select_representative_attr(tableA)
    representativeB = select_representative_attr(tableB)
    if representativeA != representativeB:
        raise ValueError(f"different representative attrs: {representativeA}, {representativeB}")
    return representativeA


def cat_blocking_topk_output(dataname, dtype, rep_attr, turn=Literal[1, 2, 3]):
    # redriect to the corresponding dataset's stat file
    topk_intermedia = "output/topk_stat/intermedia.txt"
    topk_exp_log = "output/topk_stat/" + dataname + "_" + dtype + ".txt"
    # header
    print(f"--- report top k blocking result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + topk_exp_log
    system(echo_command)
    # blocking results
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)
    # tailer and rep attr
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + topk_exp_log
    system(echo_command)
    echo_command = "echo " + rep_attr + " >> " + topk_exp_log
    system(echo_command)


def cat_blocking_topk_output_first(dataname, dtype, turn=Literal[1, 2, 3]):
    topk_intermedia = "output/topk_stat/intermedia.txt"
    topk_exp_log = "output/topk_stat/" + dataname + "_" + dtype + ".txt"
    print(f"--- report top k blocking result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + topk_exp_log
    system(echo_command)
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)


def cat_blocking_topk_output_second(dataname, dtype, rep_attr, turn=Literal[1, 2, 3]):
    topk_intermedia = "output/topk_stat/intermedia.txt"
    topk_exp_log = "output/topk_stat/" + dataname + "_" + dtype + ".txt"
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + topk_exp_log
    system(echo_command)
    echo_command = "echo " + rep_attr + " >> " + topk_exp_log
    system(echo_command)


def cat_match_res_output_first(dataname, dtype, turn=Literal[1, 2, 3]):
    match_intermedia = "output/match_stat/intermedia.txt"
    match_exp_log = "output/match_stat/" + dataname + "_" + dtype + ".txt"
    print(f"--- report matching result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + match_exp_log
    system(echo_command)
    # the first-round matching results
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)
    
    
def cat_match_res_output_mid(dataname, dtype, turn=Literal[1, 2, 3]):
    match_intermedia = "output/match_stat/intermedia.txt"
    match_exp_log = "output/match_stat/" + dataname + "_" + dtype + ".txt"
    # the second-round matching results
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)


def cat_match_res_output_second(dataname, dtype, turn=Literal[1, 2, 3]):
    match_intermedia = "output/match_stat/intermedia.txt"
    match_exp_log = "output/match_stat/" + dataname + "_" + dtype + ".txt"
    # the second-round matching results
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)
    # tailer
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + match_exp_log
    system(echo_command)
    
    
def compare_two_match_res(match_res_0, match_res_1):
    '''
    match_res_0: first-round matching
    match_res_1: sceond-round matching with interchangeable values
    
    bugs: 
        * sometimes the pairs in "match_diff_01.csv" will be predicted as non-match
        * sometimes the pairs with interchangeable values in "match_diff_01.csv" will be predicted as match
    '''
    bucket_0 = set(list(zip(match_res_0["ltable_id"], match_res_0["rtable_id"])))
    bucket_1 = set(list(zip(match_res_1["ltable_id"], match_res_1["rtable_id"])))
        
    bucket_common = bucket_0.intersection(bucket_1)
    # tuple pairs in 0 but not in 1
    bucket_diff_01 = bucket_0.difference(bucket_1)
    # tuple pairs in 1 but not in 0
    bucket_diff_10 = bucket_1.difference(bucket_0)
    
    print(f"common pairs : {len(bucket_common)}, first_match / second_match : {len(bucket_diff_01)}, \
            second_match / first_match : {len(bucket_diff_10)}")
    
    ground_truth = pd.read_csv("output/buffer/gold.csv")
    golds = set()
    for _, row in ground_truth.iterrows():
        lid = int(row["id1"])
        rid = int(row["id2"])
        golds.add((lid, rid))
        
    common_hit = bucket_common.intersection(golds)
    diff_01_hit = bucket_diff_01.intersection(golds)
    diff_10_hit = bucket_diff_10.intersection(golds)
    print(f"common golds : {len(common_hit)}, first_match / second_match : {len(diff_01_hit)}, second_match / first_match : {len(diff_10_hit)}")
    
    # read the feature vectors
    with open("output/blk_res/stat.txt", "r") as stat_file:
        stat_line = stat_file.readlines()
        total_table, _ = (int(val) for val in stat_line[0].split())
    
    path_prefix = "output/blk_res/feature_vec"
    for i in range(total_table):
        path_fea_vec_0 = path_prefix + str(i) + "_py.csv"
        path_fea_vec_1 = path_prefix + str(i) + ".csv"
        fea_vec_0 = pd.read_csv(path_fea_vec_0)
        fea_vec_1 = pd.read_csv(path_fea_vec_1)
        
        if i == 0:
            concat_fea_vec_0 = fea_vec_0
            concat_fea_vec_1 = fea_vec_1
        else:
            concat_fea_vec_0 = pd.concat([concat_fea_vec_0, fea_vec_0], ignore_index=True)
            concat_fea_vec_1 = pd.concat([concat_fea_vec_1, fea_vec_1], ignore_index=True)
            
    concat_fea_vec_0["combine_id"] = list(zip(concat_fea_vec_0["ltable_id"], concat_fea_vec_0["rtable_id"]))
    concat_fea_vec_1["combine_id"] = list(zip(concat_fea_vec_1["ltable_id"], concat_fea_vec_1["rtable_id"]))
    
    buck_01 = list(bucket_diff_01)
    buck_10 = list(bucket_diff_10)
    df_diff_01 = concat_fea_vec_0[concat_fea_vec_0["combine_id"].isin(buck_01)]
    df_diff_10 = concat_fea_vec_1[concat_fea_vec_1["combine_id"].isin(buck_10)]
    
    # dump
    df_diff_01.to_csv("test/debug/match_diff_01.csv", index=False)
    df_diff_10.to_csv("test/debug/match_diff_10.csv", index=False)
    
    
def rerun_em_on_normalized_tables():
    """
    this plan may be depracted in the future
    """
    # path_normalized_A = "output/buffer/normalized_A.csv"
    # path_normalized_B = "output/buffer/normalized_B.csv"
    # normalized_A = read_csv_table(path_normalized_A)
    # normalized_B = read_csv_table(path_normalized_B)

    # attr_types_ltable_2 = au.get_attr_types(normalized_A)
    # attr_types_rtable_2 = au.get_attr_types(normalized_B)
    # attr_types_ltable_2['manufacturer'] = "str_eq_1w"
    # attr_types_rtable_2['manufacturer'] = "str_eq_1w"

    # dump_table(normalized_A, "output/buffer/clean_A.csv")
    # dump_table(normalized_B, "output/buffer/clean_B.csv")

    # # sample a subset
    # run_sample_lib(sample_strategy="down", blocking_attr="title", cluster_tau=0.9, sample_tau=4.0, 
    #                step2_tau=0.18, num_data=2)

    # # train the model and build the graph
    # _, trigraph = train_model(normalized_A, normalized_B, gold_graph, blocking_attr="title", model_path=path_rf, tree_path=path_tree, range_path=path_range,
    #                           num_tree=11, sample_size=-1, ground_truth_label=True, training_strategy="tuning", 
    #                           inmemory=1, num_data=2, at_ltable=attr_types_ltable_2, at_rtable=attr_types_rtable_2)

    # # extract the rule-based blocker
    # extract_block_rules(trigraph=trigraph, rule_path=path_rule, move_strategy="basic", 
    #                     additional_rule_path=None, optimal_rule_path=None)

    # # block
    # run_simjoin_block_lib(blocking_attr="title", blocking_attr_type="str_bt_5w_10w", blocking_top_k=150000, 
    #                       path_tableA="", path_tableB="", path_gold="", path_rule=path_rule, 
    #                       table_size=100000, is_join_topk=0, is_idf_weighted=1, 
    #                       num_data=2)

    # match_via_cpp_features(normalized_A, normalized_B, gold_graph, len(gold), model_path=path_rf, is_interchangeable=0, flag_consistent=0, 
    #                        at_ltable=attr_types_ltable_2, at_rtable=attr_types_rtable_2, numeric_attr=["price", "year"])

    # cat output
    # exp_utils.cat_blocking_topk_output_second("amazon_google", "structured", representativeA, turn)
    # exp_utils.cat_match_res_output_second("amazon_google", "structured", turn)
    pass