# unit test
# blocking all
import sys
from os import system
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table, select_representative_attr
from simjoin_entitymatching.sampler.sample import run_sample_lib
from simjoin_entitymatching.blocker.block import run_simjoin_block_lib, extract_block_rules
from simjoin_entitymatching.matcher.match import train_model
from simjoin_entitymatching.matcher.match import match_via_cpp_features
from simjoin_entitymatching.value_matcher.interchangeable import group_interchangeable
import networkx as nx
import pandas as pd
import py_entitymatching.feature.attributeutils as au
from argparse import ArgumentParser
import time

argp = ArgumentParser()

argp.add_argument("--turn", type=int, required=True, help="the exp turn for each script, from 1 - 3")
argp.add_argument("--dtype", type=str, required=True, help="the data type, structured, dirty or textual")

args = argp.parse_args()


def main(turn, dtype):
    dir_path = "../datasets/tables/megallen/amazon-google-structured"
    path_tableA = "/".join([dir_path, "table_a.csv"])
    path_tableB = "/".join([dir_path, "table_b.csv"])
    path_gold = "/".join([dir_path, "gold.csv"])

    path_rule = "simjoin_entitymatching/blocker/rules/rules_amazon_google_" + dtype + str(turn) + ".txt"
    path_range = "simjoin_entitymatching/matcher/model/ranges/ranges_amazon_google_" + dtype + str(turn) + ".txt"
    path_tree = "simjoin_entitymatching/matcher/model/trees/trees_amazon_google_" + dtype + str(turn) + ".txt"
    path_rf = "simjoin_entitymatching/matcher/model/rf_amazon_google_" + dtype + str(turn) + ".joblib"

    gold_graph = nx.Graph()
    tableA = read_csv_table(path_tableA)
    tableB = read_csv_table(path_tableB)
    gold = read_csv_golds(path_gold, gold_graph)

    dump_table(tableA, "output/buffer/clean_A.csv")
    dump_table(tableB, "output/buffer/clean_B.csv")
    dump_table(gold, "output/buffer/gold.csv")

    attr_types_ltable = au.get_attr_types(tableA)
    attr_types_rtable = au.get_attr_types(tableB)
    attr_types_ltable['manufacturer'] = "str_eq_1w"
    attr_types_rtable['manufacturer'] = "str_eq_1w"
    
    representativeA = select_representative_attr(tableA)
    representativeB = select_representative_attr(tableB)
    if representativeA != representativeB:
        raise ValueError(f"different representative attrs: {representativeA}, {representativeB}")

    # sample a subset
    run_sample_lib(sample_strategy="down", blocking_attr="title", cluster_tau=0.9, sample_tau=4.0, 
                   step2_tau=0.18, num_data=2)

    # train the model and build the graph
    _, trigraph = train_model(tableA, tableB, gold_graph, blocking_attr="title", model_path=path_rf, tree_path=path_tree, range_path=path_range,
                              num_tree=11, sample_size=-1, ground_truth_label=True, training_strategy="tuning", 
                              inmemory=1, num_data=2, at_ltable=attr_types_ltable, at_rtable=attr_types_rtable)

    # extract the rule-based blocker
    extract_block_rules(trigraph=trigraph, rule_path=path_rule, move_strategy="basic", 
                        additional_rule_path=None, optimal_rule_path=None)

    # block
    run_simjoin_block_lib(blocking_attr="title", blocking_attr_type="str_bt_5w_10w", blocking_top_k=150000, 
                          path_tableA="", path_tableB="", path_gold="", path_rule=path_rule, 
                          table_size=100000, is_join_topk=0, is_idf_weighted=1, 
                          num_data=2)
    
    match_via_cpp_features(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=0, flag_consistent=0, 
                           at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, numeric_attr=["price", "year"])
    
    topk_intermedia = "output/topk_stat/intermedia.txt"
    topk_exp_log = "output/topk_stat/amazon_google_" + dtype + ".txt"
    print(f"--- report top k blocking result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + topk_exp_log
    system(echo_command)
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)
    
    match_intermedia = "output/match_stat/intermedia.txt"
    match_exp_log = "output/match_stat/amazon_google_" + dtype + ".txt"
    print(f"--- report matching result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + match_exp_log
    system(echo_command)
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)
    
    # group_interchangeable(tableA, tableB, group_tau=0.85, group_strategy="doc", num_data=2)
    
    # match_via_cpp_features(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=1, flag_consistent=0, 
    #                        at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, numeric_attr=["price", "year"])
    
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
    
    # '''
    # when doing experiments on matcher, change the output log in bash scripts.
    # '''

    # # cat output
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + topk_exp_log
    system(echo_command)
    echo_command = "echo " + representativeA + " >> " + topk_exp_log
    system(echo_command)
    
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + match_exp_log
    system(echo_command)
    
    
if __name__ == '__main__':
    turn = args.turn
    dtype = args.dtype
    
    if turn == 0:
        for t in range(1, 4):
            start = time.time()
            print(f"--- this is turn {turn} exp on amazon google {dtype} ---")
            main(t, dtype)
            print(f"--- exp end ---")
            print(f"elapsed time: {time.time() - start - 10}")
    else:
        start = time.time()
        print(f"--- this is turn {turn} exp on amazon google {dtype} ---")
        main(turn, dtype)
        print(f"--- exp end ---")
        print(f"elapsed time: {time.time() - start - 10}")