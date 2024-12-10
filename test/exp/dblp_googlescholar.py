"""
experiment script on dblp_googlescholar
"""

import sys
sys.path.append(".")
import pandas as pd
import py_entitymatching.feature.attributeutils as au
from argparse import ArgumentParser
import time

from simjoin_entitymatching.sampler.sample import run_sample_lib
from simjoin_entitymatching.blocker.block import run_simjoin_block_lib, extract_block_rules
from simjoin_entitymatching.matcher.match import train_model
from simjoin_entitymatching.matcher.match import match_via_cpp_features, match_via_megallen_features, match_on_neg_pres
from simjoin_entitymatching.matcher.match_exp import run_experiments
from simjoin_entitymatching.value_matcher.interchangeable import group_interchangeable

import exp_utils

argp = ArgumentParser()

argp.add_argument("--turn", type=int, required=True, help="the exp turn for each script, from 1 - 3")
argp.add_argument("--dtype", type=str, required=True, help="the data type, structured, dirty or textual")

args = argp.parse_args()
data_name = "dblp_googlescholar"


def main(turn, dtype, mode="match_exp"):
    # get all paths
    path_tableA, path_tableB, path_gold, \
    path_rule, path_range, path_tree, path_rf = exp_utils.get_paths(data_name=data_name, data_type=dtype, exp_turn=turn)
    
    # read tables from disk
    tableA, tableB, gold, gold_graph = exp_utils.read_tables(path_tableA, path_tableB, path_gold)

    # dump tables to specific paths
    exp_utils.dump_tables(tableA, tableB, gold)

    # fix attributes' type for tables
    attr_types_ltable = au.get_attr_types(tableA)
    attr_types_rtable = au.get_attr_types(tableB)
    attr_types_ltable['authors'] = "str_bt_1w_5w"
    attr_types_rtable['authors'] = "str_bt_1w_5w"
    
    # select representative / most informative attribute
    representativeA = exp_utils.get_representative_attr(tableA, tableB)

    # sample a subset for training
    run_sample_lib(sample_strategy="down", blocking_attr="title", cluster_tau=0.9, sample_tau=4.0, 
                   step2_tau=0.18, num_data=2)

    # train the model and build the graph
    _, trigraph = train_model(tableA, tableB, gold_graph, blocking_attr="title", model_path=path_rf, tree_path=path_tree, range_path=path_range,
                              num_tree=11, sample_size=-1, ground_truth_label=False, training_strategy="tuning", 
                              inmemory=1, num_data=2, at_ltable=attr_types_ltable, at_rtable=attr_types_rtable)

    # extract the rule-based blocker
    extract_block_rules(trigraph=trigraph, rule_path=path_rule, move_strategy="basic", 
                        additional_rule_path=None, optimal_rule_path=None)

    # block
    run_simjoin_block_lib(blocking_attr="title", blocking_attr_type="str_bt_5w_10w", blocking_top_k=3000000, 
                          path_tableA="", path_tableB="", path_gold="", path_rule=path_rule, 
                          table_size=100000, is_join_topk=0, is_idf_weighted=1, 
                          num_data=2)
    
    exp_utils.cat_blocking_topk_output(data_name, dtype, representativeA, turn)
    
    # if mode == "match_exp":
    #     file_name = '/'.join(["output/exp/match_stat", data_name + "_" + dtype + ".txt"])
    #     filep = open(file_name, "w")
    #     run_experiments(tableA, tableB, "title", attr_types_ltable, attr_types_rtable, gold_graph, filep, impute_strategy="mean")
    #     filep.flush()
    #     filep.close()
    # else:
    #     # first-round match
    #     match_via_megallen_features(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=0, flag_consistent=0, 
    #                                 at_ltable=attr_types_ltable, at_rtable=attr_types_rtable)
        
    #     match_res_0 = pd.read_csv("output/match_res/match_res.csv")
    #     match_res_0.to_csv("output/match_res/match_res_py.csv", index=False)
        
    #     # cat the output to stat directory
    #     exp_utils.cat_blocking_topk_output(data_name, dtype, representativeA, turn)
    #     exp_utils.cat_match_res_output_first(data_name, dtype, turn)
        
    #     # indentify interchangeable values in first-round matching results
    #     group, cluster = group_interchangeable(tableA, tableB, group_tau=0.95, group_strategy="doc", num_data=2)
        
    #     # second-round match: only on negative results of the first-round    
    #     match_on_neg_pres(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=1, flag_consistent=0, 
    #                       at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, numeric_attr=["price", "year"])
        
    #     exp_utils.cat_match_res_output_mid(data_name, dtype, turn)
        
    #     # second-round match: on the entire blocking results
    #     match_via_cpp_features(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=1, flag_consistent=0, 
    #                         at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, numeric_attr=["price", "year"])
        
    #     exp_utils.cat_match_res_output_second(data_name, dtype, turn)
        
    #     # compare the two rounds matching results
    #     match_res_1 = pd.read_csv("output/match_res/match_res.csv")
    #     exp_utils.compare_two_match_res(match_res_0, match_res_1)
    
    
if __name__ == '__main__':
    turn = args.turn
    dtype = args.dtype
    
    if turn == 0:
        for t in range(1, 4):
            start = time.time()
            print(f"--- this is turn {turn} exp on dblp google scholar {dtype} ---")
            main(t, dtype)
            print(f"--- exp end ---")
            print(f"elapsed time: {time.time() - start - 10}")
    else:
        start = time.time()
        print(f"--- this is turn {turn} exp on dblp google scholar {dtype} ---")
        main(turn, dtype)
        print(f"--- exp end ---")
        print(f"elapsed time: {time.time() - start - 10}")
        
        
