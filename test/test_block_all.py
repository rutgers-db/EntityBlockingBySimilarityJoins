# unit test
# blocking all
import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table
from simjoin_entitymatching.sampler.sample import run_sample_lib
from simjoin_entitymatching.blocker.block import run_simjoin_block_lib, extract_block_rules
from simjoin_entitymatching.matcher.match import train_model
import networkx as nx
import pandas as pd
import py_entitymatching.feature.attributeutils as au


dir_path = "../datasets/tables/megallen/amazon-google-structured"
path_tableA = "/".join([dir_path, "table_a.csv"])
path_tableB = "/".join([dir_path, "table_b.csv"])
path_gold = "/".join([dir_path, "gold.csv"])
path_rule = "test/tmp/rules_amazon_google_structured1.txt"
path_range = "test/tmp/ranges_amazon_google_structured.txt"
path_tree = "test/tmp/trees_amazon_google_structured.txt"
path_rf = "test/tmp/rf_amazon_google_structured.joblib"

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

run_sample_lib(sample_strategy="down", blocking_attr="title", cluster_tau=0.9, sample_tau=4.0, 
               step2_tau=0.18, num_data=2)

random_forest, trigraph = train_model(tableA, tableB, gold_graph, blocking_attr="title", model_path=path_rf, tree_path=path_tree, range_path=path_range,
                                      num_tree=11, sample_size=-1, ground_truth_label=True, training_strategy="tuning", 
                                      inmemory=1, num_data=2, at_ltable=attr_types_ltable, at_rtable=attr_types_rtable)

extract_block_rules(trigraph=trigraph, rule_path=path_rule, move_strategy="basic", 
                    additional_rule_path=None, optimal_rule_path=None)

run_simjoin_block_lib(blocking_attr="title", blocking_attr_type="str_bt_5w_10w", blocking_top_k=150000, 
                      path_tableA="", path_tableB="", path_gold="", path_rule=path_rule, 
                      table_size=100000, is_join_topk=0, is_idf_weighted=1, 
                      num_data=2)