# unit test
# match
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table
from simjoin_entitymatching.feature.feature import run_feature_lib
from simjoin_entitymatching.matcher.match import match_via_megallen_feature, match_via_cpp_features, train_model
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

numeric_attr = ["price", "year"]
schemas = list(tableA)[1:]
schemas = [attr for attr in schemas if attr not in numeric_attr]

attr_types_ltable = au.get_attr_types(tableA)
attr_types_rtable = au.get_attr_types(tableB)
attr_types_ltable['manufacturer'] = "str_eq_1w"
attr_types_rtable['manufacturer'] = "str_eq_1w"

random_forest, trigraph = train_model(tableA, tableB, gold_graph, blocking_attr="title", model_path=path_rf, tree_path=path_tree, range_path=path_range,
                                      num_tree=11, sample_size=-1, ground_truth_label=True, training_strategy="tuning", 
                                      inmemory=1, num_data=2, at_ltable=attr_types_ltable, at_rtable=attr_types_rtable)

match_via_cpp_features(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=0, flag_consistent=0, 
                       at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, numeric_attr=numeric_attr)

# match_via_megallen_feature(tableA, tableB, gold_graph, len(gold), model_path=path_rf, is_interchangeable=0, flag_consistent=0, 
#                            at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, group=None, cluster=None)