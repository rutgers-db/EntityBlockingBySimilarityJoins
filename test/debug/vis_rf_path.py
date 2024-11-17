import py_entitymatching as em
import py_entitymatching.feature.attributeutils as au
import sys
import pandas as pd
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds
import simjoin_entitymatching.matcher.random_forest as randf
import networkx as nx


dir_path = "../datasets/tables/megallen/amazon-google-structured"
path_tableA = "/".join([dir_path, "table_a.csv"])
path_tableB = "/".join([dir_path, "table_b.csv"])
path_gold = "/".join([dir_path, "gold.csv"])

path_rule = "simjoin_entitymatching/blocker/rules/rules_amazon_google_structured_1.txt"
path_range = "simjoin_entitymatching/matcher/model/ranges/ranges_amazon_google_structured_1.txt"
path_tree = "simjoin_entitymatching/matcher/model/trees/trees_amazon_google_structured_1.txt"
path_rf = "simjoin_entitymatching/matcher/model/rf_amazon_google_structured_1.joblib"

gold_graph = nx.Graph()
tableA = read_csv_table(path_tableA)
tableB = read_csv_table(path_tableB)
gold = read_csv_golds(path_gold, gold_graph)

map_A = {tableA.loc[ridx, "id"] : ridx for ridx in list(tableA.index)}
map_B = {tableB.loc[ridx, "id"] : ridx for ridx in list(tableB.index)}

attr_types_ltable = au.get_attr_types(tableA)
attr_types_rtable = au.get_attr_types(tableB)
attr_types_ltable['manufacturer'] = "str_eq_1w"
attr_types_rtable['manufacturer'] = "str_eq_1w"

rf = randf.RandomForest()
rf.graph = gold_graph
rf.load_model(path_rf)

# Features selection
rf.generate_features(tableA, tableB, at_ltable=attr_types_ltable, at_rtable=attr_types_rtable, wrtie_fea_names=True)

blk_res_cand = em.read_csv_metadata("test/debug/false_neg.csv", key="_id", 
                                    ltable=tableA, rtable=tableB, 
                                    fk_ltable="ltable_id", 
                                    fk_rtable="rtable_id")

H = em.extract_feature_vecs(blk_res_cand, 
                            feature_table=rf.features, 
                            show_progress=False)

rf.label_cand(H)

false_neg = pd.read_csv("test/debug/false_neg.csv")

for idx, row in false_neg.iterrows():
    lid = int(row["ltable_id"])
    rid = int(row["rtable_id"])
    
    print(f"left tuple: {tableA.loc[map_A[lid]]}")
    print(f"right tuple: {tableB.loc[map_B[rid]]}")
    
    em.debug_randomforest_matcher(rf.rf, tableA.loc[map_A[lid]], tableB.loc[map_B[rid]], rf.features, H.columns, 
                                  exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'])