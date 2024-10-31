# unit test
# sim join blocker
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table
from simjoin_entitymatching.sampler.sample import run_sample_lib
from simjoin_entitymatching.blocker.block import run_simjoin_block_lib
import networkx as nx
import pandas as pd


dir_path = "../datasets/tables/megallen/amazon-google-structured"
path_tableA = "/".join([dir_path, "table_a.csv"])
path_tableB = "/".join([dir_path, "table_b.csv"])
path_gold = "/".join([dir_path, "gold.csv"])
path_rule = "simjoin_entitymatching/blocker/rules/rules_amazon_google_structured1.txt"

gold_graph = nx.Graph()
tableA = read_csv_table(path_tableA)
tableB = read_csv_table(path_tableB)
gold = read_csv_golds(path_gold, gold_graph)

dump_table(tableA, "output/buffer/clean_A.csv")
dump_table(tableB, "output/buffer/clean_B.csv")
dump_table(gold, "output/buffer/gold.csv")

# check sample
sample_res = Path("output/buffer/sample_res.csv")
try:
    sample_abs_path = sample_res.resolve(strict=True)
except FileNotFoundError:
    run_sample_lib(sample_strategy="down", blocking_attr="title", cluster_tau=0.9, 
                   sample_tau=4.0, step2_tau=0.18, num_data=2)
    
sample_tab = pd.read_csv(sample_abs_path)
for index in list(sample_tab.index):
    id1 = str(sample_tab.loc[index, 'ltable_id']) + 'A'
    id2 = str(sample_tab.loc[index, 'rtable_id']) + 'B'
    if gold_graph.has_edge(id1, id2) == True:
        sample_tab.loc[index, 'label'] = 1
sample_tab.to_csv(sample_abs_path, index=False)

run_simjoin_block_lib(blocking_attr="title", blocking_attr_type="str_bt_5w_10w", blocking_top_k=150000, 
                      path_tableA="", path_tableB="", path_gold="", path_rule=path_rule, 
                      table_size=10000, is_join_topk=0, is_idf_weighted=1, 
                      num_data=2)