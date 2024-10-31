# unit test
# feature
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table
from simjoin_entitymatching.feature.feature import run_feature_lib, get_features_megallen
import networkx as nx
import pandas as pd


dir_path = "../datasets/tables/megallen/amazon-google-structured"
path_tableA = "/".join([dir_path, "table_a.csv"])
path_tableB = "/".join([dir_path, "table_b.csv"])
path_gold = "/".join([dir_path, "gold.csv"])

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
# run_feature_lib(is_interchangeable=0, flag_consistent=0, total_table=389, total_attr=len(schemas), usage="match", attrs=schemas)