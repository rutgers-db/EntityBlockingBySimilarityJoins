# unit test
# down sample
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from simjoin_entitymatching.utils.utils import read_csv_table, read_csv_golds, dump_table
from simjoin_entitymatching.sampler.sample import run_sample_lib
import networkx as nx
from py_entitymatching import OverlapBlocker


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

run_sample_lib(sample_strategy="down", blocking_attr="title", cluster_tau=0.9, sample_tau=4.0, step2_tau=0.18, num_data=2)
run_sample_lib(sample_strategy="cluster", blocking_attr="title", cluster_tau=0.9, sample_tau=4.0, step2_tau=0.15, num_data=2)

ob = OverlapBlocker()
C = ob.block_tables(tableA, tableB, "title", "title", 
                    word_level=True, overlap_size=4, 
                    l_output_attrs=["id"], 
                    r_output_attrs=["id"], 
                    allow_missing=False,
                    show_progress=False)

cur_golds = 0
row_index = list(C.index)

for index in row_index:
    id1 = str(C.loc[index, 'ltable_id']) + 'A'
    id2 = str(C.loc[index, 'rtable_id']) + 'B'
    if gold_graph.has_edge(id1, id2) == True:
        cur_golds += 1

recall = cur_golds / len(gold) * 1.0
density = cur_golds / len(C) * 1.0
print(f"recall: {recall}")
print(f"density: {density}")