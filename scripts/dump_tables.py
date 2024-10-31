import pandas as pd
import sys

path_table_A = sys.argv[1]
path_table_B = sys.argv[2]
path_gold = sys.argv[3]

A = pd.read_csv(path_table_A)
B = pd.read_csv(path_table_B)
G = pd.read_csv(path_gold)

A.to_csv("output/buffer/clean_A.csv", index=False)
B.to_csv("output/buffer/clean_B.csv", index=False)
G.to_csv("output/buffer/gold.csv", index=False)