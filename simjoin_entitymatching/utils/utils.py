# author: Yunqi Li
# contact: liyunqixa@gmail.com

import py_entitymatching as em
import pandas as pd
from collections import defaultdict


def read_csv_table(path):
	table = em.read_csv_metadata(path, key="_id")
	if "categb ry" in list(table):
		table.rename(columns={"categb ry": "category"}, inplace=True)
	table.rename(columns={'_id':'id'}, inplace=True)
	em.set_key(table, "id")
	return table


def read_parquet_table(path):
	table = pd.read_parquet(path, engine='fastparquet')
	if "categb ry" in list(table):
		table.rename(columns={"categb ry": "category"}, inplace=True)
	table.rename(columns={'_id':'id'}, inplace=True)
	em.set_key(table, "id")
	return table


def dump_table(table, path):
	table.to_csv(path, index=False)

 
def read_csv_golds(path, graph):
	gold = pd.read_csv(path)
	row_indexs = list(gold.index)
	for index in row_indexs:
		idA = str(gold.loc[index, 'id1']) + 'A'
		idB = str(gold.loc[index, 'id2']) + 'B'
		graph.add_edge(idA, idB)
	return gold
		

def read_parquet_golds(path, graph):
	gold = pd.read_parquet(path, engine='fastparquet')
	row_indexs = list(gold.index)
	for index in row_indexs:
		idA = str(gold.loc[index, 'id1']) + 'A'
		idB = str(gold.loc[index, 'id2']) + 'B'
		graph.add_edge(idA, idB)
	return gold


def get_attr_types(table):
	atypes = em.get_attr_types(table)
	for attr in list(table):
		print(f"attribute: {attr}, type: {atypes[attr]}")
		
		
def select_representative_attr(table):
	# drop id
	schemas = list(table)[1:]
	atypes = em.get_attr_types(table)
	candidates = [attr for attr in schemas if atypes[attr] == "str_gt_10w" or atypes[attr] == "str_bt_5w_10w"]
	
	# select candidates based on tokens num
	delims = " \"\',\\\t\r\n"
	collect = defaultdict(set)
	
	for attr in candidates:
		for tup in table[attr]:
			if pd.isnull(tup):
				continue
			tokens = em.tok_delim(tup, delims)
			collect[attr].update(tokens)

	max_len = 0
	representative = ""
	for k, v in collect.items():
		print(f"candidate: {k}, {len(v)}")
		if len(v) > max_len:
			max_len = len(v)
			representative = k
	
	return representative