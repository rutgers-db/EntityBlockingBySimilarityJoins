# author: Yunqi Li
# contact: liyunqixa@gmail.com
from simjoin_entitymatching.value_matcher.doc2vec import Doc2Vec
from typing import Literal
import pathlib
import pandas as pd


def normalize_values(ori_group, ori_clt, normalized_attrs, default_buffer_dir=""):
	cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
	if default_buffer_dir == "":
		table_dir = "/".join([cur_parent_dir, "..", "..", "output", "buffer"])
		path_clean_A = "/".join([table_dir, "clean_A.csv"])
		path_clean_B = "/".join([table_dir, "clean_B.csv"])
		path_normalized_A = "/".join([table_dir, "normalized_A.csv"])
		path_normalized_B = "/".join([table_dir, "normalized_B.csv"])
	else:
		default_buffer_dir = default_buffer_dir[ : -1] if default_buffer_dir[-1] == '/' \
													   else default_buffer_dir
		path_clean_A = "/".join([default_buffer_dir, "clean_A.csv"])
		path_clean_B = "/".join([default_buffer_dir, "clean_B.csv"])
		path_normalized_A = "/".join([default_buffer_dir, "normalized_A.csv"])
		path_normalized_B = "/".join([default_buffer_dir, "normalized_B.csv"])
  
	clean_A = pd.read_csv(path_clean_A)
	clean_B = pd.read_csv(path_clean_B)
	schema = list(clean_A.columns)
	row_index_A = list(clean_A.index)
	row_index_B = list(clean_B.index)
 
	for attr in schema:
		if attr not in normalized_attrs:
			continue
	
		cur_grp = ori_group[attr]
		cur_clt = ori_clt[attr]
		
		for ridx in row_index_A:
			val = clean_A.loc[ridx, attr]
			if val not in cur_clt.keys():
				continue
			clt_id = cur_clt[val]
			cur_docs = list(cur_grp[clt_id])
			if len(cur_docs) > 1:
				clean_A.loc[ridx, attr] = cur_docs[0]
   
		for ridx in row_index_B:
			val = clean_B.loc[ridx, attr]
			if val not in cur_clt.keys():
				continue
			clt_id = cur_clt[val]
			cur_docs = list(cur_grp[clt_id])
			if len(cur_docs) > 1:
				clean_B.loc[ridx, attr] = cur_docs[0]
	
	clean_A.rename(columns={"id": "_id"}, inplace=True)
	clean_B.rename(columns={"id": "_id"}, inplace=True)
	clean_A.to_csv(path_normalized_A, index=False)
	clean_B.to_csv(path_normalized_B, index=False)


def group_interchangeable(tableA, tableB, group_tau, group_strategy=Literal["doc", "mix"], num_data=Literal[1, 2], 
						  default_match_res_dir="", default_vmatcher_dir="", default_icv_dir="", default_buffer_dir=""):
	'''
	apply value matcher, group interchangeable values on matching result
		1. use doc2vec for all attrs, since for str_eq_1w there may exist values that are longer than 1 word in raw data
		2. use doc2vec & word2vec(for str_eq_1w), we omit the impact of such abnormal(longer) words in 1
	'''
	
	cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
	if default_match_res_dir == "":
		path_match_stat = "/".join([cur_parent_dir, "..", "..", "output", "match_res", "stat.txt"])
	else:
		default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
															 else default_match_res_dir
		path_match_stat = "/".join([default_match_res_dir, "stat.txt"])

	with open(path_match_stat, "r") as stat_file:
		stat_line = stat_file.readlines()
		total_table, _ = (int(val) for val in stat_line[0].split())

	train_tableB = None if num_data == 1 else tableB

	# drop numeric
	numeric_attr = ["price", "year"]
	attrs = list(tableA)[1:]
	attrs = [attr for attr in attrs if attr not in numeric_attr]

	group, cluster = {}, {}

	if group_strategy == 'doc':
		doc2vec = Doc2Vec(inmemory_=0)
		doc2vec.load_match_res(tableA=tableA, tableB=tableB, default_match_res_dir=default_match_res_dir)
		doc2vec.train_all_and_save(attrs, tableA, train_tableB, default_vmatcher_dir)
		print('training done', flush=True)
		for attr_ in attrs:
			doc2vec.load_model(usage=1, attr=attr_, default_model_dir=default_vmatcher_dir)
			grp, clt = doc2vec.group_interchangeable_parallel(attr_, group_tau, total_table, default_icv_dir)
			group[attr_], cluster[attr_] = grp, clt

	elif group_strategy == 'mix':
		raise NotImplementedError("mix group not established")

	normalize_values(group, cluster, normalized_attrs=attrs, default_buffer_dir=default_buffer_dir)