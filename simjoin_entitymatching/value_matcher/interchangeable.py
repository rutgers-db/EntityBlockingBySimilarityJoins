# author: Yunqi Li
# contact: liyunqixa@gmail.com
from simjoin_entitymatching.value_matcher.doc2vec import Doc2Vec
from typing import Literal
import pathlib


def group_interchangeable(tableA, tableB, group_tau, group_strategy=Literal["doc", "mix"], num_data=Literal[1, 2], 
						 default_match_res_dir="", default_vmatcher_dir="", default_icv_dir=""):
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