# author: Yunqi Li
# contact: liyunqixa@gmail.com
import networkx as nx
from typing import Literal
import pathlib
import pandas as pd
from collections import defaultdict

from simjoin_entitymatching.value_matcher.group import run_group_lib
from simjoin_entitymatching.value_matcher.doc2vec import Doc2Vec
import simjoin_entitymatching.value_matcher.fasttext as ft
from simjoin_entitymatching.feature.feature import run_feature_lib
import simjoin_entitymatching.utils.path_helper as ph
import simjoin_entitymatching.matcher.random_forest as randf


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
 
 
def cluster_pairs(ori_clt, representative_attr, gold_graph, default_match_res_dir=""):
	cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
	if default_match_res_dir == "":
		match_res_dir = '/'.join([cur_parent_dir, "..", "..", "output", "match_res"])
		match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "match_res.csv"])
		neg_match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "neg_match_res.csv"])
	else:
		default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
															 else default_match_res_dir 
		match_res_dir = default_match_res_dir
		match_res_path = '/'.join([default_match_res_dir, "match_res.csv"])
		neg_match_res_path = '/'.join([default_match_res_dir, "neg_match_res.csv"])
		
	pos_res_df = pd.read_csv(match_res_path)
	neg_res_df = pd.read_csv(neg_match_res_path)
	
	if representative_attr not in ori_clt.keys():
		raise KeyError(f"no such attribute: {representative_attr}")
	cur_clt = ori_clt[representative_attr]
	
	lrep_key = "ltable_" + representative_attr
	rrep_key = "rtable_" + representative_attr
	
	# init dsu
	cluster_graph = nx.Graph()
	clt_bucket = defaultdict(set)
	
	for _, row in pos_res_df.iterrows():
		lid = row["ltable_id"]
		rid = row["rtable_id"]
		lrep_attr = row[lrep_key]
		rrep_attr = row[rrep_key]
		
		if lrep_attr in cur_clt.keys():
			lrep_clt_id = cur_clt[lrep_attr]
			clt_bucket[lrep_clt_id].add(lid) 
		if rrep_attr in cur_clt.keys():
			rrep_clt_id = cur_clt[rrep_attr]
			clt_bucket[rrep_clt_id].add(rid)
			
	for _, row in pos_res_df.iterrows():
		lid = row["ltable_id"]
		rid = row["rtable_id"]
		lrep_attr = row[lrep_key]
		rrep_attr = row[rrep_key]
		
		glid = str(lid) + "A"
		grid = str(rid) + "B"
		cluster_graph.add_edge(glid, grid)
		
		if lrep_attr not in cur_clt.keys() and rrep_attr not in cur_clt.keys():
			continue
		elif lrep_attr in cur_clt.keys() and rrep_attr not in cur_clt.keys():
			lrep_clt_id = cur_clt[lrep_attr]
			for _lid in clt_bucket[lrep_clt_id]:
				_glid = str(_lid) + "A"
				if cluster_graph.has_edge(_glid, grid) == False:
					cluster_graph.add_edge(_glid, grid)
		elif lrep_attr not in cur_clt.keys() and rrep_attr in cur_clt.keys():
			rrep_clt_id = cur_clt[rrep_attr]
			for _rid in clt_bucket[rrep_clt_id]:
				_grid = str(_rid) + "B"
				if cluster_graph.has_edge(glid, _grid) == False:
					cluster_graph.add_edge(glid, _grid)
		else:
			lrep_clt_id = cur_clt[lrep_attr]
			rrep_clt_id = cur_clt[rrep_attr]
			for _lid in clt_bucket[lrep_clt_id]:
				_glid = str(_lid) + "A"
				for _rid in clt_bucket[rrep_clt_id]:
					_grid = str(_rid) + "B"
					if cluster_graph.has_edge(_glid, _grid) == False:
						cluster_graph.add_edge(_glid, _grid)
	
	new_count = 0
	new_gold_count = 0
 
	for _, row in neg_res_df.iterrows():
		lid = row["ltable_id"]
		rid = row["rtable_id"]
		glid = str(lid) + "A"
		grid = str(rid) + "B"
  
		if cluster_graph.has_edge(glid, grid) == True:
			new_count += 1
			if gold_graph.has_edge(glid, grid) == True:
				new_gold_count += 1
   
	print(f"new count on neg result: {new_count}, gold count: {new_gold_count}, out of {len(neg_res_df)}")


def group_interchangeable(tableA, tableB, group_tau, group_strategy=Literal["doc", "mix"], num_data=Literal[1, 2], external_group=False,
						  external_group_strategy=Literal["graph", "cluster"], is_transitive_closure=False, 
        				  default_match_res_dir="", default_vmatcher_dir="", default_icv_dir="", 
              			  default_buffer_dir=""):
	'''
	apply value matcher, group interchangeable values on matching result
		1. use doc2vec for all attrs, since for str_eq_1w there may exist values that are longer than 1 word in raw data
		2. use doc2vec & word2vec(for str_eq_1w), we omit the impact of such abnormal(longer) words in 1
	'''
	
	path_match_stat = ph.get_match_res_stat_path(default_match_res_dir)
	with open(path_match_stat, "r") as stat_file:
		stat_line = stat_file.readlines()
		total_table, _ = (int(val) for val in stat_line[0].split())

	train_tableB = None if num_data == 1 else tableB

	# drop numeric
	numeric_attr = ["price", "year", ""]
	attrs = list(tableA)[1:]
	attrs = [attr for attr in attrs if attr not in numeric_attr]

	group, cluster = {}, {}

	if group_strategy == 'doc':
		doc2vec = Doc2Vec(inmemory_=0)
		doc2vec.load_match_res(tableA=tableA, tableB=tableB, default_match_res_dir=default_match_res_dir)
		# doc2vec.train_all_and_save(attrs, tableA, train_tableB, default_vmatcher_dir)
		doc2vec.train_on_raw_table(attrs, num_data, default_buffer_dir, default_vmatcher_dir)
		print('training done', flush=True)
		for attr_ in attrs:
			doc2vec.load_model(usage=1, attr=attr_, default_model_dir=default_vmatcher_dir)
			if external_group == True:
				vec_dict = doc2vec._group_interchangeable_parallel(attr_, group_tau, total_table, default_icv_dir, 
																   default_match_res_dir)
				run_group_lib(attr_, external_group_strategy, group_tau, is_transitive_closure, default_icv_dir)
			else:
				grp, clt = doc2vec.group_interchangeable_parallel(attr_, group_tau, total_table, default_icv_dir, 
																default_match_res_dir)
				group[attr_], cluster[attr_] = grp, clt

	elif group_strategy == 'mix':
		raise NotImplementedError("mix group not established")

	return group, cluster


def group_interchangeable_fasttext(target_attr, group_tau, external_group_strategy=Literal["graph", "cluster"], 
                                   is_transitive_closure=False, default_match_res_dir="", default_vmatcher_dir="", 
                                   default_icv_dir=""):
	'''
	we currently only apply fasttext on representative attribute and use external group strategy "coherent group"
	please only use it in experiments at this stage
	'''

	path_match_stat = ph.get_match_res_stat_path(default_match_res_dir)
	with open(path_match_stat, "r") as stat_file:
		stat_line = stat_file.readlines()
		total_table, _ = (int(val) for val in stat_line[0].split())
	
	# load pre-trained model
	model = ft._load_wiki_pre_trained_model(default_vmatcher_dir)
	ft._dump_model(model, default_vmatcher_dir)
	print(f"done loading pre-trained fasttext model")
 
	# group in experiments mode
	vec_dict = ft.group_interchangeable_external_exp(target_attr, model, default_match_res_dir, default_icv_dir)
	
	# coherent group
	run_group_lib(target_attr, external_group_strategy, group_tau, is_transitive_closure, default_icv_dir)


def _sample_neg_match_res(sample_size, default_match_res_dir=""):
	'''
	sample the negative match results to estimate the best group threshold
	'''
	path_match_stat = ph.get_match_res_stat_path(default_match_res_dir)
	with open(path_match_stat, "r") as stat_file:
		stat_line = stat_file.readlines()
		total_table, _ = (int(val) for val in stat_line[0].split())
		
	for i in range(total_table):
		_, path_neg_match_res = ph.get_chunked_match_res_path(i, default_match_res_dir)
		neg_match_res = pd.read_csv(path_neg_match_res)
		
		sample_neg_match_res = neg_match_res.sample(sample_size)
		path_sample_neg_match_res = path_neg_match_res[ : -4] + "_sample.csv"
		sample_neg_match_res.to_csv(path_sample_neg_match_res, index=False)
		tot_sample_neg_match_res = sample_neg_match_res if i == 0 \
			else pd.concat([tot_sample_neg_match_res, sample_neg_match_res], ignore_index=False)
		 
	path_tot = path_match_stat.split('/')[ : -1] + "neg_match_res_sample.csv"
	tot_sample_neg_match_res.to_csv(path_tot, index=False)