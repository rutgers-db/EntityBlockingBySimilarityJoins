# author: Yunqi Li
# contact: liyunqixa@gmail.com
import py_entitymatching as em
import networkx as nx
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import multiprocessing
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
import joblib
import copy
import pathlib
import subprocess
import cloudpickle
from typing import Literal
import py_entitymatching.utils.generic_helper as gh
# debug
from py_entitymatching.catalog.catalog import Catalog
import py_entitymatching.catalog.catalog_manager as cm
import time
# self-defined
from simjoin_entitymatching.feature.feature_base import NewFeatures
import simjoin_entitymatching.utils.path_helper as ph


class RandomForest:
	'''
	Random forest matcher
	'''


	def __init__(self):
		self.graph = nx.Graph()           # gold graph
		self.cand = ''                    # candidates for training
		self.cand_backup = ' '
		self.test_table = ' '
		self.features = []                # features for training
		self.rf = ''                      # random forest
		self.sparkrf = ''                 # spark random forest
		self.num_total = 0
		self.num_training = 0
  

	def _entropy(self, p1, p2):
		log_p1 = 0 if p1 == 0 else np.log2(p1)
		log_p2 = 0 if p2 == 0 else np.log2(p2)
		return -1 * (p1 * log_p1 + p2 * log_p2)


	def _set_metadata(self,dataframe, key, fk_ltable, fk_rtable, ltable, rtable):
		'''
		py_entitymatching maintain a catalog as a dict with the id of dataframe as key
		on each operation like extract feature vector requires metadata
		but if your dataframe is not read using its api read_csv_metadata
		then you need to set it by yourself
		'''
		em.set_key(dataframe, key)
		em.set_fk_ltable(dataframe, fk_ltable)
		em.set_fk_rtable(dataframe, fk_rtable)
		em.set_ltable(dataframe, ltable)
		em.set_rtable(dataframe, rtable)


	# TODO: Check the difference without 'A' & 'B' in id1 & id2
	def get_recall(self, candidates, num_golds, external_report=False, 
                   default_res_dir=""):
		cur_golds = 0
		row_index = list(candidates.index)

		for index in row_index:
			id1 = str(candidates.loc[index, 'ltable_id']) + 'A'
			id2 = str(candidates.loc[index, 'rtable_id']) + 'B'
			if self.graph.has_edge(id1, id2) == True:
				cur_golds += 1

		recall = cur_golds / num_golds * 1.0
		density = cur_golds / len(candidates) * 1.0 if len(candidates) > 0 else 0.0
		f1 = 2 * ((recall * density) / (recall + density)) if recall + density > 0.0 else 0.0
  
		if external_report == True:
			if default_res_dir == "":
				cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
				output_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_stat", "intermedia.txt"])
			else:
				default_res_dir = default_res_dir[ : -1] if default_res_dir[-1] == '/' \
														 else default_res_dir
				output_path = '/'.join([default_res_dir, "intermedia.txt"])
			with open(output_path, "w") as stat_file:
				print("recall     : %.4f" % recall, file=stat_file)
				print("precision  : %.4f" % density, file=stat_file)
				print("F1 Score   : %.4f" % f1, file=stat_file)
				print(cur_golds, num_golds, len(candidates), file=stat_file)
		else:
			print("recall     : %.4f" % recall)
			print("precision  : %.4f" % density)
			print("F1 Score   : %.4f" % f1)
			print(cur_golds, num_golds, len(candidates))

		return recall, density, f1
 
 
	def get_recall_check(self, candidates, num_golds, check_file_path, check=0):
		cur_golds = 0
		row_index = list(candidates.index)
		matched = []

		for index in row_index:
			id1 = str(candidates.loc[index, 'ltable_id']) + 'A'
			id2 = str(candidates.loc[index, 'rtable_id']) + 'B'
			if self.graph.has_edge(id1, id2) == True:
				cur_golds += 1
				matched.append((candidates.loc[index, 'ltable_id'], candidates.loc[index, 'rtable_id']))

		# check_name = "buffer/check_res" + str(check) + ".txt"
		with open(check_file_path, "w") as checkfile:
			for pair in matched:
				print(pair[0], pair[1], file=checkfile)

		recall = cur_golds / num_golds * 1.0
		density = cur_golds / len(candidates) * 1.0 if len(candidates) > 0 else 0.0
		f1 = 2 * ((recall * density) / (recall + density)) if recall + density > 0.0 else 0.0
		print("recall     : %.4f" % recall)
		print("density    : %.4f" % density)
		print("F1 Score   : %.4f" % f1)
		print(cur_golds, num_golds, len(candidates))

		return recall, density


	def label_cand(self, C):
		C.insert(C.shape[1], 'label', 0)
		row_index = list(C.index)

		for index in row_index:
			id1 = str(C.loc[index, 'ltable_id']) + 'A'
			id2 = str(C.loc[index, 'rtable_id']) + 'B'
			if self.graph.has_edge(id1, id2) == True:
				C.loc[index, 'label'] = 1


	def cand_stat(self, C):
		positive_count = (C['label'] == 1).sum()
		print(f"cand|     : {len(C)}")
		print(f"positive  : {positive_count}")
		print(f"negative  : {len(C) - positive_count}")


	def over_sample(self, C):
		Y = np.zeros((len(C),), dtype=int)
		row_index = list(C.index)

		for yindex, index in enumerate(row_index):
			label = C.loc[index, 'label']
			Y[yindex] = (int(label) == 1)

		ros = RandomOverSampler(random_state=0)
		nC, Y = ros.fit_resample(C, Y)

		return nC


	def under_sample(self, C):
		Y = np.zeros((len(C),), dtype=int)
		row_index = list(C.index)

		for yindex, index in enumerate(row_index):
			label = C.loc[index, 'label']
			Y[yindex] = (int(label) == 1)

		ros = RandomUnderSampler(random_state=0)
		nC, Y = ros.fit_resample(C, Y)

		return nC
		

	def fix_null(self, tableA, tableB):
		outattrA = list(tableA)[1:]

		for idx in outattrA:
			if tableA[idx].dtypes == object:
				tableA[idx] = tableA[idx].fillna('none')
				tableB[idx] = tableB[idx].fillna('none')
				# tableA[idx] = tableA[idx].astype('|S')
				# tableB[idx] = tableB[idx].astype('|S')
			elif tableA[idx].dtypes == float:
				tableA[idx] = tableA[idx].fillna(0.0)
				tableB[idx] = tableB[idx].fillna(0.0)
			else:
				tableA[idx] = tableA[idx].fillna(0)
				tableB[idx] = tableB[idx].fillna(0)
				
	
	def sample_data(self, tableA, tableB, default_sample_res_dir=""):
		'''
		To get the training set for random forest,
		we should firstly block the tables and label raw results.
		'''
		
		# Blocking
		em.set_key(tableA, 'id')
		em.set_key(tableB, 'id')

		if default_sample_res_dir == "":
			cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
			sample_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "buffer", "sample_res.csv"])
		else:
			default_sample_res_dir = default_sample_res_dir[ : -1] if default_sample_res_dir[-1] == '/' \
																   else default_sample_res_dir
			sample_res_path = '/'.join([default_sample_res_dir, "sample_res.csv"])
   
		# if the format is parquet:
		test_sample = pathlib.Path(sample_res_path)
		if not test_sample.exists():
			sample_format = sample_res_path.split('.')[-1]
			if sample_format != "parquet":
				raise FileNotFoundError(f"no such file : {sample_res_path}")
			sample_parquet = pd.read_parquet(sample_res_path, engine='fastparquet')
			sample_csv_path = sample_res_path[ : -7] + "csv"
			sample_parquet.to_csv(sample_csv_path, index=False)
   
		C = em.read_csv_metadata(sample_res_path, 
								 key='_id',
								 ltable=tableA, rtable=tableB, 
								 fk_ltable='ltable_id', fk_rtable='rtable_id')
		self.cand, self.cand_backup = C, C

		# over sample
		# self.cand = self.over_sample(self.cand)
		self.cand_stat(self.cand)

		# Format 
		self.cand.rename(columns={'_id':'id'}, inplace=True)
		self.cand.reset_index(drop=True, inplace=True)
		self.cand['id'] = range(len(self.cand))
		em.set_key(self.cand, 'id')
  
		# set 
		self._set_metadata(self.cand, 
                       	   key="id", 
                           fk_ltable="ltable_id", fk_rtable="rtable_id", 
                           ltable=tableA, rtable=tableB)
		

	def generate_features(self, tableA, tableB, at_ltable=None, at_rtable=None, dataname=None, 
                       	  default_output_dir="", wrtie_fea_names=True):
		# check types of attrs, the last entry is the df itself
		aindex = list(tableA)
		bindex = list(tableB)
		typesA, typesB = [], []
		aindex = list(tableA)
		for idx in aindex:
			typesA.append(tableA[idx].dtypes)
			typesB.append(tableB[idx].dtypes)
		for i in range(len(aindex)):
			idx = aindex[i]
			tableA[idx] = tableA[idx].astype(typesA[i])
			tableB[idx] = tableB[idx].astype(typesB[i])
		
		atypes = em.get_attr_types(tableA)
		btypes = em.get_attr_types(tableB)

		# Skip id, check types
		atype_length = len(atypes) - 1
		for i in range(1, atype_length):
			if atypes[aindex[i]] < btypes[bindex[i]]:
				print(f"\033[31mFix: {btypes[bindex[i]]} to {atypes[aindex[i]]} for {aindex[i]}\033[0m")
				btypes[bindex[i]] = atypes[aindex[i]]
			elif atypes[aindex[i]] > btypes[bindex[i]]:
				print(f"\033[31mFix: {atypes[aindex[i]]} to {btypes[bindex[i]]} for {aindex[i]}\033[0m")
				atypes[aindex[i]] = btypes[bindex[i]]

		# Fix features
		self.features = NewFeatures.get_supported_features_for_matching(tableA, tableB, 
													 	 		   		validate_inferred_attr_types=False, 
                       													at_ltable=at_ltable, at_rtable=at_rtable, 
                                    									dataname=dataname)
		
		# Remove id
		self.features = self.features[self.features.left_attribute!='id']
		self.features = self.features[self.features.right_attribute!='id']
  
		if default_output_dir == "":
			cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
			feature_name_path = '/'.join([cur_parent_dir, "..", "..", "output", "buffer", "feature_names.txt"])
		else:
			default_output_dir = default_output_dir[ : -1] if default_output_dir[-1] == '/' \
														   else default_output_dir
			feature_name_path = '/'.join([default_output_dir, "feature_names.txt"])
		
		# Reformat
		self.features = self.features.reset_index(drop=True)

		if wrtie_fea_names == True:
			with open(feature_name_path, "w") as fname_file:
				print(len(self.features.feature_name), file=fname_file)
				for fname in self.features.feature_name:
					print(fname, file=fname_file)
  
  
	def re_sample_data(self, tableA, tableB, sample_size=-1, default_blk_res_dir=""):
		path_blk_res_stat = ph.get_blk_res_stat_path(default_blk_res_dir)
		with open(path_blk_res_stat, "r") as stat_file:
			stat_line = stat_file.readlines()
			total_table, _ = (int(val) for val in stat_line[0].split())
   
		for tab_id in range(total_table):
			path_blk_res = ph.get_chunked_blk_res_path(tab_id, default_blk_res_dir)
			blk_res = pd.read_csv(path_blk_res)
			tot_blk_res = blk_res if tab_id == 0 else pd.concat([tot_blk_res, blk_res], ignore_index=True)
   
		self._set_metadata(tot_blk_res, key="_id", fk_ltable="ltable_id", fk_rtable="rtable_id", ltable=tableA, rtable=tableB)
		self.cand = tot_blk_res if sample_size == -1 else em.sample_table(tot_blk_res, sample_size=sample_size)
  
	
	def train_model_normal(self, tableA, tableB, num_tree, sample_size, if_balanced=True):
		if len(self.cand) < sample_size and sample_size != -1:
			raise ValueError(f"too small cand: {len(self.cand)}, {sample_size}")
		
		sample_cand = em.sample_table(self.cand, sample_size=sample_size) if sample_size != -1 else self.cand
     
		IJ = em.split_train_test(sample_cand, train_proportion=0.67, random_state=0)
		I = IJ['train']
		J = IJ['test']
		
		self.num_training = len(I)
		self.num_total = len(sample_cand)

		self._set_metadata(I, key='id', 
                     	   fk_ltable='ltable_id', fk_rtable='rtable_id',
                           ltable=tableA, rtable=tableB)
		self._set_metadata(J, key='id', 
                     	   fk_ltable='ltable_id', fk_rtable='rtable_id',
                           ltable=tableA, rtable=tableB)
  
		H = em.extract_feature_vecs(I, 
									feature_table=self.features, 
									attrs_after='label',
									show_progress=False)
		L = em.extract_feature_vecs(J, 
							  		feature_table=self.features,
                            		attrs_after='label', 
									show_progress=False)
  
		# init random forest
		if if_balanced == True:
			self.rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree, class_weight='balanced')
		else:
			self.rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree)
   
		# train
		self.rf.fit(table=H, 
			   		exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
			   		target_attr='label')
		
		# Convert J into a set of feature vectors using F
		L = em.extract_feature_vecs(J, feature_table=self.features,
                            		attrs_after='label', show_progress=False)
		
		# Predict on L 
		predictions = self.rf.predict(table=L, exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
									  append=True, target_attr='predicted', inplace=False, 
								 	  return_probs=True, probs_attr='proba')
		
		# Evaluate the predictions
		eval_result = em.eval_matches(predictions, 'label', 'predicted')
		em.print_eval_summary(eval_result)
  
  
	def train_model_tuning(self, tableA, tableB, num_tree, sample_size, if_balanced=True):
		'''
		tune the hyper patameters to avoid overfitting
		5-fold cross-validation
  		'''
     
		if len(self.cand) < sample_size and sample_size != -1:
			raise ValueError(f"too small cand: {len(self.cand)}, {sample_size}")
		
		# sample_cand = em.sample_table(self.cand, sample_size=sample_size) if sample_size != -1 else self.cand
		if sample_size == -1:
			sample_cand = self.cand 
		else:
			pos_cand = self.cand.loc[self.cand['label'] == 1].copy()
			neg_cand = self.cand.loc[self.cand['label'] == 0].copy()
			half_size = int(sample_size / 2.0)
			posI = pos_cand.sample(half_size)
			negI = neg_cand.sample(half_size)
			sample_cand = pd.concat([posI, negI], ignore_index=True)
			self.cand.drop(sample_cand.index.values, inplace=True)
			self.cand.reset_index(drop=True, inplace=True)
			self._set_metadata(sample_cand, key='id', 
                     	   	   fk_ltable='ltable_id', fk_rtable='rtable_id',
                           	   ltable=tableA, rtable=tableB)

		# train : validation : test as 3 : 1 : 1
		tmp = em.split_train_test(sample_cand, train_proportion=0.8, random_state=0)
		train = tmp['train']
		test = tmp['test']
  
		train = self.over_sample(train)
		# train = self.under_sample(train)
		train.reset_index(drop=True, inplace=True)
		train['id'] = range(len(train))
		em.set_key(train, 'id')
  
		self.num_training = len(train)
		self.num_total = len(sample_cand)

		# test.drop("label", axis=1, inplace=True)
		# self.label_cand(test)

		self._set_metadata(train, key='id', 
                     	   fk_ltable='ltable_id', fk_rtable='rtable_id',
                           ltable=tableA, rtable=tableB)
		self._set_metadata(test, key='id', 
                     	   fk_ltable='ltable_id', fk_rtable='rtable_id',
                           ltable=tableA, rtable=tableB)
  
		train_H = em.extract_feature_vecs(train, 
										  feature_table=self.features, 
										  attrs_after='label',
										  show_progress=False)
		test_H = em.extract_feature_vecs(test, 
							  			 feature_table=self.features,
                            			 attrs_after='label', 
										 show_progress=False)

		# train_H = em.impute_table(train_H, exclude_attrs=["id", "ltable_id", "rtable_id", "label"], strategy="mean")
		train_H = em.impute_table(train_H, exclude_attrs=["id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)
  
		print(train_H.head())
  
		# init random forest
		if if_balanced == True:
			self.rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree, class_weight='balanced')
		else:
			self.rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree)

		# corss validation to avoid overfitting
		param_grid = {
			"max_depth": [None, 10, 20, 30],  # Maximum depth of the tree
			"min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
			"min_samples_leaf": [1, 2, 4]  # Minimum number of samples required to be at a leaf node
		}
  
		grid_search = GridSearchCV(estimator=self.rf.clf, param_grid=param_grid,
								   cv=5, n_jobs=1, verbose=2)
		
		# process the feature tables to numpy ndarray
		exclude_attrs = ["id", "ltable_id", "rtable_id", "label"]
		target_attr = "label"
		attributes_to_project = gh.list_diff(list(train_H.columns), exclude_attrs)
  
		X_train = train_H[attributes_to_project]
		y_train = train_H[target_attr]
		X_train, y_train = self.rf._get_data_for_sklearn(X_train, y_train)

		grid_search.fit(X_train, y_train)
   
		# train
		self.rf.clf = grid_search.best_estimator_
  
		# self.rf.fit(table=train_H, 
		# 	   		exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
		# 	   		target_attr='label')
		
		# Predict on L 
		predictions = self.rf.predict(table=test_H, exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
									  append=True, target_attr='predicted', inplace=False, 
								 	  return_probs=True, probs_attr='proba')
		
		# Evaluate the predictions
		eval_result = em.eval_matches(predictions, 'label', 'predicted')
		em.print_eval_summary(eval_result)
	

	def train_model_active(self, tableA, tableB, num_tree, sample_size, if_balanced=True):		
		# monitor set
		monitor_set = self.cand.sample(frac=.03)
		self._set_metadata(monitor_set, key='id', 
						   fk_ltable='ltable_id', fk_rtable='rtable_id', 
						   ltable=tableA, rtable=tableB)

		monitor_set = em.extract_feature_vecs(monitor_set, 
											  feature_table=self.features, 
											  attrs_after='label', 
											  show_progress=False)
		print(f"Monitor set size: {len(monitor_set)}")
		self.cand.drop(monitor_set.index.values, inplace=True)
		self.cand.reset_index(drop=True, inplace=True)
		
		# split data
		pos_cand = self.cand.loc[self.cand['label'] == 1].copy()
		neg_cand = self.cand.loc[self.cand['label'] == 0].copy()
		half_size = int(sample_size / 2.0)
		posI = pos_cand.sample(half_size)
		negI = neg_cand.sample(half_size)
		I = pd.concat([posI, negI], ignore_index=True)
		self.cand.drop(I.index.values, inplace=True)
		self.cand.reset_index(drop=True, inplace=True)
		em.set_key(I, 'id')
		J = self.cand

		self.num_training = len(I)
		self.num_total = len(self.cand)

		# Convert I & J into feature vectors using updated F
		self._set_metadata(I, key='id', 
						   fk_ltable='ltable_id', fk_rtable='rtable_id', 
						   ltable=tableA, rtable=tableB)
		H = em.extract_feature_vecs(I, 
									feature_table=self.features, 
									attrs_after='label',
									show_progress=False)
		L = em.extract_feature_vecs(J, 
							  		feature_table=self.features,
                            		attrs_after='label', 
									show_progress=False)

		# init random forest
		if if_balanced == True:
			self.rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree, class_weight='balanced')
		else:
			self.rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree)
		
		# active learning
		max_iteration = 30
		window_size = 5
		epsilon = 0.01
		n_converged = 20
		n_high = 3
		n_degrade = 15
		confidence_set = np.zeros((max_iteration,), dtype=float)
		left_window_len = int((window_size - 1) / 2)
		right_window_len = int(window_size - 1 - left_window_len)

		for turn in range(max_iteration):
			print(f"~~~training on epoch: {turn}, training set size: {len(H)}, verifying set size: {len(L)}~~~")

			self.rf.fit(table=H, 
						exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
						target_attr='label')
		
			# Predict on L 
			predictions = self.rf.predict(table=L, 
								 		  exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
										  append=True, target_attr='predicted', inplace=False, 
										  return_probs=True, probs_attr='proba')
			# Evaluate the predictions
			eval_result = em.eval_matches(predictions, 'label', 'predicted')
			em.print_eval_summary(eval_result)

			# using weighted sampling
			predictions = pd.DataFrame(predictions)
			predictions['entropy'] = predictions['proba'].map(lambda row: self._entropy(1-row, row))
			entropies = predictions[['entropy']].copy()
			entropies.sort_values(by="entropy", ascending=False, inplace=True)

			if len(entropies.index) < 100:
				break
			top_entropies = entropies.head(100)

			try:
				weighted_sample_indexes = top_entropies.sample(20, weights="entropy").index.values
			except ValueError:
				weighted_sample_indexes = top_entropies.head(20).index.values
			H = pd.concat([H, L.loc[weighted_sample_indexes]], ignore_index=True)
			L.drop(weighted_sample_indexes, inplace=True)

			# determine if stop
			# calculate current condifence score
			mpredict = self.rf.predict(table=monitor_set, 
								 	   exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
									   append=True, target_attr='predicted', inplace=False, 
									   return_probs=True, probs_attr='proba')
			mpredict = pd.DataFrame(mpredict)
			mpredict['confidence'] = mpredict['proba'].map(lambda row: 1 - self._entropy(1-row, row))
			confidence = mpredict['confidence'].sum()
			confidence_set[turn] = confidence / (len(mpredict) * 1.0)

			# maintain a window
			if turn < window_size - 1:
				continue
			# smooth 
			for smooth_idx in range(left_window_len, turn - right_window_len + 1):
				average = np.cumsum(confidence_set[smooth_idx - left_window_len : smooth_idx + right_window_len + 1], dtype=float)[-1]
				average = average / (window_size * 1.0)
				confidence_set[turn] = average

			# converged confidence
			if turn >= n_converged - 1:
				is_success = True
				# only enumerate the inner n_converged - 2 items
				for idx in range(turn + 2 - n_converged, turn):
					is_success = is_success & ((abs(confidence_set[idx] - confidence_set[idx - 1]) <= epsilon) \
											 or (abs(confidence_set[idx] - confidence_set[idx + 1]) <= epsilon))
				if is_success:
					print(f"Exit training: converged confidence: {confidence_set}")
					break
			# near-absolute confidence
			if turn >= n_high - 1:
				is_success = (confidence_set[turn-2] >= 1 - epsilon) \
							& (confidence_set[turn-1] >= 1 - epsilon) \
							& (confidence_set[turn] >= 1 - epsilon)
				if is_success:
					print(f"Exit training, near-absolute confidence: {confidence_set}")
					break
			# degrading confidence
			if turn >= n_degrade * 2 - 1:
				window1 = confidence_set[turn + 1 - n_degrade:turn + 1]
				window2 = confidence_set[turn + 1 - 2 * n_degrade : turn + 1 - n_degrade]
				if window2.max() - window1.max() >= epsilon:
					print(f"Exit training, degrading confidence: {confidence_set}")
					break


	def _apply_model_worker(self, tableid, tableA, tableB, external_fea_extract=False, if_report_pre=False, 
                         	default_blk_res_dir="", default_match_res_dir=""):
		'''
		external_extract: indicates whether using cpp to extract features, 
						  if not, the em package only supports non-interchangeable
  		'''
		cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
  
		if default_blk_res_dir == "":
			fea_vec_path = '/'.join([cur_parent_dir, "..", "..", "output", "blk_res", "feature_vec" + str(tableid) + ".csv"])
		else:
			default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
															 else default_blk_res_dir
			fea_vec_path = '/'.join([default_blk_res_dir, "feature_vec" + str(tableid) + ".csv"])

		if default_match_res_dir == "":
			match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "match_res" + str(tableid) + ".csv"])
			neg_match_res_path = '/'.join([cur_parent_dir, "..", "..", "output", "match_res", "neg_match_res" + str(tableid) + ".csv"])
		else:
			default_match_res_dir = default_match_res_dir[ : -1] if default_match_res_dir[-1] == '/' \
																 else default_match_res_dir 
			match_res_path = '/'.join([default_match_res_dir, "match_res" + str(tableid) + ".csv"])
			neg_match_res_path = '/'.join([default_match_res_dir, "neg_match_res" + str(tableid) + ".csv"])
		
		# Convert I into feature vectors using updated F
		if external_fea_extract == True:
			# print("~~~using external (cpp) feature extraction ~~~")
			H = em.read_csv_metadata(fea_vec_path, 
                            		 key="id", 
									 ltable=tableA, rtable=tableB,
									 fk_ltable="ltable_id", fk_rtable="rtable_id")
			H.rename(columns={"id": "_id"}, inplace=True)
			em.set_key(H, "_id")
		else:
			fea_vec_path = fea_vec_path[ : -4] + "_py.csv"
			H = em.read_csv_metadata(fea_vec_path, 
                            		 key="_id", 
									 ltable=tableA, rtable=tableB,
									 fk_ltable="ltable_id", fk_rtable="rtable_id")
		self.label_cand(H)
  
		# H = em.impute_table(H, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="mean")
		H = em.impute_table(H, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)

		predictions = self.rf.predict(table=H, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
								 	  append=True, target_attr='predicted', inplace=True, 
								 	  return_probs=True, probs_attr='proba')
  
		# need to rename since "itertuples()" can not parse the attrs with space or underline at the front
		predictions.rename(columns={'_id':'id'}, inplace=True)
		
		# Save predictions
		columns_ = ["_id", "ltable_id", "rtable_id"]
		schemas = list(tableA)[1:]
		lsch = ["ltable_" + sch for sch in schemas]
		rsch = ["rtable_" + sch for sch in schemas]
		columns_.extend(lsch)
		columns_.extend(rsch)
		rowsA = list(tableA.index)
		rowsB = list(tableB.index)
		mapA = {tableA.loc[rowidx, 'id'] : rowidx for rowidx in rowsA}
		mapB = {tableB.loc[rowidx, 'id'] : rowidx for rowidx in rowsB}
		
		pres_df = pd.DataFrame(columns=columns_)
		neg_pres_df = pd.DataFrame(columns=columns_)
  
		for row in predictions.itertuples():
			pres = getattr(row, 'predicted')
			rowidx = getattr(row, 'id')
			lid, rid = getattr(row, 'ltable_id'), getattr(row, 'rtable_id')
			lidx, ridx = mapA[lid], mapB[rid]
			new_line = [rowidx, lid, rid]
			lval = [tableA.loc[lidx, sch] for sch in schemas]
			rval = [tableB.loc[ridx, sch] for sch in schemas]
			new_line.extend(lval)
			new_line.extend(rval)
			if int(pres) == 1:
				pres_df.loc[len(pres_df)] = new_line
			else:
				neg_pres_df.loc[len(neg_pres_df)] = new_line
				

		if if_report_pre == True:
			pres_res = H[H['predicted'] == 1]
			pres_res.to_csv('check_prediction.csv', index=False)
		
		# save
		pres_df.to_csv(match_res_path, index=False)
		neg_pres_df.to_csv(neg_match_res_path, index=False)


	def apply_model(self, tottable, tableA, tableB, external_fea_extract=False, 
                    is_match_on_neg=False, default_blk_res_dir="", default_match_res_dir=""):
		'''
		Chunk the blocking result into pieces with size 1M
		then apply random forests concurrently
		each piece of the table has a new process
		'''

		args_ = [(i, tableA, tableB, external_fea_extract, False, default_blk_res_dir,
            	  default_match_res_dir) for i in range(tottable)]
		processes = []
  
		# Run subprocesses
		if tottable == 1:
			self._apply_model_worker(0, tableA, tableB, external_fea_extract=external_fea_extract, 
                            		 default_blk_res_dir=default_blk_res_dir, 
                               		 default_match_res_dir=default_match_res_dir)
		else:
			for i in range(tottable):
				papply = multiprocessing.Process(target=self._apply_model_worker, 
												 args=args_[i])
				processes.append(papply)
				papply.start()
			for pa in processes:
				pa.join()
				if pa.exitcode >0:
					raise ValueError(f"error in worker: {pa.exitcode}")

		# collect
		columns_ = ["_id", "ltable_id", "rtable_id"]
		schemas = list(tableA)[1:]
		lsch = ["ltable_" + sch for sch in schemas]
		rsch = ["rtable_" + sch for sch in schemas]
		columns_.extend(lsch)
		columns_.extend(rsch)
  
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
  
		res_df = pd.DataFrame(columns=columns_)
		neg_res_df = pd.DataFrame(columns=columns_)
		for tableid in range(tottable):
			path = '/'.join([match_res_dir, "match_res" + str(tableid) + ".csv"])
			neg_path = '/'.join([match_res_dir, "neg_match_res" + str(tableid) + ".csv"])
			pdf = pd.read_csv(path)
			neg_pdf = pd.read_csv(neg_path)
			# pdf.drop_duplicates(inplace=True)
			# neg_pdf.drop_duplicates(inplace=True)
			res_df = pd.concat([res_df, pdf], ignore_index=True)
			neg_res_df = pd.concat([neg_res_df, neg_pdf], ignore_index=True)

		stat_file_path = "/".join([match_res_dir, "stat.txt"])
		with open(stat_file_path, "w") as stat_file:
			print(tottable, len(res_df), file=stat_file)
   
		# flush
		# res_df.drop_duplicates(inplace=True)
		if is_match_on_neg == True:
			pre_res_df = pd.read_csv(match_res_path)
			res_df = pd.concat([pre_res_df, res_df], ignore_index=True)
		res_df.to_csv(match_res_path, index=False)
		neg_res_df.to_csv(neg_match_res_path, index=False)
		return res_df


	# utils
	def report_tree_to_text(self, path):
		with open(path, 'w') as tree_file:
			for tree in self.rf.clf.estimators_:
				r = export_text(tree, feature_names=self.features.feature_name)
				print(r, file=tree_file)

	def store_model(self, path):
		joblib.dump(self.rf, path)

	def load_model(self, path):
		self.rf = joblib.load(path)

	def add_attrs_blk_res(self, tableA, tableB, path):
		blk_res = pd.read_csv(path)
		
		rowsA = list(tableA.index)
		rowsB = list(tableB.index)
		mapA = {tableA.loc[row, 'id'] : row for row in rowsA}
		mapB = {tableB.loc[row, 'id'] : row for row in rowsB}
		
		attrs = list(tableA)[1:]
		rows = list(blk_res.index)

		blk_res = blk_res.rename(columns={'id1':'ltable_id', 'id2':'rtable_id'}, inplace=False)

		for attr in attrs:
			l_attr = 'ltable_' + attr
			r_attr = 'rtable_' + attr

			blk_res.insert(blk_res.shape[1], l_attr, '')
			blk_res.insert(blk_res.shape[1], r_attr, ' ')

			for row in rows:
				idA = blk_res.loc[row, 'ltable_id']
				idB = blk_res.loc[row, 'rtable_id']
				locA, locB = mapA[idA], mapB[idB]

				blk_res.loc[row, l_attr] = tableA.loc[locA, attr]
				blk_res.loc[row, r_attr] = tableB.loc[locB, attr]

		blk_res.to_csv(path, index=False)
	