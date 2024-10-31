import os
import pandas as pd
import numpy as np
import py_entitymatching as em
import copy
from py_entitymatching.catalog.catalog import Catalog
from collections import defaultdict


class DataSettings:
	'''
	Settings for data
	'''

	# all variables are static
	dataroot = '/yout/data/root'

	str_gt_10w = ['name', 'title', 'description']
	str_bt_5w_10w = []
	str_bt_1w_5w = []
	str_eq_1w = ['brand', 'category']
	numeric = ['price']
 
	supported_sim_funcs = ["jaccard", "cosine", "dice", "overlap", "lev_dist", "exm", "anm"]
	supported_tokenizers = ["dlm", "qgm", "alphanumeric", "wspace"]
	
	def __init__(self):
		pass

	@classmethod
	def print_data_root(cls):
		print(f"current data root: {cls.dataroot}")
  
	@classmethod
	def print_supported_sim_funcs(cls):
		for sim_func in cls.supported_sim_funcs:
			print(f"{sim_func}\t")
		print("\n")
	
	@classmethod
	def print_supported_tokenizers(cls):
		for tok in cls.supported_tokenizers:
			print(f"{tok}\t")
		print("\n")
  

class Dataset:
	'''
	A dataset may be csv or parquet fromat.
	'''
 

	def __init__(self):
		self.tableA = None
		self.tableB = None
		self.golds = None
		self.num_tables = 2


	def dump_files(self, path_table_A, path_table_B, path_gold):
		'''
		Flush files to buffer directory
		'''
  
		self.golds.to_csv(path_gold, index=False)
		self.tableA.to_csv(path_table_A, index=False)
		if self.num_tables == 3:
			self.tableB.to_csv(path_table_B, index=False)
   
   	
	def format_and_check(self):
		# format
		if "categb ry" in list(self.tableA):
			self.tableA.rename(columns={"categb ry": "category"}, inplace=True)
		if "categb ry" in list(self.tableB):
			self.tableB.rename(columns={"categb ry": "category"}, inplace=True)

		self.tableA.rename(columns={'_id':'id'}, inplace=True)
		self.tableB.rename(columns={'_id':'id'}, inplace=True)
		em.set_key(self.tableA, 'id')
		em.set_key(self.tableB, 'id')

		# check 
		columnA = list(self.tableA)
		columnB = list(self.tableB)
		if len(columnA) != len(columnB):
			raise ValueError(f"Inconsistent column numbers: {len(columnA)}, {len(columnB)}")
		for i in range(len(columnA)):
			if columnA[i] != columnB[i]:
				raise ValueError(f"Inconsistent schemas: {columnA[i]}, {columnB[i]}")


	def load_files(self, dir_path):
		'''
		Load files
		'''

		print("Current loading:", dir_path)
		files = os.listdir(dir_path)
		filename = files[0].split('.')
		filetype = filename[-1]

		for i in range(0, len(files)):
			print(files[i], end=' ')
			files[i] = dir_path + files[i]
		print('\n', end='')
		
		# Read
		if len(files) == 2:
			print("Single data table")
			if filetype == 'csv':
				self.golds = pd.read_csv(files[0])
				self.tableA = pd.read_csv(files[1])
			elif filetype == 'parquet':
				self.golds = pd.read_parquet(files[0], engine='pyarrow')
				self.tableA = pd.read_parquet(files[1], engine='pyarrow')
			else:
				raise ValueError("Error in file's type", filetype)
			self.tableB = self.tableA

		elif len(files) == 3:
			print("Dual data tables")
			self.num_tables = 3
			if filetype == 'csv':
				self.golds = pd.read_csv(files[0])
				self.tableA = pd.read_csv(files[1])
				self.tableB = pd.read_csv(files[2])
			elif filetype == 'parquet':
				self.golds = pd.read_parquet(files[0], engine='pyarrow')
				self.tableA = pd.read_parquet(files[1], engine='pyarrow')
				self.tableB = pd.read_parquet(files[2], engine='pyarrow')
			else:
				raise ValueError("Error in file's type", filetype)
		elif len(files) == 4:
			gold_path = dir_path + "gold.parquet"
			table_path = dir_path + "table_a.parquet"
			self.golds = pd.read_parquet(gold_path, engine='pyarrow')
			self.tableA = pd.read_parquet(table_path, engine='pyarrow')
			self.tableB = self.tableA

		else:
			raise FileExistsError(f"Error in directory: {dir_path}")
		
	
	def read_csv(self, path_table_A, path_table_B, path_gold, num_data=2):
		if os.path.exists(path_table_A) == False or\
		   os.path.exists(path_table_B) == False and\
		   num_data == 2:
			raise Exception("No clean csv files in buffer")
		elif os.path.exists(path_table_A) == False and\
			 num_data == 1:
			raise Exception("No clean csv files in buffer")

		self.golds = pd.read_csv(path_gold, low_memory=False)
		self.tableA = em.read_csv_metadata(path_table_A, key='_id')

		if num_data == 2:
			self.tableB = em.read_csv_metadata(path_table_B, key='_id')
		else:
			self.tableB = self.tableA
		
		self.tableA.rename(columns={'_id':'id'}, inplace=True)
		self.tableB.rename(columns={'_id':'id'}, inplace=True)
		em.set_key(self.tableA, 'id')
		em.set_key(self.tableB, 'id')

		columnA = list(self.tableA)
		columnB = list(self.tableB)
		if len(columnA) != len(columnB):
			raise Exception("Inconsistent column numbers")
		for i in range(len(columnA)):
			if columnA[i] != columnB[i]:
				raise Exception("Inconsistent column contents:", columnA[i], columnB[i])


	def load_golds(self, graph):
		'''
		Load the ground truth to the graph in RandomForest class
		'''

		row_indexs = list(self.golds.index)
		for index in row_indexs:
			idA = str(self.golds.loc[index, 'id1']) + 'A'
			idB = str(self.golds.loc[index, 'id2']) + 'B'
			graph.add_edge(idA, idB)
				

	def read_dir(self, dirname, graph):
		'''
		read from disk, then flush back to disk but to the buffer
		'''

		# Load from dataset
		self.load_files(dirname)
		self.format_and_check()
		self.load_golds(graph)
		# dump to buffer
		self.dump_files()
			