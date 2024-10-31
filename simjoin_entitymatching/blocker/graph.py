import networkx as nx
import py_entitymatching as em
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import pathlib


class TripartiteGraph:
	'''
	Graph used for feature selection
	'''


	def __init__(self):
		self.INF = 100000

		self.num_tree = 0
		self.num_rule = 0
		self.num_feature = 0

		self.tree_node = []
		self.rule_node = []
		self.feature_node = []

		self.rule_list = []
		self.rule_signature = defaultdict(int)
		self.rule_signature_trees = defaultdict(set)
		self.feature_list = []
		self.feature_range = []                                      # list of dicts: (sign, thres) -> [list of rules]
		self.feature_div = np.full((100000,), 0, dtype=int)          # div of the sign in a feature's range, the length of first sign
		self.feature_appeared = np.full((100000,), -1, dtype=int)

		self.trigraph = nx.Graph()
		self.bigraList1 = []                                         # edges between trees & rules
		self.bigraList2 = []                                         # edges between rules & features


	def generate_signature(self, temp_rule, cur_tree_node, cur_rule_node):
		'''
		return values: 
  			the first bool: whether add tree node; 
    		the second bool: whether add rule node
  		'''
    
		fea_id_list = [(self.feature_appeared[cur_fea], temp_rule[1][idx], idx) 
                 	   for idx, cur_fea in enumerate(temp_rule[0])]
		
		fea_id_list.sort(key=lambda tup : (tup[0], tup[1]))
		signature = ""

		for cur_fea in fea_id_list:
			fea_sig = str(temp_rule[0][cur_fea[2]])
			thres_sig = str(temp_rule[1][cur_fea[2]])
			sign_sig = str(temp_rule[2][cur_fea[2]])
			signature += fea_sig + thres_sig + sign_sig + "#"

		if signature not in self.rule_signature:
			self.rule_signature[signature] = cur_rule_node
			return signature, True, True
		else:
			if cur_tree_node not in self.rule_signature_trees[signature]:
				self.rule_signature_trees[signature].add(cur_tree_node)
				return signature, True, False
			else:
				return signature, False, False


	def extract_rules(self, tree, node, temp_rule):
		'''
		Args:
			temp_rule: [[], [], []] feature, threshold, sign (0: <, 1: >)
		'''

		# Reach the leaf
		if tree.children_left[node] == tree.children_right[node]:
			if tree.value[node][0, 0] > tree.value[node][0, 1]:
				return
			
			# If already existed
			tmp_tree_node = len(self.bigraList1) - 1
			tmp_rule_node = len(self.bigraList2)
			rule_sig_status = self.generate_signature(temp_rule, tmp_tree_node, tmp_rule_node)
			if rule_sig_status[1] == False and rule_sig_status[2] == False:
				return
			elif rule_sig_status[1] == True and rule_sig_status[2] == False:
				last_rule_node = self.rule_signature[rule_sig_status[0]]
				self.bigraList1[tmp_tree_node].append(last_rule_node)
				return
			
			# combine duplicate features
			new_rule = copy.deepcopy(temp_rule)
			self.rule_list.append(new_rule)
			self.bigraList2.append([])
			self.num_rule += 1

			# add edges: tree & rules
			cur_tree_node = len(self.bigraList1) - 1
			cur_rule_node = len(self.bigraList2) - 1
			self.bigraList1[cur_tree_node].append(cur_rule_node)
   
			for idx, cur_fea in enumerate(new_rule[0]):
				cur_thres, cur_sign = new_rule[1][idx], new_rule[2][idx]
				fea_id = self.feature_appeared[cur_fea]
				fea_key = (cur_sign, cur_thres)
				# add ranges
				self.feature_range[fea_id][fea_key].append(cur_rule_node)
				# add edges: rules & features
				self.bigraList2[cur_rule_node].append(fea_id)

			return
		
		cur_feature = tree.feature[node]
		cur_threshold = round(tree.threshold[node], 4)
		
		# add a new feature
		if self.feature_appeared[cur_feature] == -1:
			self.feature_appeared[cur_feature] = self.num_feature
			self.feature_list.append(cur_feature)
			self.feature_range.append(defaultdict(list))
			self.num_feature += 1

		# left
		temp_rule[0].append(cur_feature)
		temp_rule[1].append(cur_threshold)
		temp_rule[2].append(0)
		self.extract_rules(tree, tree.children_left[node], temp_rule)
		
		# right
		temp_rule[2].pop()
		temp_rule[2].append(1)
		self.extract_rules(tree, tree.children_right[node], temp_rule)

		# backtracking
		temp_rule[2].pop()
		temp_rule[1].pop()
		temp_rule[0].pop()


	def build_graph(self, rf, feature_names, if_report=False, default_feature_names_dir=""):
		self.num_tree = rf.n_estimators

		temp_rule = [[], [], []]
		for dtree_ in rf.estimators_:
			dtree = dtree_.tree_
			# Extract features & rules
			self.bigraList1.append([])
			self.extract_rules(dtree, 0, temp_rule)

		# Update node id
		self.tree_node = np.arange(0, self.num_tree, 1)
		self.rule_node = np.arange(self.num_tree, self.num_tree + self.num_rule, 1)
		self.feature_node = np.arange(self.num_tree + self.num_rule, 
								      self.num_tree + self.num_rule + self.num_feature, 1)
		for treev in self.tree_node.flat:
			self.trigraph.add_node(treev, bipartite=0)
		for rulev in self.rule_node.flat:
			self.trigraph.add_node(rulev, bipartite=1)
		for featurev in self.feature_node.flat:
			self.trigraph.add_node(featurev, bipartite=2)
		
		# Add edges 
		# Tree and rule
		for idx, nodes in enumerate(self.bigraList1):
			tree_node_id = idx
			for v in nodes:
				rule_node_id = self.rule_node[v]
				self.trigraph.add_edge(tree_node_id, rule_node_id)

		# Rule and feature
		for idx, nodes in enumerate(self.bigraList2):
			rule_node_id = self.rule_node[idx]
			for v in nodes:
				feature_node_id = self.feature_node[v]
				self.trigraph.add_edge(rule_node_id, feature_node_id)

		# Update list & Report
		self.feature_list.sort()
		for idx, fea in enumerate(self.feature_list):
			self.feature_list[idx] = feature_names.feature_name.loc[fea]

		if if_report == True:
			if default_feature_names_dir == "":
				cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
				feature_name_path = '/'.join([cur_parent_dir, "..", "..", "output", "buffer", "feature_names.txt"])
			else:
				default_feature_names_dir = default_feature_names_dir[ : -1] if default_feature_names_dir[-1] == '/' \
															   				 else default_feature_names_dir
				feature_name_path = '/'.join([default_feature_names_dir, "feature_names.txt"])
    
			with open(feature_name_path, "w") as fname_file:
				print(len(self.feature_list), file=fname_file)
				for fname in self.feature_list:
					print(fname, file=fname_file)


	def update_range2(self, idx, upper_bound):
		'''
		Sort & update the range dict taking feature into consideration
		'''

		lazy_tags = defaultdict(tuple)
		temp_range = defaultdict(list)
  
		for fea_key in self.feature_range[idx].keys():
			# Skip 'inf'
			if fea_key[1] == float('inf') or math.isnan(fea_key[1]):
				continue

			# New key
			new_key = ''
			if fea_key[0] == 0:
				new_key = (0, -fea_key[1])
			elif fea_key[0] == 1:
				new_key = (1, fea_key[1]-upper_bound)

			# Update
			lazy_tags[new_key] = fea_key
		
		# Lazy modify
		for key, val in lazy_tags.items():
			temp_range[key] = self.feature_range[idx][val]
		self.feature_range[idx].clear()

		# Sorted
		self.feature_range[idx] = {lazy_tags[key]: temp_range[key] for key in sorted(temp_range, reverse=True)}


	def special_case_update_range2(self, idx):
		'''
		For 'lev' and 'rdf', the value should be as small as possible
		'''

		lazy_tags = defaultdict(tuple)
		temp_range = defaultdict(list)

		for fea_key in self.feature_range[idx].keys():
			# Skip 'inf'
			if fea_key[1] == float('inf') or math.isnan(fea_key[1]):
				continue

			# New key
			new_key = ''
			if fea_key[0] == 0:
				new_key = (0, fea_key[1])
			elif fea_key[0] == 1:
				new_key = (1, -fea_key[1])

			# Update
			lazy_tags[new_key] = fea_key
		
		# Lazy modify
		for key, val in lazy_tags.items():
			temp_range[key] = self.feature_range[idx][val]
		self.feature_range[idx].clear()

		# Sorted
		self.feature_range[idx] = {lazy_tags[key]: temp_range[key] for key in sorted(temp_range)}


	def special_case_sort_ranges2(self, idx, sim_str):
		'''
		When we use some sim funcs out of supported funcs.
		update_range2 does not support them.
			1. monge_elkan: mel
			2. needleman_wunsch: nmw
			3. smith_waterman: sw
			4. jaro: jar
			5. jaro_winkler: jwn
			6. rel_diff: rdf
			7. lev_dist: lev_dist
		'''

		zero2one_sims = ['mel', 'jar', 'jwn']
		zero2one_small_sims = ['nmw', 'sw']
		one2inf_sims = ['rdf', 'lev_dist']

		if sim_str in zero2one_sims:
			self.update_range2(idx, 1)
		elif sim_str in zero2one_small_sims:
			self.update_range2(idx, 0)
		elif sim_str in one2inf_sims:
			self.special_case_update_range2(idx)


	def sort_ranges2(self):
		'''
		Comparing wirh sort_ranges:
			1. Drop features like 'jaro' & 'needleman'
			2. Sort p(c) in a more resonable way
			3. Do not consider interval for a feature range
		'''
		
		for i in range(self.num_feature):
			cur_fea = self.feature_list[i]
			fea_str = cur_fea.split('_')
			sim_str = fea_str[2]

			if sim_str == 'lev':
				sim_str = sim_str + '_' + fea_str[3]

			# where overlap is from 1 to inf
			zero2one_sim = ['exact', 'exm', 'abs', 'anm', 'lev_sim', 'jac', 'jaccard', 
				            'cos', 'cosine', 'dice', 'overlap']
	
			if sim_str in zero2one_sim:
				self.update_range2(i, 1)
			else:
				self.special_case_sort_ranges2(i, sim_str)


	def update_range_rule_node(self):
		'''
		After sorting, we need to add rule node in tighter range to the looser one.
			e.g., (1, 0.9): [rule_id1] & (1, 0.8): [rule_id2]
			it's obviously that if (1, 0.8) cannot be satisfied, rule_id1 cannot be satisfied
			use set to avoid duplicate
		'''

		for i in range(self.num_feature):
			div_flag, prev_key= False, ''

			for fidx, fkey in enumerate(self.feature_range[i].keys()):
				if fidx == 0:
					prev_key = fkey
					continue

				# If reach end of a sign
				if fkey[0] == prev_key[0]:
					div_flag = False
				else:
					div_flag = True
					self.feature_div[i] = fidx

				# Update rule id
				if div_flag == False:
					tmp_set = list(set(self.feature_range[i][prev_key] + self.feature_range[i][fkey]))
					self.feature_range[i][fkey] = tmp_set
     
				prev_key = fkey


	# Output APIs
	def visualize_graph(self):
		fig = plt.figure()
		nodes = self.trigraph.nodes()

		# for each of the parts create a set 
		nodes_0  = set([n for n in nodes if  self.trigraph.nodes[n]['bipartite']==0])
		nodes_1  = set([n for n in nodes if  self.trigraph.nodes[n]['bipartite']==1])
		nodes_2  = set([n for n in nodes if  self.trigraph.nodes[n]['bipartite']==2])

		# set the location of the nodes for each set
		pos = dict()
		pos.update( (n, (1, i)) for i, n in enumerate(nodes_0) ) # put nodes from X at x=1
		pos.update( (n, (2, i)) for i, n in enumerate(nodes_1) ) # put nodes from Y at x=2
		pos.update( (n, (3, i)) for i, n in enumerate(nodes_2) ) # put nodes from X at x=1

		nx.draw(self.trigraph, pos=pos)
		# plt.savefig('buffer/graph.png', format="PNG")


	def report_ranges(self, path):
		with open(path, 'w') as range_file:
			for i in range(self.num_feature):
				print(self.feature_list[i], file=range_file)
				for key, val in self.feature_range[i].items():
					print(key, val, end=' ## ', file=range_file)
				print(file=range_file)


	def graph_stat(self):
		print("---------- tripartite graph stat ----------")
		print(f"number of tree nodes: {self.num_tree}")
		print(f"number of rule nodes: {self.num_rule}")
		print(f"number of feature nodes: {self.num_feature}")
		print("---------- tripartite graph stat ----------")