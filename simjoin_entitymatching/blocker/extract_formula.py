import numpy as np
from collections import defaultdict


class ExtractFormula:
	'''
	The filtering formula is shown as following:
		# of rules
		feature_name sign threshold
		...
		feature_name sign threshold

	It is temporarily stored in buffer/rules.txt for immediately using.
	It is permantly stored in data/rules/datasetname/rules.txt.
	'''

	def __init__(self):
		self.partial_num_features = 0 # straightforwardly select partial features no. from '1' to 'num_features'
		self.formulas = []
		self.feature_index = []
  
  
	def _print_one_rule(self, trigraph, i, buffer_rules):
		iidx = self.feature_index[i]
		print(trigraph.feature_list[i], end=' ', file=buffer_rules)
		if trigraph.feature_range[i][iidx][0][0] == 0:
			print('-', end=' ', file=buffer_rules)
			print("%.4f" % trigraph.feature_range[i][iidx][0][1], file=buffer_rules)
		else:
			print('+', end=' ', file=buffer_rules)
			print("%.4f" % trigraph.feature_range[i][iidx][0][1], file=buffer_rules)


	def flush_rules(self, trigraph, path, selected=[]):
		'''
		Save rules and flush them to buffer for blocker to read

		# Rewrite firstly
		# The key for interval in feature_range has changed!
		# The key tuple has also changed
		'''

		# Flush to buffer
		if len(selected) == 0:
			'''Flush all'''
			with open(path, 'w') as buffer_rules:
				minus_print_num = sum(1 for i in range(self.partial_num_features) if len(trigraph.feature_range[i]) == 0)
				print(self.partial_num_features - minus_print_num, file=buffer_rules)

				for i in range(self.partial_num_features):
					if len(trigraph.feature_range[i]) == 0:
						continue

					self._print_one_rule(trigraph, i, buffer_rules)

		else:
			'''Flush selected'''
			new_path = path.split('.')[:-1]
			path_selected = new_path[0] + "_selected.txt"

			with open(path_selected, 'w') as buffer_rules_selected:
				minus_print_num = sum(1 for i in selected if len(trigraph.feature_range[i]) == 0)
				print(len(selected) - minus_print_num, file=buffer_rules_selected)

				for i in selected:
					if len(trigraph.feature_range[i]) == 0:
						continue

					self._print_one_rule(trigraph, i, buffer_rules)


	def get_tree_nodes(self, trigraph, lower, upper, cur_rules, tree_visited, 
					   if_index=False, index=[]):
		'''
		Deduce current feature nodes can reach how many tree nodes
		Args:
			lower: smallest feature number
			upper: largest feature number
			index: a list of feature id
		'''
		if if_index == False:
			for i in range(lower, upper):
				# In sorting ranges, we drop ranges with 'inf'
				# If a feature only has ranges with 'inf'
				# then after sorting, it will not have ranges left
				if len(trigraph.feature_range[i]) == 0:
					continue

				iidx = self.feature_index[i]
				dict_pair = trigraph.feature_range[i][iidx]

				# Get rules
				for rule in dict_pair[1]:
					if rule not in cur_rules:
						if rule > len(trigraph.rule_node):
							raise ValueError(f"rule node: {rule}, total nodes: {trigraph.rule_node}")
						cur_rules.append(rule)
		else:
			for i in index:
				if len(trigraph.feature_range[i]) == 0:
					continue

				iidx = self.feature_index[i]
				dict_pair = trigraph.feature_range[i][iidx]

				# Get rules
				for rule in dict_pair[1]:
					if rule not in cur_rules:
						if rule > len(trigraph.rule_node):
							raise ValueError(f"rule node: {rule}, total nodes: {trigraph.rule_node}")
						cur_rules.append(rule)

		# If all rules visited
		cur_len = len(cur_rules)
		if cur_len == trigraph.num_rule:
			return trigraph.num_tree
		
		# Get rule nodes
		for i in range(cur_len):
			cur_rules[i] = trigraph.rule_node[cur_rules[i]]

		# Get edges
		for rnode in cur_rules:
			for v, w in trigraph.trigraph.adj[rnode].items():
				if v < trigraph.num_tree:
					tree_visited[v] += 1
		
		# Check each tree
		num_visited = 0
		for treen in trigraph.tree_node:
			if trigraph.trigraph.degree(treen) == tree_visited[treen]:
				num_visited += 1
			elif trigraph.trigraph.degree(treen) < tree_visited[treen]:
				raise ValueError("Error in tree node degree", treen, 
								 trigraph.trigraph.degree(treen), 
								 tree_visited[v])
			
		return num_visited


	def get_rules_cur_comb(self, trigraph, path):
		invertedList = defaultdict(set)
		for i in range(0, trigraph.num_feature):
			if len(trigraph.feature_range[i]) == 0:
				continue
			iidx = self.feature_index[i]
			dict_pair = trigraph.feature_range[i][iidx]
			for rule in dict_pair[1]:
				invertedList[rule].add(i)

		bucket = [tuple(v) for k, v in invertedList.items()]
		bucket = set(bucket)

		featureInvertedList = defaultdict(list)
		for v in bucket:
			if len(v) <= 1:
				continue
			for fea_id in v:
				tmp = [ffea_id for ffea_id in v]
				tmp.remove(fea_id)
				featureInvertedList[fea_id].append(set(tmp))

		with open(path, "w") as additional_rule:
			for k, v in featureInvertedList.items():
				self._print_one_rule(trigraph, k, fea_id, additional_rule)
				print(file=additional_rule)

				for vv in v:
					for fea_id in vv:
						self._print_one_rule(trigraph, fea_id, additional_rule)
					print(file=additional_rule)

	
	def get_connected_tree(self, trigraph, rules):
		'''
		Return the number of trees connected by a set of rules
		'''

		new_rules = []
		for i in range(len(rules)):
			new_rules.append(trigraph.rule_node[rules[i]])
		new_rules = list(set(new_rules))
  
		tree_visited = set()
		for rule in new_rules:
			for v, w in trigraph.trigraph.adj[rule].items():
				if v < trigraph.num_tree:
					tree_visited.add(v)
		return len(tree_visited)


	def move_index_basic(self, trigraph, cur_pos, if_all_end, if_drop_left=True):
		candidates, lcandidates = [], []

		for idx in range(trigraph.num_feature):
			new_pos = (cur_pos + idx) % trigraph.num_feature
			fstr = trigraph.feature_list[new_pos].split('_')

			if if_all_end[new_pos] == 0:
				candidates.append(new_pos)

		# If no more short to move
		if len(candidates) == 0:
			candidates = lcandidates	

		# Select the first
		cur_pos = candidates[0]

		# If we decide: Move to the very end, do not drop unreasonable
		if if_drop_left == False:
			if self.feature_index[cur_pos] < len(trigraph.feature_range[cur_pos]) - 1:
				self.feature_index[cur_pos] += 1
			if self.feature_index[cur_pos] == len(trigraph.feature_range[cur_pos]) - 1:
				if_all_end[cur_pos] = 1
		else:
			if self.feature_index[cur_pos] < trigraph.feature_div[cur_pos] - 1:
				self.feature_index[cur_pos] += 1
			if self.feature_index[cur_pos] == trigraph.feature_div[cur_pos] - 1:
				if_all_end[cur_pos] = 1

		# Move
		cur_pos += 1
		if cur_pos >= trigraph.num_feature:
			cur_pos = 0
		return cur_pos

	
	def move_index_greedy(self, trigraph, if_all_end, if_drop_left=True, last_move=-1):
		'''
		Move the feature index using greedy.
		Every time move the feature that will introduce the most rule nodes.
		'''

		candidates, lcandidates = [], []

		for i in range(trigraph.num_feature):
			# Normal
			if if_all_end[i] == 0 and i != last_move:
				candidates.append(i)

		# If no more short to move
		if len(candidates) == 0:
			if last_move != -1:
				candidates.append(last_move)
			else:
				return -1
			# candidates = lcandidates

		# load current reached rule nodes
		all_rule_nodes = set()
		for i in range(trigraph.num_feature):
			if len(trigraph.feature_range[i]) == 0:
				continue
			idx = self.feature_index[i]
			rule_nodes = set(trigraph.feature_range[i][idx][1])
			all_rule_nodes = all_rule_nodes.union(rule_nodes)
		cur_num_rule_nodes = len(all_rule_nodes)

		# Check next range
		max_tree_diff, max_rule_diff, max_fea_id = -1, -1, 0
		for fea in candidates:
			idx = self.feature_index[fea]
			# Get diff # of trees
			ntree1 = self.get_connected_tree(trigraph, trigraph.feature_range[fea][idx][1])
			ntree2 = self.get_connected_tree(trigraph, trigraph.feature_range[fea][idx+1][1])
			new_tree_diff = ntree2 - ntree1
			# Get diff of # of rules
			new_all_rule_nodes = set()
			for ii in range(trigraph.num_feature):
				if len(trigraph.feature_range[ii]) == 0:
					continue
				idx = self.feature_index[ii] if ii != fea else self.feature_index[ii] + 1
				rule_nodes = set(trigraph.feature_range[ii][idx][1])
				new_all_rule_nodes = new_all_rule_nodes.union(rule_nodes)
			new_rule_diff = len(new_all_rule_nodes) - cur_num_rule_nodes

			if new_rule_diff > max_rule_diff:
				max_rule_diff = new_rule_diff
				max_fea_id = fea
			# if new_tree_diff > max_tree_diff:
			# 	max_tree_diff = new_tree_diff
			# 	max_fea_id = fea

		# Move 
		self.feature_index[max_fea_id] += 1
		if if_drop_left == False and self.feature_index[max_fea_id] == len(trigraph.feature_range[max_fea_id]) - 1:
			if_all_end[max_fea_id] = 1
		elif if_drop_left == True and self.feature_index[max_fea_id] == trigraph.feature_div[max_fea_id] - 1:
			if_all_end[max_fea_id] = 1

		return max_fea_id


	def select_partial(self, trigraph, short_attrs):
		'''
		Try to select a fraction of features that are enough for blocking.

		Using two strategies, select the results that are shortest.
		'''
		
		short_features = []
		long_features = []
		features = []

		for i in range(trigraph.num_feature):
			fstr = trigraph.feature_list[i].split('_')

			# skip empty
			if len(trigraph.feature_range[i]) == 0:
				continue

			iidx = self.feature_index[i]
			features.append((i, len(trigraph.feature_range[i][iidx][1])))

			if fstr[0] in short_attrs:
				short_features.append((i, len(trigraph.feature_range[i][iidx][1])))
			else:
				long_features.append(i)

		selected1, selected2 = [], []

		# strategy 1
		# sort all features in descending order of the number of
		# rules they connected
		features = sorted(features, key=lambda x: x[1], reverse=True)

		for num_fea in range(len(long_features), len(features)+1):
			selected_feature = [f[0] for f in features[:num_fea]]
			print(selected_feature)

			# check for trees
			tree_visited = np.zeros((trigraph.num_tree,), dtype=int)
			cur_rules = []

			num_visited = self.get_tree_nodes(trigraph, 0, trigraph.num_feature, 
											 cur_rules, tree_visited, 
											 if_index=True, index=selected_feature)
				
			if num_visited > trigraph.num_tree / 2:
				print("\033[31mFind a fraction of rules!\033[0m", num_fea)
				selected1 = selected_feature
				break
			else:
				print("\033[31mProceed to add more attrs\033[0m", num_visited)

		# strategy 2
		# sort the short attrs in descending order of the number of
		# rules they connected
		short_attrs = sorted(short_attrs, key=lambda x: x[1], reverse=True)
		
		for num_short in range(len(short_features)+1):
			# first try all long attrs
			selected_feature = [lf for lf in long_features]

			# try some short attrs
			selected_feature = selected_feature + [sf[0] for sf in short_features[:num_short]]
			print(selected_feature)

			# check for trees
			tree_visited = np.zeros((trigraph.num_tree,), dtype=int)
			cur_rules = []

			num_visited = self.get_tree_nodes(trigraph, 0, trigraph.num_feature, 
											  cur_rules, tree_visited, 
											  if_index=True, index=selected_feature)
				
			if num_visited > trigraph.num_tree / 2:
				print("\033[31mFind a fraction of rules!\033[0m", num_short + len(long_features))
				selected2 = selected_feature
				break
			else:
				print("\033[31mProceed to add more short attrs\033[0m", num_visited)

		# choose
		if len(selected1) <= len(selected2):
			return selected1
		else:
			return selected2

 
	def dfs_optimal(self, trigraph, values, idx, selected, selected_value, selected_rules, max_val, max_selected, is_found):
		cur_rules = set()
		tree_visited = np.zeros((trigraph.num_tree,), dtype=int)
	
		# Get rule nodes
		for rule_buck in selected_rules:
			cur_rules.update(rule_buck)

		# Get edges
		for rnode in cur_rules:
			for v, _ in trigraph.trigraph.adj[rnode].items():
				if v < trigraph.num_tree:
					tree_visited[v] += 1
		
		# Check each tree
		num_visited = 0
		for treen in trigraph.tree_node:
			if trigraph.trigraph.degree(treen) == tree_visited[treen]:
				num_visited += 1
			elif trigraph.trigraph.degree(treen) < tree_visited[treen]:
				raise ValueError("Error in tree node degree", treen, 
								 trigraph.trigraph.degree(treen), 
								 tree_visited[v])
  
		if num_visited > trigraph.num_tree / 2:
			cur_val = 0
			for val in selected_value:
				cur_val += val
			if cur_val > max_val[0]:
				max_val[0] = cur_val
				max_selected[:] = selected
				is_found[0] = True
			return

		if idx >= trigraph.num_feature:
			return

		# select
		if trigraph.feature_div[idx] > 0:
			for i in range(trigraph.feature_div[idx]):
				selected.append((idx, i))
				selected_value.append(values[idx][i])
				selected_rules.append(trigraph.feature_range[idx][i][1])
				self.dfs_optimal(trigraph, values, idx + 1, selected, selected_value, selected_rules, max_val, max_selected, is_found)
				selected_rules.pop()
				selected_value.pop()
				selected.pop()

				if is_found[0]:
					is_found[0] = False
					break
 
		# do not select
		self.dfs_optimal(trigraph, values, idx + 1, selected, selected_value, selected_rules, max_val, max_selected, is_found)


	def get_optimal_rules_comb(self, trigraph, path_selected):
		'''
		The optimal rules comb, which is selecting rules to make all of them as tight as possible.
		It can be viewed as a multiple-Knapsack, where are "num_feature" groups of items, each of 
		group can only contributes at most 1 item to the comb.
		Also, the rule nodes which a feature connected are viewed as the weights, and the tighter
		range will result in a larger value.
  		'''
		
		max_val = trigraph.feature_div.max()
		values = np.zeros(shape=(trigraph.num_feature, max_val), dtype=int)
		for idx in range(trigraph.num_feature):
			for val in range(trigraph.feature_div[idx]):
				values[idx][val] = trigraph.feature_div[idx] - val
		
		selected, selected_value, selected_rules, max_val, max_selected = [], [], [], [0], []
		self.dfs_optimal(trigraph, values, 0, selected, selected_value, selected_rules, max_val, max_selected, [False])
  
		final_selected = []
		for tup in max_selected:
			final_selected.append(tup[0])
			self.feature_index[tup[0]] = tup[1]

		self.flush_rules(trigraph, path_selected, final_selected)


	# TODO: After sorting, the key with sign '2' only contains the interval value without bound value
	def extract(self, trigraph, if_drop_left=True, move_strategy=1):
		'''
		If the order of moving for each features influence the final results?

		Args:
			if_drop_left: Drop the unreasonable rules, like 'jac < xxx'
			move_strategy: 0 -> Basic move
						   1 -> Greedy move
		'''

		find, cur_pos = False, 0
		self.feature_index = np.zeros((trigraph.num_feature,), dtype=int)
		if_all_end = np.zeros((trigraph.num_feature,), dtype=int)

		if if_drop_left == False:
			for i in range(trigraph.num_feature):
				if_all_end[i] = int(len(trigraph.feature_range[i]) == 1 or len(trigraph.feature_range[i]) == 0)
		else:
			for i in range(trigraph.num_feature):
				if_all_end[i] = int(trigraph.feature_div[i] == 1 or trigraph.feature_div[i] == 0)

		# Pre-process: dict -> list of tuples
		for i in range(trigraph.num_feature):
			trigraph.feature_range[i] = [(k, v) for k,v in trigraph.feature_range[i].items()]
			# print(trigraph.feature_range[i])
		
		last_move = -1
		while find == False:
			if np.sum(if_all_end==1) == trigraph.num_feature:
				print("\033[31mReach to the end of every range\033[0m")
				break

			tree_visited = np.zeros((trigraph.num_tree,), dtype=int)

			cur_rules = []
			num_visited = self.get_tree_nodes(trigraph, 0, trigraph.num_feature, 
									 		  cur_rules, tree_visited)
			
			if num_visited > trigraph.num_tree / 2:
				break

			# Move index
			if move_strategy == 0:
				cur_pos = self.move_index_basic(trigraph, cur_pos, if_all_end, if_drop_left=if_drop_left)
			elif move_strategy == 1:
				last_move = self.move_index_greedy(trigraph, if_all_end, if_drop_left=if_drop_left, last_move=last_move)
			# print(self.feature_index)
		
		self.partial_num_features = trigraph.num_feature
