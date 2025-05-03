# Sensitivity

We divide the sensitivity analysis for three parts: **blocker**, **matcher** and **value matcher**.

0. We will select the longest attribute as the most representative attribute for sample / topk (<span style="color:orange">This may be specified by users if there are mutiple long attributes with nearly the same average length).
1. The attributes' types may need to be adjustment according to different datasets. Since the **py_entitymatching**'s attribute type auto-inference may not be precise

## Blocker
0. We use heap to maintain up to 1e7 results (each thread) for each join algorithms with the largest (smallest for edit distance) sim join values.
1. We may restrict the join result set's size in addition to that 1e7 restriction to reduce the final output size. However, on Megallen datasets which are small, we will lose roughly ~8% recall.
2. For the final TopK, we may select the word-weighted (by idf) similarity functions or unweighted, the weighted one has a slightly better performance on Megallen datasets, roughly ~1%. But for secret, unweighted is better (Also for the join TopK in 2, we could also has 2 versions).
3. For edit distance join, we can not keep a heap to maintain the top result since it will take roughly 8 hours on secret datasets.  
4. The practically best way is to only maintain value for attributes which are "str_bt_5w_10w" or "str_gt_10w"

## Matcher
* <span style="color:green">parameters are same for all datasets</span>.
* <span style="color:yellow">parameters depend on datasets' sizes / type (self-join or RS-join)</span>.
* <span style="color:pink">parameters needs to be adjusted according to different datasets</span>.

Parameters: <span style="color:pink">blocking_attr</span>, <span style="color:green">sample_strategy</span>, <span style="color:green">training_strategy</span>, <span style="color:green">move_strategy</span>, <span style="color:green">num_tree</span>, <span style="color:green">sample_size</span>, <span style="color:yellow">inmemory</span>, <span style="color:green">ground_truth_label</span>, <span style="color:pink">cluster_tau</span>, <span style="color:pink">sample_tau</span>, <span style="color:pink">step2_tau</span>, <span style="color:yellow">num_data</span>.

* <span style="color:pink">blocking_attr</span> As stated above, it needs to be selected by a human sometimes.
* <span style="color:pink">cluster_tau, sample_tau, step2_tau</span> We set different values for Megallen and synethic datasets, but for all datasets with in Megallen (synethic), they share the same values.
* <span style="color:yellow">inmemory</span> We set this to be true (false) if the table's size is small (large), or if you could afford the expense for training value matcher, then set it always to be true.
* <span style="color:yellow">num_data</span> This parameter simply depends whether the dataset is self-join or RS-join.

## Value matcher
0. The threshold for value matcher is default to 0.8 for all settings. 

# Issues
## Fixed at this stage
1. Set join parallel on large datasets with large threshold (e.g., 0.97+) will fail 

<span style="color:orange">this is because of adaptive grouping, but further investigation is needed. See the details in "notes.md"</span>

2. there may be some miss-commented lines in set join source files 

<span style="color:orange">set join works at this stage</span>

3. "buffer" folder should contain: *clean_A.csv, clean_B.csv, gold.csv, sample_res.csv, feature_name.txt*

<span style="color:orange">"buffer" folder currently is well-origanized</span>

## To be careful
1. think about one attribute share different types in different datasets
2. check all marcos (MAX_PAIR_SIZE & MAX_RES_SIZE refers to the exact same thing)
3. make three join algorithm classes consistent, e.g., write a base class for them to make the class declaration consistent
4. there are still some warnings