# author: Yunqi Li
# contact: liyunqixa@gmail.com
import simjoin_entitymatching.matcher.random_forest as randf
import simjoin_entitymatching.value_matcher.doc2vec as docv
import simjoin_entitymatching.blocker.graph as bg
from simjoin_entitymatching.feature.feature import run_feature_lib, run_feature_megallen
from typing import Literal
import pathlib


def train_model(tableA, tableB, gold_graph, blocking_attr, 
                model_path, tree_path, range_path, num_tree, 
                sample_size, ground_truth_label, 
                write_all_features=True, write_used_features=False,
                training_strategy = Literal['basic', 'tuning', 'active'], 
                inmemory = Literal[0, 1], num_data = Literal[1, 2], 
                at_ltable=None, at_rtable=None, dataname=None, 
                default_sample_res_dir="", default_vmatcher_dir="", 
                default_gold_dir="", default_icv_dir="", 
                default_feature_name_dir=""):
    rf = randf.RandomForest()
    doc2vec = docv.Doc2Vec(inmemory)

    rf.graph = gold_graph

    # sample
    # run_sample_exe(blocking_attr, sample_strategy, cluster_tau, sample_tau, step2_tau, num_data)

    # value matcher
    doc2vec.load_sample_res(tableA, tableB, default_sample_res_dir)

    if num_data == 1:
        doc2vec.train_and_save(blocking_attr, tableA, None, default_vmatcher_dir)
    else:
        doc2vec.train_and_save(blocking_attr, tableA, tableB, default_vmatcher_dir)

    # Sample
    doc2vec.apply_sample(blocking_attr, 0.8, ground_truth_label=ground_truth_label, 
                         default_icv_dir=default_icv_dir, 
                         default_gold_dir=default_gold_dir, 
                         default_sample_res_dir=default_sample_res_dir)

    rf.sample_data(tableA, tableB, default_sample_res_dir)
    # Features selection
    rf.generate_features(tableA, tableB, at_ltable=at_ltable, at_rtable=at_rtable, dataname=dataname,
                         default_output_dir=default_feature_name_dir, wrtie_fea_names=write_all_features)

    # Train
    if training_strategy == "active":
        rf.train_model_active(tableA, tableB, num_tree, sample_size)
    elif training_strategy == "basic":
        rf.train_model_normal(tableA, tableB, num_tree, sample_size)
    elif training_strategy == "tuning":
        rf.train_model_tuning(tableA, tableB, num_tree, sample_size)
    rf.report_tree_to_text(tree_path)
    rf.store_model(model_path)
    
    print(f"total sample: {rf.num_total}, training sample: {rf.num_training}")
    
    trigraph = bg.TripartiteGraph()
    # Build
    trigraph.build_graph(rf.rf.clf, rf.features, if_report=write_used_features, default_feature_names_dir=default_feature_name_dir)
    # Sort
    trigraph.sort_ranges2()
    trigraph.update_range_rule_node()
    # Output
    trigraph.report_ranges(range_path)
    trigraph.graph_stat()
    
    return rf, trigraph


def match_via_megallen_features(tableA, tableB, gold_graph, gold_len, model_path, is_interchangeable, flag_consistent, 
                               at_ltable=None, at_rtable=None, group=None, cluster=None, default_blk_res_dir="", 
                               default_match_res_dir="", default_fea_names_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        path_block_stat = "/".join([cur_parent_dir, "..", "..", "output", "blk_res", "stat.txt"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        path_block_stat = "/".join([default_blk_res_dir, "stat.txt"])
    
    with open(path_block_stat, "r") as stat_file:
        stat_line = stat_file.readlines()
        total_table, _ = (int(val) for val in stat_line[0].split())

    rf = randf.RandomForest()
    rf.graph = gold_graph
    rf.load_model(model_path)

    # Features selection
    rf.generate_features(tableA, tableB, at_ltable=at_ltable, at_rtable=at_rtable, default_output_dir=default_fea_names_dir, 
                         wrtie_fea_names=True)

    print("~~~Start matching~~~", flush=True)
    
    run_feature_megallen(tableA, tableB, feature_tab=rf.features, total_table=total_table, is_interchangeable=is_interchangeable, 
                         flag_consistent=flag_consistent, attrs_after=None, group=group, cluster=cluster, 
                         default_blk_res_dir=default_blk_res_dir, n_jobs=-1)
    
    rfpres = rf.apply_model(total_table, tableA, tableB, external_fea_extract=False, 
                            default_blk_res_dir=default_blk_res_dir, 
                            default_match_res_dir=default_match_res_dir)
    rf.get_recall(rfpres, gold_len, external_report=True)


def match_via_cpp_features(tableA, tableB, gold_graph, gold_len, model_path, is_interchangeable, flag_consistent, 
                           numeric_attr, at_ltable=None, at_rtable=None, default_blk_res_dir="", default_match_res_dir="", 
                           default_fea_names_dir="", default_icv_dir=""):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        path_block_stat = "/".join([cur_parent_dir, "..", "..", "output", "blk_res", "stat.txt"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        path_block_stat = "/".join([default_blk_res_dir, "stat.txt"])
    
    with open(path_block_stat, "r") as stat_file:
        stat_line = stat_file.readlines()
        total_table, _ = (int(val) for val in stat_line[0].split())

    rf = randf.RandomForest()
    rf.graph = gold_graph
    rf.load_model(model_path)

    # Features selection
    rf.generate_features(tableA, tableB, at_ltable=at_ltable, at_rtable=at_rtable, default_output_dir=default_fea_names_dir, 
                         wrtie_fea_names=True)

    print("~~~Start matching~~~", flush=True)
    
    schemas = list(tableA)[1:]
    schemas = [attr for attr in schemas if attr not in numeric_attr]
	
    run_feature_lib(is_interchangeable=is_interchangeable, flag_consistent=flag_consistent, total_table=total_table, total_attr=len(schemas), 
                    attrs=schemas, usage="match", default_fea_vec_dir=default_blk_res_dir, default_icv_dir=default_icv_dir, 
                    default_fea_names_dir=default_fea_names_dir)
 
    rfpres = rf.apply_model(total_table, tableA, tableB, external_fea_extract=True, 
                            default_blk_res_dir=default_blk_res_dir, 
                            default_match_res_dir=default_match_res_dir)
    rf.get_recall(rfpres, gold_len, external_report=True)
    
    
def match_on_neg_pres(tableA, tableB, gold_graph, gold_len, model_path, is_interchangeable, flag_consistent, 
                      numeric_attr, at_ltable=None, at_rtable=None, default_match_res_dir="",
                      default_fea_names_dir="", default_icv_dir=""):
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

    rf = randf.RandomForest()
    rf.graph = gold_graph
    rf.load_model(model_path)

    # Features selection
    rf.generate_features(tableA, tableB, at_ltable=at_ltable, at_rtable=at_rtable, default_output_dir=default_fea_names_dir, 
                         wrtie_fea_names=True)

    print("~~~Start matching~~~", flush=True)
    
    schemas = list(tableA)[1:]
    schemas = [attr for attr in schemas if attr not in numeric_attr]
	
    default_fea_vec_dir = path_match_stat.rsplit('/', 1)[0]
    print(f"neg fea vec writing to ... {default_fea_vec_dir}")
    run_feature_lib(is_interchangeable=is_interchangeable, flag_consistent=flag_consistent, total_table=total_table, total_attr=len(schemas), 
                    attrs=schemas, usage="match", default_fea_vec_dir=default_fea_vec_dir, default_res_tab_name="neg_match_res", 
                    default_icv_dir=default_icv_dir, default_fea_names_dir=default_fea_names_dir)

    rfpres = rf.apply_model(total_table, tableA, tableB, external_fea_extract=True, is_match_on_neg=True,
                            default_blk_res_dir=default_fea_vec_dir, 
                            default_match_res_dir=default_match_res_dir)
    rf.get_recall(rfpres, gold_len, external_report=True)
    
    
def debug_rf_matcher(tableA, tableB, gold_graph, gold_len, model_path, is_interchangeable, flag_consistent, 
                     numeric_attr, at_ltable=None, at_rtable=None, default_match_res_dir="",
                     default_fea_names_dir="", default_icv_dir=""):
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

    rf = randf.RandomForest()
    rf.graph = gold_graph
    rf.load_model(model_path)

    # Features selection
    rf.generate_features(tableA, tableB, at_ltable=at_ltable, at_rtable=at_rtable, default_output_dir=default_fea_names_dir, 
                         wrtie_fea_names=True)

    print("~~~Start matching~~~", flush=True)
    
    schemas = list(tableA)[1:]
    schemas = [attr for attr in schemas if attr not in numeric_attr]
	
    default_fea_vec_dir = path_match_stat.rsplit('/', 1)[0]
    print(f"neg fea vec writing to ... {default_fea_vec_dir}")
    run_feature_lib(is_interchangeable=is_interchangeable, flag_consistent=flag_consistent, total_table=total_table, total_attr=len(schemas), 
                    attrs=schemas, usage="match", default_fea_vec_dir=default_fea_vec_dir, default_res_tab_name="neg_match_res", 
                    default_icv_dir=default_icv_dir, default_fea_names_dir=default_fea_names_dir)

    rfpres = rf.apply_model(total_table, tableA, tableB, external_fea_extract=True, 
                            default_blk_res_dir=default_fea_vec_dir, 
                            default_match_res_dir=default_match_res_dir)
    rf.get_recall(rfpres, gold_len)