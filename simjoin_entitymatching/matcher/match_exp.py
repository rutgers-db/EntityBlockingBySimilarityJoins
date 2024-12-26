# author: Yunqi Li
# contact: liyunqixa@gmail.com
'''
this file can only be used in experiments
'''
import py_entitymatching as em
from typing import Literal
import pandas as pd
import os
import six
import joblib
import networkx as nx
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import gensim.utils
from numpy import dot
from numpy.linalg import norm

import py_entitymatching.utils.generic_helper as gh
from py_entitymatching.debugmatcher.debug_gui_utils import _get_metric

import simjoin_entitymatching.utils.path_helper as ph
import simjoin_entitymatching.utils.visualize_helper as vis
import simjoin_entitymatching.matcher.random_forest as randf
from simjoin_entitymatching.feature.feature import run_feature_lib, run_feature_megallen
from simjoin_entitymatching.value_matcher.interchangeable import group_interchangeable, group_interchangeable_fasttext
from simjoin_entitymatching.matcher.search import filter_match_res_memory, filter_match_res_disk


def _label_cand(gold_graph, tab):
    tab.insert(tab.shape[1], 'label', 0)
    
    for index in tab.index:
        id1 = str(tab.loc[index, 'ltable_id']) + 'A'
        id2 = str(tab.loc[index, 'rtable_id']) + 'B'
        if gold_graph.has_edge(id1, id2) == True:
            tab.loc[index, 'label'] = 1
            
    return tab


def _set_metadata(dataframe, key, fk_ltable, fk_rtable, ltable, rtable):
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
    
    return dataframe


def _print_eval_summary(eval_summary, filep):
    m = _get_metric(eval_summary)
    for key, value in six.iteritems(m):
        print(key + " : " + value, file=filep)
        
        
def _eval_results(pred_df, tableA, tableB, filep):
    _set_metadata(pred_df, "_id", "ltable_id", "rtable_id", tableA, tableB)
    eval_result = em.eval_matches(pred_df, 'label', 'predicted')
    print("------ report exp model results ------")
    _print_eval_summary(eval_result, filep=filep)
    print("------ end ------")
    
    
def _get_tab_schemas(table):
    columns_ = ["_id", "ltable_id", "rtable_id"]
    schemas = list(table)[1:]
    lsch = ["ltable_" + sch for sch in schemas]
    rsch = ["rtable_" + sch for sch in schemas]
    columns_.extend(lsch)
    columns_.extend(rsch)
    columns_.extend(["label", "predicted"])
    return columns_


def _get_tab_row_inv_map(table):
    rows = list(table.index)
    inv_map = {table.loc[rowidx, "id"] : rowidx for rowidx in rows}
    return inv_map


def _save_one_line(row, tableA, tableB, mapA, mapB):
    rowidx = row["_id"]
    lid, rid = row["ltable_id"], row["rtable_id"]
    lidx, ridx = mapA[lid], mapB[rid]
    new_line = [rowidx, lid, rid]
    
    lval = [tableA.loc[lidx, sch] for sch in list(tableA.columns)[1:]]
    rval = [tableB.loc[ridx, sch] for sch in list(tableB.columns)[1:]]
    new_line.extend(lval)
    new_line.extend(rval)
    
    new_line.extend([row["label"], row["predicted"]])
    
    return new_line, row["predicted"]


def _save_pred_to_df(predictions, tableA, tableB):
    columns_ = _get_tab_schemas(tableA)

    mapA = _get_tab_row_inv_map(tableA)
    mapB = _get_tab_row_inv_map(tableB)
    
    tab = pd.DataFrame(columns=columns_)
    
    for _, row in predictions.iterrows():
        new_line, _ = _save_one_line(row, tableA, tableB, mapA, mapB)
        tab.loc[len(tab)] = new_line
        
    return tab
    

def _save_neg_match_res(predictions, tableA, tableB):
    # Save predictions
    columns_ = _get_tab_schemas(tableA)

    mapA = _get_tab_row_inv_map(tableA)
    mapB = _get_tab_row_inv_map(tableB)
    
    pres_df = pd.DataFrame(columns=columns_)
    neg_pres_df = pd.DataFrame(columns=columns_)
    tot_df = pd.DataFrame(columns=columns_)

    for _, row in predictions.iterrows():
        new_line, pres = _save_one_line(row, tableA, tableB, mapA, mapB)
        
        if int(pres) == 1:
            pres_df.loc[len(pres_df)] = new_line
        else:
            neg_pres_df.loc[len(neg_pres_df)] = new_line
        tot_df.loc[len(tot_df)] = new_line
    
    # save
    tot_df.to_csv("output/exp/tot_match_res0.csv", index=False)
    pres_df.to_csv("output/exp/match_res0.csv", index=False)
    pres_df.to_csv("output/exp/match_res.csv", index=False)
    neg_pres_df.to_csv("output/exp/neg_match_res0.csv", index=False)
    
    return tot_df, pres_df, neg_pres_df
    
    
def _save_second_match_res(predictions, prev_predictions, idx_map, tableA, tableB):
    # Save predictions
    columns_ = _get_tab_schemas(tableA)
    columns_.extend(["first proba", "second proba", "cosine"])
    
    mapA = _get_tab_row_inv_map(tableA)
    mapB = _get_tab_row_inv_map(tableB)
    
    true_positive = pd.DataFrame(columns=columns_)
    false_positive = pd.DataFrame(columns=columns_)
    
    value_matcher = joblib.load("simjoin_entitymatching/value_matcher/model/doc2vec_title.joblib")
            
    for _, row in predictions.iterrows():
        lid = row["ltable_id"]
        rid = row["rtable_id"]
        prev_ridx = idx_map[(lid, rid)]
        
        if prev_predictions.loc[prev_ridx, "predicted"] == 1:
            raise ValueError(f"error in previous prediction")

        # change prediction
        new_line, _ = _save_one_line(row, tableA, tableB, mapA, mapB)
        new_line.extend([prev_predictions.loc[prev_ridx, "proba"], row["proba"]])
        
        # get cosine similarity
        l_title = tableA.loc[mapA[lid], "title"]
        r_title = tableB.loc[mapB[rid], "title"]
        l_docs = gensim.utils.simple_preprocess(l_title)
        r_docs = gensim.utils.simple_preprocess(r_title)
        l_vec = value_matcher.infer_vector(l_docs)
        r_vec = value_matcher.infer_vector(r_docs)
        cos_sim = dot(l_vec, r_vec) / (norm(l_vec) * norm(r_vec))
        new_line.append(cos_sim)
        
        # true positive
        if row["label"] == 1:
            true_positive.loc[len(true_positive)] = new_line
        # false positive
        elif row["label"] == 0:
            false_positive.loc[len(false_positive)] = new_line
    
    # save
    true_positive.to_csv("output/debug/true_positive_second.csv", index=False)
    false_positive.to_csv("output/debug/false_positive_second.csv", index=False)


def split_and_dump_data(tableA, tableB, gold_graph, external_extract=False, T_index=None, E_index=None, 
                        impute_strategy=Literal["mean", "constant", "none"], default_blk_res_dir=""):
    # read stat
    path_blk_res_stat = ph.get_blk_res_stat_path(default_blk_res_dir)
    with open(path_blk_res_stat, "r") as stat_file:
        stat_line = stat_file.readlines()
        total_table, _ = (int(val) for val in stat_line[0].split())
        
    # dump
    exp_dir = "output/exp"
    is_exist = os.path.exists(exp_dir)
    if not is_exist:
        os.mkdir(exp_dir)
        print(f"experiments directory is not presented, creat a new one")

    if external_extract == True:
        # read H1
        for tab_id in range(total_table):
            path_fea_vec = ph.get_chunked_fea_vec_path(tab_id, default_blk_res_dir)
            H1 = em.read_csv_metadata(path_fea_vec, 
                                    key="id", 
                                    ltable=tableA, rtable=tableB,
                                    fk_ltable="ltable_id", fk_rtable="rtable_id")
            H1.rename(columns={"id": "_id"}, inplace=True)
            em.set_key(H1, "_id")
            
            # label and impute
            _label_cand(gold_graph, H1)
            if impute_strategy == "mean":
                H1 = em.impute_table(H1, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="mean")
            elif impute_strategy == "constant":
                H1 = em.impute_table(H1, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)
                
            # concat
            tot_H1 = H1 if tab_id == 0 else pd.concat([tot_H1, H1], ignore_index=True)
    else:
        # read H2
        for tab_id in range(total_table):
            path_fea_vec = ph.get_chunked_fea_vec_path(tab_id, default_blk_res_dir)
            path_fea_vec = path_fea_vec[ : -4] + "_py.csv"
            H2 = em.read_csv_metadata(path_fea_vec, 
                                    key="_id", 
                                    ltable=tableA, rtable=tableB,
                                    fk_ltable="ltable_id", fk_rtable="rtable_id")
            
            # label and impute    
            _label_cand(gold_graph, H2)
            if impute_strategy == "mean":
                H2 = em.impute_table(H2, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="mean")
            elif impute_strategy == "constant":
                H2 = em.impute_table(H2, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)
                
            # concat
            tot_H2 = H2 if tab_id == 0 else pd.concat([tot_H2, H2], ignore_index=True)
        
    # split
    # T: train, E: test
    # T : E = 1 : 1
    if external_extract == True:
        tot_H1 = _set_metadata(tot_H1, key="_id", fk_ltable="ltable_id", fk_rtable="rtable_id", ltable=tableA, rtable=tableB)
        
        T2 = tot_H1.loc[T_index.values]
        E2 = tot_H1.loc[E_index.values]
        
        path_train2 = "/".join([exp_dir, "train2.csv"])
        path_test2 = "/".join([exp_dir, "test2.csv"])
        T2.to_csv(path_train2, index=False)
        E2.to_csv(path_test2, index=False)
        
        return T2, E2
    else:
        tot_H2 = _set_metadata(tot_H2, key="_id", fk_ltable="ltable_id", fk_rtable="rtable_id", ltable=tableA, rtable=tableB)
        
        TE = em.split_train_test(tot_H2, train_proportion=0.5, random_state=0)
        T = TE["train"]
        E = TE["test"]
        
        path_train = "/".join([exp_dir, "train1.csv"])
        path_test = "/".join([exp_dir, "test1.csv"])
        T.to_csv(path_train, index=False)
        E.to_csv(path_test, index=False)
        
        return T, E


def train_model(T, num_tree=10):
    exp_rf = em.RFMatcher(name='RF', random_state=0, n_estimators=num_tree, class_weight='balanced')

    # corss validation to avoid overfitting
    # 5-fold
    param_grid = {
        "max_depth": [None, 10, 20, 30],  # Maximum depth of the tree
        "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }

    grid_search = GridSearchCV(estimator=exp_rf.clf, param_grid=param_grid,
                                cv=5, n_jobs=-1, verbose=0)
    
    # process the feature tables to numpy ndarray
    exclude_attrs = ["_id", "ltable_id", "rtable_id", "label"]
    target_attr = "label"
    attributes_to_project = gh.list_diff(list(T.columns), exclude_attrs)

    X_train = T[attributes_to_project]
    y_train = T[target_attr]
    X_train, y_train = exp_rf._get_data_for_sklearn(X_train, y_train)

    grid_search.fit(X_train, y_train)

    # train
    exp_rf.clf = grid_search.best_estimator_
    return exp_rf
    
    
def apply_model(tableA, tableB, exp_rf, E, filep, is_concat=False, prev_pred=None):
    _set_metadata(E, "_id", "ltable_id", "rtable_id", tableA, tableB)
    # Predict on E
    predictions = exp_rf.predict(table=E, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
                                 append=True, target_attr='predicted', inplace=False, 
                                 return_probs=True, probs_attr='proba')
    print(predictions.head())
    eval_pred = predictions
    
    if is_concat:        
        idx_map = defaultdict()
        row_index = prev_pred.index
        for ridx in row_index:
            lid = prev_pred.loc[ridx, "ltable_id"]
            rid = prev_pred.loc[ridx, "rtable_id"]
            idx_map[(lid, rid)] = ridx
            
        predictions = predictions[predictions["predicted"] == 1]
        
        prev_pred_copy = prev_pred.copy()
        for _, row in predictions.iterrows():
            lid = row["ltable_id"]
            rid = row["rtable_id"]
            prev_ridx = idx_map[(lid, rid)]
            prev_pred_copy.loc[prev_ridx, "predicted"] = 1
            
        _set_metadata(prev_pred_copy, "_id", "ltable_id", "rtable_id", tableA, tableB)
        copy_eval_result = em.eval_matches(prev_pred_copy, 'label', 'predicted')
        print("------ report exp model results ------")
        _print_eval_summary(copy_eval_result, filep=filep)
        print("------ end ------")
        
        # save and slim
        pred_df = _save_pred_to_df(predictions, tableA, tableB)
        slim_pred = filter_match_res_memory(match_tab=pred_df, attr="title", K=1, 
                                            threshold=0.8, search_strategy="exact")
        
        # save results for debug
        _save_second_match_res(predictions, prev_pred, idx_map, tableA, tableB)
            
        predictions = slim_pred[slim_pred["predicted"] == 1]
        print(f"after slimmed : {len(pred_df)}, {len(predictions)}")
        # save results for debug
        predictions.insert(predictions.shape[1], "proba", 0)
        # _save_second_match_res(predictions, prev_pred, idx_map, tableA, tableB)
        
        for _, row in predictions.iterrows():
            lid = row["ltable_id"]
            rid = row["rtable_id"]
            prev_ridx = idx_map[(lid, rid)]
            prev_pred.loc[prev_ridx, "predicted"] = 1
        
        _set_metadata(prev_pred, "_id", "ltable_id", "rtable_id", tableA, tableB)
        
        eval_pred = prev_pred
    
    # Evaluate the predictions
    eval_result = em.eval_matches(eval_pred, 'label', 'predicted')
    print("------ report exp model results ------")
    _print_eval_summary(eval_result, filep=filep)
    print("------ end ------")
    
    tot_df, pres_df, neg_pres_df = _save_neg_match_res(eval_pred, tableA, tableB)
    
    return predictions, tot_df, pres_df, neg_pres_df
    
    
def run_experiments(tableA, tableB, rep_attr, at_ltable, at_rtable, gold_graph, filep, impute_strategy=Literal["mean", "constant", "none"]):
    # select features
    rf = randf.RandomForest()
    rf.generate_features(tableA, tableB, at_ltable=at_ltable, at_rtable=at_rtable, wrtie_fea_names=True)
    
    # generate features
    path_block_stat = ph.get_blk_res_stat_path()
    with open(path_block_stat, "r") as stat_file:
        stat_line = stat_file.readlines()
        total_table, _ = (int(val) for val in stat_line[0].split())
    
    run_feature_megallen(tableA, tableB, feature_tab=rf.features, total_table=total_table, is_interchangeable=0, 
                         flag_consistent=0, attrs_after=None, group=None, cluster=None, n_jobs=-1)
    
    # split original data
    train, test = split_and_dump_data(tableA, tableB, gold_graph, impute_strategy=impute_strategy)
    
    # train the model on half portion of the blk_res
    model = train_model(train)
    
    # apply on test result
    pred1, _, pres_df1, _ = apply_model(tableA, tableB, model, test, filep)
    
    # additional filter
    # slim_pred1 = filter_match_res_memory(match_tab=pres_df1, attr="title", K=1, search_strategy="exact")
    # slim_pred1 = pred1
    
    # Evaluate the predictions
    # _eval_results(slim_pred1, tableA, tableB, filep)
    
    # group
    group_interchangeable_fasttext(rep_attr, group_tau=0.4, external_group_strategy="graph", is_transitive_closure=False, 
                                   default_match_res_dir="output/exp")
    print("group done", flush=True)
    
    schemas = [rep_attr]
    
    # get the negative results
    default_fea_vec_dir = "output/exp"
    print(f"neg fea vec writing to ... {default_fea_vec_dir}", flush=True)
    run_feature_lib(is_interchangeable=0, flag_consistent=0, total_table=1, total_attr=len(schemas), 
                    attrs=schemas, usage="match", default_fea_vec_dir=default_fea_vec_dir, 
                    default_res_tab_name="neg_match_res", group_strategy="graph")
    
    # apply on negative match results
    # read (tn + fn) feature vectors
    H2 = pd.read_csv("output/exp/feature_vec0.csv")
    _label_cand(gold_graph, H2)
    H2.rename(columns={"id": "_id"}, inplace=True)
    _set_metadata(H2, "_id", "ltable_id", "rtable_id", tableA, tableB)
    
    # impute
    if impute_strategy == "mean":
        em.impute_table(H2, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="mean")
    elif impute_strategy == "constant":
        em.impute_table(H2, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)
        
    # apply
    pred2, _, pres_df2, _ = apply_model(tableA, tableB, model, H2, filep, True, pred1)
    
    # additional filter
    # slim_pred2 = filter_match_res_memory(match_tab=pres_df2, attr="title", K=1, search_strategy="exact")
    # slim_pred2 = pred2
    
    # Evaluate the predictions
    # _eval_results(slim_pred2, tableA, tableB, filep)