# author: Yunqi Li
# contact: liyunqixa@gmail.com
'''
this file can only be used in experiments
'''
import py_entitymatching as em
from typing import Literal
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
import py_entitymatching.utils.generic_helper as gh
import simjoin_entitymatching.utils.path_helper as ph
import simjoin_entitymatching.matcher.random_forest as randf
from simjoin_entitymatching.feature.feature import run_feature_lib, run_feature_megallen


def _get_recall(gold_graph, candidates, num_golds):
    cur_golds = 0
    row_index = list(candidates.index)

    for index in row_index:
        id1 = str(candidates.loc[index, 'ltable_id']) + 'A'
        id2 = str(candidates.loc[index, 'rtable_id']) + 'B'
        if gold_graph.has_edge(id1, id2) == True:
            cur_golds += 1

    recall = cur_golds / num_golds * 1.0
    density = cur_golds / len(candidates) * 1.0 if len(candidates) > 0 else 0.0
    f1 = 2 * ((recall * density) / (recall + density)) if recall + density > 0.0 else 0.0
    
    print("recall     : %.4f" % recall)
    print("precision  : %.4f" % density)
    print("F1 Score   : %.4f" % f1)
    print(cur_golds, num_golds, len(candidates))

    return recall, density, f1


def _label_cand(gold_graph, C):
    C.insert(C.shape[1], 'label', 0)
    row_index = list(C.index)

    for index in row_index:
        id1 = str(C.loc[index, 'ltable_id']) + 'A'
        id2 = str(C.loc[index, 'rtable_id']) + 'B'
        if gold_graph.has_edge(id1, id2) == True:
            C.loc[index, 'label'] = 1
            
    return C


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


def _save_neg_match_res(predictions, tableA, tableB):
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
    
    # save
    pres_df.to_csv("output/exp/match_res0.csv", index=False)
    neg_pres_df.to_csv("output/exp/neg_match_res0.csv", index=False)


def split_and_dump_data(tableA, tableB, gold_graph, impute_strategy=Literal["mean", "constant"], default_blk_res_dir=""):
    path_blk_res_stat = ph.get_blk_res_stat_path(default_blk_res_dir)
    with open(path_blk_res_stat, "r") as stat_file:
        stat_line = stat_file.readlines()
        total_table, _ = (int(val) for val in stat_line[0].split())

    for tab_id in range(total_table):
        # read H1
        path_fea_vec = ph.get_chunked_fea_vec_path(tab_id, default_blk_res_dir)
        H1 = em.read_csv_metadata(path_fea_vec, 
                                  key="id", 
                                  ltable=tableA, rtable=tableB,
                                  fk_ltable="ltable_id", fk_rtable="rtable_id")
        H1.rename(columns={"id": "_id"}, inplace=True)
        em.set_key(H1, "_id")
        
        # read H2
        path_fea_vec = path_fea_vec[ : -4] + "_py.csv"
        H2 = em.read_csv_metadata(path_fea_vec, 
                                  key="_id", 
                                  ltable=tableA, rtable=tableB,
                                  fk_ltable="ltable_id", fk_rtable="rtable_id")
        
        # label and impute    
        _label_cand(gold_graph, H1)
        _label_cand(gold_graph, H2)
        
        if impute_strategy == "mean":
            H1 = em.impute_table(H1, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="mean")
            H2 = em.impute_table(H2, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="mean")
        else:
            H1 = em.impute_table(H1, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)
            H2 = em.impute_table(H2, exclude_attrs=["_id", "ltable_id", "rtable_id", "label"], strategy="constant", fill_value=0.0)
            
        # conact
        tot_H1 = H1 if tab_id == 0 else pd.concat([tot_H1, H1], ignore_index=True)
        tot_H2 = H2 if tab_id == 0 else pd.concat([tot_H2, H2], ignore_index=True)
        
    tot_H1 = _set_metadata(tot_H1, key="_id", fk_ltable="ltable_id", fk_rtable="rtable_id", ltable=tableA, rtable=tableB)
    tot_H2 = _set_metadata(tot_H2, key="_id", fk_ltable="ltable_id", fk_rtable="rtable_id", ltable=tableA, rtable=tableB)
    
    # split
    # T: train, E: test
    # T : E = 1 : 1
    TE = em.split_train_test(tot_H1, train_proportion=0.5, random_state=0)
    T = TE["train"]
    E = TE["test"]
    T2 = tot_H2.loc[T.index.values]
    E2 = tot_H2.loc[E.index.values]
    T2.reset_index(drop=True, inplace=True)
    E2.reset_index(drop=True, inplace=True)
    
    # dump
    exp_dir = "output/exp"
    is_exist = os.path.exists(exp_dir)
    if not is_exist:
        os.mkdir(exp_dir)
        print(f"experiments directory is not presented, creat a new one")
    
    path_train = "/".join([exp_dir, "train1.csv"])
    path_test = "/".join([exp_dir, "test1.csv"])
    path_train2 = "/".join([exp_dir, "train2.csv"])
    path_test2 = "/".join([exp_dir, "test2.csv"])
    T.to_csv(path_train, index=False)
    E.to_csv(path_test, index=False)
    T2.to_csv(path_train2, index=False)
    E2.to_csv(path_test2, index=False)
    
    return T, E, T2, E2


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
                                cv=5, n_jobs=1, verbose=2)
    
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
    
    
def apply_model(tableA, tableB, exp_rf, E, is_conact=False, prev_pred=None):
    _set_metadata(E, "_id", "ltable_id", "rtable_id", tableA, tableB)
    # Predict on E
    predictions = exp_rf.predict(table=E, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'], 
                                 append=True, target_attr='predicted', inplace=False, 
                                 return_probs=True, probs_attr='proba')
    print(predictions.head())
    
    if is_conact:
        predictions = pd.concat([predictions[predictions['predicted'] == 1], prev_pred], ignore_index=True)
        _set_metadata(predictions, "_id", "ltable_id", "rtable_id", tableA, tableB)
    
    # Evaluate the predictions
    eval_result = em.eval_matches(predictions, 'label', 'predicted')
    print("------ report exp model results ------")
    em.print_eval_summary(eval_result)
    print("------ end ------")
    
    _save_neg_match_res(predictions, tableA, tableB)
    
    return predictions
    
    
def run_experiments(tableA, tableB, at_ltable, at_rtable, gold_graph, gold_len, impute_strategy=Literal["mean", "constant"]):
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
    
    schemas = list(tableA)[1:]
    schemas = [attr for attr in schemas if attr not in ["price", "year"]]
    run_feature_lib(is_interchangeable=1, flag_consistent=0, total_table=total_table, total_attr=len(schemas), 
                    attrs=schemas, usage="match")
    
    # split original data
    train, test, train2, test2 = split_and_dump_data(tableA, tableB, gold_graph, impute_strategy=impute_strategy)
    
    # train the model on half portion of the blk_res
    model = train_model(train)
    
    # apply on test result
    pred1 = apply_model(tableA, tableB, model, test)
    # _get_recall(gold_graph, pred1, gold_len)
    
    # get the negative results
    default_fea_vec_dir = "output/exp"
    print(f"neg fea vec writing to ... {default_fea_vec_dir}")
    run_feature_lib(is_interchangeable=1, flag_consistent=0, total_table=1, total_attr=len(schemas), 
                    attrs=schemas, usage="match", default_fea_vec_dir=default_fea_vec_dir, default_res_tab_name="neg_match_res")
    
    # apply on negative match results
    H2 = pd.read_csv("output/exp/feature_vec0.csv")
    _label_cand(gold_graph, H2)
    H2.rename(columns={"id": "_id"}, inplace=True)
    pred2 = apply_model(tableA, tableB, model, H2, True, pred1)
    # _get_recall(gold_graph, pred2, gold_len)
    
    # apply on the second-round test data
    pred3 = apply_model(tableA, tableB, model, test2)
    # _get_recall(gold_graph, pred3, gold_len)
    
    