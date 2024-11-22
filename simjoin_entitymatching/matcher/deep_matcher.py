# author: Yunqi Li
# contact: liyunqixa@gmail.com
import deepmatcher as dm
import py_entitymatching as em
import pandas as pd
import torch
from typing import Literal
import simjoin_entitymatching.utils.path_helper as ph


class DeepMatcher:
    '''
    Deeping learning matcher
    '''
    
    def __init__(self):
        pass
    
    
    def label_table(self, gold, table):
        ground_truth = set()
        for _, row in gold.iterrows():
            lid = row["id1"]
            rid = row["id2"]
            ground_truth.add((lid, rid))
            
        new_col = []
        for _, row in table.iterrows():
            lid = row["ltable_id"]
            rid = row["rtable_id"]
            new_col.append(int((lid, rid) in ground_truth))
        
        # right after "id"
        new_tab = table.insert(loc=1, column="label", value=new_col)
        return new_tab
    
    
    def fix_table(self, gold, table):
        '''
        the "py_entitymatching" and "deepmatcher" use different tables format
        '''
        # id
        formated_table = table.rename({"_id": "id"}, inplace=False) if "_id" in table.columns else table
        
        # label
        formated_table = self.label_table(gold, formated_table)
        
        # fix the left schemas
        formated_table = formated_table.drop("ltable_id", axis=1)
        formated_table = formated_table.drop("rtable_id", axis=1)
        
        schemas = list(formated_table)[2:]
        for sch in schemas:
            split_sch = sch.split("_")
            which_tab = split_sch[0]
            which_atr = split_sch[1]
            if which_tab == "ltable":
                new_sch = "_".join(["left", which_atr])
            elif which_tab == "rtable":
                new_sch = "_".join(["right", which_atr])
            else:
                raise ValueError(f"error in schema : {which_tab}, {which_atr}")
            formated_table = formated_table.rename({sch: new_sch}, inplace=False)
            
        return formated_table
    
    
    def process_data(self, total_table, gold, default_blk_res_dir=""):
        for i in range(total_table):
            # read the chunked table
            path_blk_res = ph.get_chunked_blk_res_path(i, default_blk_res_dir)
            c_blk_res = pd.read_csv(path_blk_res)
            
            # reformat
            c_blk_res = self.fix_table(gold, c_blk_res)
            
            # concat
            blk_res = c_blk_res if i == 0 else pd.concat([blk_res, c_blk_res], ignore_index=True)
            
        # split train : validation : test = 0.3334 : 0.1667 : 0.5
        IJ = em.split_train_test(blk_res, train_proportion=0.5, random_state=0)
        train = IJ["train"]
        test = IJ["test"]
        GH = em.split_train_test(train, train_proportion=0.6667, random_state=0)
        train = GH["train"]
        validation = GH["test"]
        
        # dump
        path_dm_blk_res, _, path_dm_train, \
        path_dm_validation, path_dm_test = ph.get_deep_matcher_input_path(default_blk_res_dir)
        blk_res.to_csv(path_dm_blk_res, index=False)
        
        # three subtables
        train.to_csv(path_dm_train, index=False)
        validation.to_csv(path_dm_validation, index=False)
        test.to_csv(path_dm_test, index=False)
        
        
    def train_model(self, path_model, mode=Literal["sif", "rnn", "attention", "hybrid"], default_blk_res_dir=""):
        _, dir_dm_input, _, _, _ = ph.get_deep_matcher_input_path(default_blk_res_dir)
        
        # load
        train, validation, test = dm.data.process(
            path=dir_dm_input,
            train='train.csv',
            validation='validation.csv',
            test='test.csv')
        print(train.head())
        
        # model
        model = dm.MatchingModel(attr_summarizer=mode)
        
        # check model path
        path_model_split = path_model.split('.')
        path_format = path_model_split[-1]
        if path_format != "pth":
            raise NameError(f"model path is incorrect : {path_model}")
        
        # train
        model.run_train(
            train,
            validation,
            epochs=10,
            batch_size=16,
            best_save_path=path_model,
            pos_neg_ratio=3)

        return model
    
    
    def apply_model(self, path_model, mode=Literal["sif", "rnn", "attention", "hybrid"], default_blk_res_dir=""):
        # check model path
        path_model_split = path_model.split('.')
        path_format = path_model_split[-1]
        if path_format != "pth":
            raise NameError(f"model path is incorrect : {path_model}")
        
        model = dm.MatchingModel(attr_summarizer=mode)
        model.load_state_dict(torch.load(path_model))
        
        # apply the model to test
        # load
        _, dir_dm_input, _, _, _ = ph.get_deep_matcher_input_path(default_blk_res_dir)
        _, _, test = dm.data.process(
            path=dir_dm_input,
            train='train.csv',
            validation='validation.csv',
            test='test.csv')
        print(test.head())
        model.run_eval(test)