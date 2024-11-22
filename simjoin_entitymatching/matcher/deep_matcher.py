# author: Yunqi Li
# contact: liyunqixa@gmail.com
import deepmatcher as dm
import pandas as pd
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
        
        # dump
        path_dm_blk_res = ph.get_deep_matcher_input_path(default_blk_res_dir)
        blk_res.to_csv(path_dm_blk_res, index=False)
        
        
    def train_model(self, default_blk_res_dir=""):
        pass
    
    
    def store_model(self, path_model):
        pass