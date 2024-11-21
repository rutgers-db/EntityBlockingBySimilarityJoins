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
    
    
    def fix_table(self, table):
        '''
        the "py_entitymatching" and "deepmatcher" use different tables format
        '''
    
    
    def process_data(self, total_table, default_blk_res_dir="", default_match_res_dir=""):
        for i in range(total_table):
            # read the chunked table
            path_blk_res = ph.get_chunked_blk_res_path(i, default_blk_res_dir)
            c_blk_res = pd.read_csv(path_blk_res)
            # concat
            blk_res = c_blk_res if i == 0 else pd.concat([blk_res, c_blk_res], ignore_index=True)