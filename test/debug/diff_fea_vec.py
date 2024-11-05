import pandas as pd


path_block_stat = "output/blk_res/stat.txt"
with open(path_block_stat, "r") as stat_file:
    stat_line = stat_file.readlines()
    total_table, _ = (int(val) for val in stat_line[0].split())
    
tmp_df = pd.read_csv("output/blk_res/feature_vec0.csv")
schemas = list(tmp_df.columns)
diff_ori_fea_vec = pd.DataFrame(columns=schemas)
diff_pro_fea_vec = pd.DataFrame(columns=schemas)
    
for i in range(total_table):
    print(f"processing feature vector {i} ...")
    path_ori_fea_vec = "output/blk_res/feature_vec" + str(i) + "_py.csv"
    path_pro_fea_vec = "output/blk_res/feature_vec" + str(i) + ".csv"
    
    ori_fea_vec = pd.read_csv(path_ori_fea_vec)
    pro_fea_vec = pd.read_csv(path_pro_fea_vec)
    
    ori_fea_vec.rename(columns={"_id": "id"}, inplace=True)
    
    ori_row_index = list(ori_fea_vec.index)
    pro_row_inedx = list(pro_fea_vec.index)
    length = len(ori_row_index)
    
    for j in range(length):
        ori_ridx = ori_row_index[j]
        pro_ridx = pro_row_inedx[j]
        for attr in schemas:
            if ori_fea_vec.loc[ori_ridx, attr] != pro_fea_vec.loc[pro_ridx, attr] or \
               pd.isna(ori_fea_vec.loc[ori_ridx, attr]) ^ pd.isna(pro_fea_vec.loc[pro_ridx, attr]) == True:
            #    diff_ori_fea_vec[-1] = list(ori_fea_vec.loc[ori_ridx])
            #    diff_pro_fea_vec[-1] = list(pro_fea_vec.loc[pro_ridx])
            #    diff_ori_fea_vec.index = diff_ori_fea_vec.index + 1
            #    diff_pro_fea_vec.index = diff_pro_fea_vec.index + 1
            #    diff_ori_fea_vec.sort_index(inplace=True)
            #    diff_pro_fea_vec.sort_index(inplace=True)
                   
                diff_ori_fea_vec = pd.concat([pd.DataFrame([list(ori_fea_vec.loc[ori_ridx])], columns=diff_ori_fea_vec.columns), diff_ori_fea_vec], ignore_index=True)
                diff_pro_fea_vec = pd.concat([pd.DataFrame([list(pro_fea_vec.loc[pro_ridx])], columns=diff_pro_fea_vec.columns), diff_pro_fea_vec], ignore_index=True)

diff_ori_fea_vec.to_csv("test/debug/diff_ori_fea_vec.csv", index=False)
diff_pro_fea_vec.to_csv("test/debug/diff_pro_fea_vec.csv", index=False)