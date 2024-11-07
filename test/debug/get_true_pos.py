import pandas as pd


gold = pd.read_csv("output/buffer/gold.csv")
ground_truth = set()

for _, row in gold.iterrows():
    lid = int(row["id1"])
    rid = int(row["id2"])
    ground_truth.add((lid, rid))
    
match_res = pd.read_csv("output/match_res/match_res.csv")
true_positive = pd.DataFrame(columns=match_res.columns)
false_negative = pd.DataFrame(columns=["_id", "ltable_id", "rtable_id", "ltable_title", "rtable_title"])

for _, row in match_res.iterrows():
    lid = int(row["ltable_id"])
    rid = int(row["rtable_id"])
    if (lid, rid) in ground_truth:
        true_positive.loc[len(true_positive)] = list(row)
        ground_truth.remove((lid, rid))
        
tableA = pd.read_csv("output/buffer/clean_A.csv")
tableB = pd.read_csv("output/buffer/clean_B.csv")
map_A = {tableA.loc[ridx, "id"] : ridx for ridx in list(tableA.index)}
map_B = {tableB.loc[ridx, "id"] : ridx for ridx in list(tableB.index)}

for idx, tup in enumerate(ground_truth):
    lridx = map_A[tup[0]]
    rridx = map_B[tup[1]]
    addlist = [idx, tup[0], tup[1], tableA.loc[lridx, "title"], tableB.loc[rridx, "title"]]
    false_negative.loc[len(false_negative)] = addlist
        
true_positive.to_csv("test/debug/true_pos.csv", index=False)
false_negative.to_csv("test/debug/false_neg.csv", index=False)