# General
0. add patch on modifcation on py_entitymatching 
1. KNN blocker, part of simjoin blocker
2. interchangeable value for blocker
3. feature part implements some redundant parts (e.g., sim funcs), could be eliminated in the future
4. Modify the SimFuncs return value for empty sets from 1 to NaN
5. group.cc should use feature_index
6. usage of length filter in calculation features, but what if no pairs pass length filter?
7. make the similarity join apis public
8. word2vec & glove value matcher

# Optimization
1. serial string join optimization: using sharing prefix
2. overlap join (except parallel self): using small/large case
3. vectorized TopK algorithm
4. optimization on serial set join index memory allocation
5. iterative verification for all 4 string joins
6. sampler could support both dlm & qgm
7. Add another two sim joins implementation in "simjoin.hpp"
8. re-arrange the files in "blocker" folder, should we keep extern global values in "simjoin.hpp"?
9. interchangeable values in blocking
10. Add namespace

Please refer to ```developer_notes.md``` for remaining issues.