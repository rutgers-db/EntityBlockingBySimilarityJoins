/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "blocker/blocker_config.h"

// blocker common
// global variables
ui num_tables = 0;
ui num_rules = 0;
Table table_A; 
Table table_B; 
Table gold;
Rule* rules;
std::vector<std::vector<ui>> id_mapA;
std::vector<std::vector<ui>> id_mapB;
std::vector<std::vector<ui>> idStringMapA; 
std::vector<std::vector<ui>> idStringMapB;
std::vector<std::vector<std::vector<ui>>> recordsA;
std::vector<std::vector<std::vector<ui>>> recordsB;
std::vector<std::vector<double>> weightsA;
std::vector<std::vector<double>> weightsB;
std::vector<std::vector<double>> wordwt;
std::unordered_map<std::string, ui> datasets_map; // "tok" + "tok_setting" + "column"
std::vector<std::vector<int>> final_pairs;
std::vector<std::vector<std::pair<int, int>>> passedRules;