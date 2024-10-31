/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _BLOCKER_CONFIG_H_
#define _BLOCKER_CONFIG_H_

#include "common/dataframe.h"

// blocker common
// global variables
extern ui num_tables;
extern ui num_rules;
extern Table table_A; 
extern Table table_B; 
extern Table gold;
extern Rule* rules;
extern std::vector<std::vector<ui>> id_mapA;
extern std::vector<std::vector<ui>> id_mapB;
extern std::vector<std::vector<ui>> idStringMapA; 
extern std::vector<std::vector<ui>> idStringMapB;
extern std::vector<std::vector<std::vector<ui>>> recordsA;
extern std::vector<std::vector<std::vector<ui>>> recordsB;
extern std::vector<std::vector<double>> weightsA;
extern std::vector<std::vector<double>> weightsB;
extern std::vector<std::vector<double>> wordwt;
extern std::unordered_map<std::string, ui> datasets_map; // "tok" + "tok_setting" + "column"
extern std::vector<std::vector<int>> final_pairs;
extern std::vector<std::vector<std::pair<int, int>>> passedRules;


#endif // _BLOCKER_CONFIG_H_