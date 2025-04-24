/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
/*
 * utils for blocker
 */
#ifndef _BLOCKER_UTIL_H_
#define _BLOCKER_UTIL_H_

#include "common/config.h"
#include "common/dataframe.h"
#include "topk/topk.h"
#include "blocker/blocker_config.h"
#include <vector>
#include <assert.h>

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


class BlockerUtil
{
public:
    BlockerUtil() = default;
    ~BlockerUtil() = default;
    BlockerUtil(const BlockerUtil &other) = delete;
    BlockerUtil(BlockerUtil &&other) = delete;

    // merge the result pairs
	// the result pairs will have a format as adjacency list
	// this design is for post-processing
private:
	// merge pairs from a adjacency list
    static void mergePairs(const std::vector<std::vector<int>> &bucket);

public:
	// These two functions will merge result into an adjacency list
	// synthesize pairs according id_map in Self join
    static void synthesizePairsSelf(ui pos, std::vector<std::pair<int, int>> &pairs, int mapid);
	// synthesize pairs according id_map in RS join
    static void synthesizePairsRS(ui pos, std::vector<std::pair<int, int>> &pairs, int mapid);

	// pre-filter when res pair already exceed maximum value, default 1e9
    // we only employ TA since the all scores method needs sim weights
    // TA
	static void pretopKviaTASelf(uint64_t K, const std::string &topKattr, const std::string &attrType, bool isWeighted);
	static void pretopKviaTARS(uint64_t K, const std::string &topKattr, const std::string &attrType, bool isWeighted);
};


#endif // _BLOCKER_UTIL_H_