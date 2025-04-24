/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _SIMJOIN_BLOCKER_H_
#define _SIMJOIN_BLOCKER_H_

#include "common/config.h"
#include "common/dataframe.h"
#include "common/io.h"
#include "common/setjoin.h"
#include "common/setjoin_parallel.h"
#include "common/ovlpjoin.h"
#include "common/ovlpjoin_parallel.h"
#include "common/stringjoin.h"
#include "common/stringjoin_parallel.h"
#include "blocker/blocker_util.h"
#include "topk/topk.h"
#include "group/group_old.h"
#include "blocker/blocker_config.h"
#include <bitset>
#include <fstream>
#include <parallel/algorithm>
#define NDEBUG
#include <assert.h>


class SimJoinBlocker
{
public:
    SimJoinBlocker() = default;
    ~SimJoinBlocker() = default;
    SimJoinBlocker(const SimJoinBlocker &other) = delete;
    SimJoinBlocker(SimJoinBlocker &&other) = delete;

public:
    static void selfSimilarityJoinParallel(uint64_t K, const std::string &topKattr, 
										   const std::string &attrType, bool ifWeighted);

    static void RSSimilarityJoinSerial(uint64_t K, const std::string &topKattr, 
									   const std::string &attrType, bool ifWeighted, 
									   bool isJoinTopK);
	// TODO:
	static void selfSimilarityJoinSerial(uint64_t K, const std::string &topKattr, 
										 const std::string &attrType, bool ifWeighted);

	static void RSSimilarityJoinParallel(uint64_t K, const std::string &topKattr, 
										 const std::string &attrType, bool ifWeighted);

public:
	static void estimateDensity(bool isWeighted, std::vector<double> &densities, 
								std::unordered_map<std::string, double> &attrAverage, 
								const std::string &defaultSampleResDir = "");
	// take account interchangeable values
	static void selfInterchangeableJoin(uint64_t K, const std::string &topKattr, 
										const std::string &attrType, bool ifWeighted);
};


#endif // _SIMJOIN_BLOCKER_H_