/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
/*
 * configs for blocker
 * this file can only be included by files inside "blocker" folder
 */
#ifndef _BLOCK_H_
#define _BLOCK_H_

#include "blocker/blocker_config.h"
#include "blocker/knn_blocker.h"
#include "blocker/simjoin_blocker.h"
#include <iomanip>
#include <omp.h>

// only in block.cc
// global variables
extern ui num_word;
extern std::vector<TokenizerType> tok_type;
extern std::vector<ui> q; 


class Block
{
public:
	Block() = default;
	~Block() = default;
	Block(const Block &other) = delete;
	Block(Block &&other) = delete;

public:
	static void clearBuffers();

public:
	static void prepareRecordsRS(ui columnA, ui columnB, TokenizerType tt, ui q);
	static void prepareRecordsSelf(ui columnA, TokenizerType tt, ui q);
	static void sortColumns();
	static void showPara(int jt, int js, uint64_t topK, const std::string &topKattr, 
						 const std::string &attrType, const std::string &pathTableA, 
						 const std::string &pathTableB, const std::string &pathGold, 
						 const std::string &pathRule, int tableSize);

	// pre-process & post-process
	static void readCSVTables(int isRS, const std::string &pathTableA, const std::string &pathTableB, 
					  		  const std::string &pathGold);
	static void readRules(const std::string &pathRule);
	static void tokenize(int isRS);
	static void getRecall(int isRS);
	static void getRecall4Rules(int isRS);
};


#endif // _BLOCK_CONFIG_H_