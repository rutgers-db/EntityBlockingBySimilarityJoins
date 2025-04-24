/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _TOP_K_H_
#define _TOP_K_H_

#include "common/config.h"
#include "common/dataframe.h"
#include "common/tokenizer.h"
#include "common/simfunc.h"
#include <fstream>
#include <iostream>
#include <bitset>
#include <parallel/algorithm>
#include <unordered_set>
#include <queue>
#include <iomanip>
#include <numeric>

// topk table entry
struct TableEntry
{
	uint64_t id{0};
	double val{0.0};

	TableEntry() = default;
	TableEntry(uint64_t _id, double _val)
	: id(_id), val(_val) { }
};

// top k bucket entry
struct PQComparison
{
	bool operator() (const TableEntry &lhs, const TableEntry &rhs) const{
		return lhs.val > rhs.val;
	}
};


class TopK
{
public:
    TopK() = default;
    ~TopK() = default;
    TopK(const TopK &other) = delete;
    TopK(TopK &&other) = delete;

private:
	// buffers
	static void allocateBuffers(uint64_t numEntity, ui numDimension, TableEntry **&valueTable, double **&backup);
	static void releaseBuffers(ui numDimension, TableEntry **&valueTable, double **&backup);

	// fill in table
    // using weighted similarity metrics or original one
    // it seems that the original is better, but the reason remains unclear
	static void prepareSelf(const std::vector<std::vector<ui>> &records, const std::vector<ui> &id_map, 
							const std::vector<std::vector<int>> &final_pairs, 
							std::vector<std::pair<int, int>> &id2Pair, 
							TableEntry **&valueTable, double **&backup,
							ui numRow, uint64_t &numEntity);

	static void prepareSelfWeighted(const std::vector<std::vector<ui>> &records, const std::vector<ui> &id_map, 
									const std::vector<double> &recWeights, const std::vector<double> &wordwt, 
									const std::vector<std::vector<int>> &final_pairs, 
									std::vector<std::pair<int, int>> &id2Pair, 	
									TableEntry **&valueTable, double **&backup,
									ui numRow, uint64_t &numEntity);

    // for RS join, the pair must be in form: (idA, idB) as in ground truth
    // that is, we need to synethsis before doing top k
	static void prepareRS(const std::vector<std::vector<ui>> &recordsA, const std::vector<std::vector<ui>> &recordsB, 
						  const std::vector<ui> &id_mapA, const std::vector<ui> &id_mapB, 
						  const std::vector<std::vector<int>> &final_pairs, 
						  std::vector<std::pair<int, int>> &id2Pair, 
						  TableEntry **&valueTable, double **&backup,
						  ui numRowA, ui numRowB, uint64_t &numEntity);

    static void prepareRSWeighted(const std::vector<std::vector<ui>> &recordsA, const std::vector<std::vector<ui>> &recordsB, 
                                  const std::vector<ui> &id_mapA, const std::vector<ui> &id_mapB, 
                                  const std::vector<double> &recWeightsA, const std::vector<double> &recWeightsB,
                                  const std::vector<double> &wordwt, 
                                  const std::vector<std::vector<int>> &final_pairs, 
                                  std::vector<std::pair<int, int>> &id2Pair, 
                                  TableEntry **&valueTable, double **&backup, 
                                  ui numRowA, ui numRowB, uint64_t &numEntity);

	// intuitively, the topK attr will be the attr that different entities 
	// will have different values on it
	// thus, the attr tends to be long, with as much as info as possible.
	// so we use dlm_dc0 to tokenize it, and it must be already tokenzied before

	// calculate recall within the functions
	// mode: "report" only reports without change final_pairs; "replace" only changes without reports, 
	//       "hybrid" for both, "exp" is only used for experiments
private:
	static void updateFinalPairs(const std::vector<ui> &topKidMapA, const std::vector<ui> &topKidMapB, const std::vector<std::pair<int, int>> &id2Pair,
								 std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> &q,
								 std::vector<std::vector<int>> &final_pairs, uint64_t K);

	static void getCurrentRecall(const std::vector<ui> &topKidMapA, const std::vector<ui> &topKidMapB, const std::vector<std::pair<int, int>> &id2Pair,
								 std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> &q,
								 const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian);
	// only used for "exp" mode
	static void getCurrentRecallExp(const std::vector<ui> &topKidMapA, const std::vector<ui> &topKidMapB, const std::vector<std::pair<int, int>> &id2Pair,
									std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> &q,
									const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian, 
									std::ofstream &statStream);

public:
    // TA
    static void topKviaTASelf(const Table &table_A, const std::string &topKattr, const std::string &attrType,
							  const std::vector<std::vector<std::vector<ui>>> &recordsA,
							  const std::vector<std::vector<double>> &recWeights, 
							  const std::vector<std::vector<double>> &wordwt,
							  const std::unordered_map<std::string, ui> &datasets_map, 
							  const std::vector<std::vector<ui>> &id_mapA, 
							  std::vector<std::vector<int>> &final_pairs,
							  const Table &groundTruth, uint64_t K, std::ofstream &statStream,
							  bool ifWeighted = false, const std::string &mode = "exp");

	static void topKviaTARS(const Table &table_A, const Table &table_B, const std::string &topKattr, const std::string &attrType,
							const std::vector<std::vector<std::vector<ui>>> &recordsA,
							const std::vector<std::vector<std::vector<ui>>> &recordsB,
							const std::vector<std::vector<double>> &recWeightsA, 
							const std::vector<std::vector<double>> &recWeightsB, 
							const std::vector<std::vector<double>> &wordwt,
							const std::unordered_map<std::string, ui> &datasets_map, 
							const std::vector<std::vector<ui>> &id_mapA,
							const std::vector<std::vector<ui>> &id_mapB, 
							std::vector<std::vector<int>> &final_pairs,
							const Table &groundTruth, uint64_t K, std::ofstream &statStream,
							bool ifWeighted = false, const std::string &mode = "exp");
							
	// optimized version
	// multithread & SIMD
    // to be done
	static void topKviaTASelfOpt(const Table &table_A, const std::string &topKattr, const std::string &attrType,
							 	 const std::vector<std::vector<std::vector<ui>>> &recordsA, 
							 	 const std::unordered_map<std::string, ui> &datasets_map, 
							 	 const std::vector<std::vector<ui>> &id_mapA, 
							 	 std::vector<std::vector<int>> &final_pairs,
							 	 uint64_t K);

    // NRA: Non-random Access Algorithm
    // this algorithm is the primary choice of the blocker
    // but it's not efficient, so decrapated now

	// weighted all similarity scores sort
private:
	static void updateFinalPairs(const std::vector<ui> &sortedIdMap, const std::vector<std::pair<int, int>> &allPairs,
								 std::vector<std::vector<int>> &final_pairs, uint64_t K);

	static void getCurrentRecall(const std::vector<ui> &sortedIdMap, const std::vector<std::pair<int, int>> &allPairs,
								 const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian);
	// only used for "exp" mode
	static void getCurrentRecallExp(const std::vector<ui> &sortedIdMap, const std::vector<std::pair<int, int>> &allPairs,
									const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian, 
									std::ofstream &statStream);

public:
	static void topKviaAllSimilarityScoresRS(const Table &table_A, const Table &table_B, const Rule *rules,
											 const std::vector<double> &simWeights,
											 const std::unordered_map<std::string, double> &attrAverage,
											 const std::vector<std::vector<std::vector<ui>>> &recordsA,
											 const std::vector<std::vector<std::vector<ui>>> &recordsB,
											 const std::vector<std::vector<double>> &recWeightsA, 
											 const std::vector<std::vector<double>> &recWeightsB, 
											 const std::vector<std::vector<double>> &wordwt,
											 const std::unordered_map<std::string, ui> &datasets_map, 
											 const std::vector<std::vector<ui>> &id_mapA,
											 const std::vector<std::vector<ui>> &id_mapB, 
											 const std::vector<std::vector<ui>> &id_mapAString,
											 const std::vector<std::vector<ui>> &id_mapBString,
											 std::vector<std::vector<int>> &final_pairs,
											 const Table &groundTruth, uint64_t K, ui numRule, 
											 std::ofstream &statStream, bool isWeighted = false, 
											 const std::string &mode = "exp");

	static void topKviaAllSimilarityScoreSelf(const Table &table_A, const Rule *rules, const std::vector<double> &simWeights,
											  const std::unordered_map<std::string, double> &attrAverage, 
											  const std::vector<std::vector<std::vector<ui>>> &records, 
											  const std::vector<std::vector<double>> &recWeights, 
											  const std::vector<std::vector<double>> &wordwt, 
											  const std::unordered_map<std::string, ui> &datasets_map, 
											  const std::vector<std::vector<ui>> &id_map, 
											  const std::vector<std::vector<ui>> &id_mapString, 
											  std::vector<std::vector<int>> &final_pairs, 
											  const Table &groundTruth, u_int64_t K, ui numRule, 
											  std::ofstream &statStream, bool isWeighted = false, 
											  const std::string &mode = "exp");

public:
	// on match result
	void topKviaTATable(const Table &matchRes, uint64_t K, std::vector<std::vector<std::string>> &newTable);
};


#endif // _TOP_K_H_