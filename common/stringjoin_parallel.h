/*
 * author: Dong Deng 
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _STRING_JOIN_PARALLEL_H_
#define _STRING_JOIN_PARALLEL_H_

#include "config.h"
#include "joinutil.h"
#include "index.h"
#include "simfunc.h"
#ifdef EDLIB_INSTALLED
#include "edlib.h"
#endif
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <immintrin.h>
#include <assert.h>


class StringJoinParallel
{
public:
    using InvListsParallel = std::unordered_map<uint64_t, std::vector<int>>;
    using InvListPrefix = std::unordered_map<uint64_t, std::vector<std::pair<int, int>>>;

#if TIMER_ON == 1
public:
	double indexProbingCost[MAXTHREADNUM]{0};
	double verifyingCost[MAXTHREADNUM]{0};
	uint64_t globalTimerCount{0};
#endif

public:
	int sharePrefix{0}; // approximate
	std::vector<uint64_t> workPrefixHash;
	std::vector<uint64_t> queryPrefixHash;
	int earlyTerminated[MAXTHREADNUM] = { 0 };

public:
	int workMinDictLen{19260817};
	int workMaxDictLen{0};
	int queryMinDictLen{19260817};
	int queryMaxDictLen{0};
	int maxDictLen{0};
	int minDictLen{0};
	uint64_t avgDictLen{0};
	int workN{0};
	int queryN{0};
	int N{0};
	int D{0}; 							             // threshold
	int PN{0};                                       // part num
	int hashNumber{31};                              // prime number for hashing
	int modNumber{1000000007};                       // prime mod number for hashing
	std::vector<std::string> work_dataset;		     // work dataset
	std::vector<std::string> query_dataset;          // query dataset, null if self-join
	std::vector<std::pair<int, int>> pairs[MAXTHREADNUM];       		  

	uint64_t candNum{0};  				             // number of candidate pairs
	uint64_t veriNum{0};  				             // number of pairs passed left verification
	uint64_t listNum{0};  				             // number of inverted list matched
	uint64_t realNum{0};  				             // number of results
	bool valid[MAXTHREADNUM];                        // used for early termination
	int left[MAXTHREADNUM];
	int right[MAXTHREADNUM];       		             // verify range of left part: [D - left, D + right]
	int _left[MAXTHREADNUM];
	int _right[MAXTHREADNUM];     		             // verify range of right part: [D - _left, D + _right]

	int **matrix[MAXTHREADNUM];                      // matrix used to verify left parts
	int **_matrix[MAXTHREADNUM];                     // matrix used to verify right parts
	bool *quickRef[MAXTHREADNUM];                    // record results to avoid duplicate verification
	int **partLen{nullptr};        		             // record length of all partitions
	int **partPos{nullptr};        		             // record start position of all partitions
	int *dist{nullptr};            		             // id of string with different length to its former one
	std::vector<int> workLengthArray;                // lengths of work records
	std::vector<int> queryLengthArray;
	std::vector<std::vector<int>> worklengthMap;     // inverted index -> lenght: {ids}
	std::vector<std::vector<int>> querylengthMap;
	uint64_t *power{nullptr};      		             // the power of hash number, used to hash segment and substring
	InvListsParallel **invLists{nullptr};            // inverted index of different string length and different segments
	InvListPrefix **invListsPre{nullptr};
	std::vector<PIndex> **partIndex{nullptr};        // the index record with pair of substring and segment should be checked
	hashValue *hv[MAXTHREADNUM];
	// added index
	std::unordered_set<std::string> strCount;        // count total different strings
	int *workInvSC{nullptr};                         // inverted index for strCount
	int *queryInvSC{nullptr};

	ui maxHeapSize{0};
#if MAINTAIN_VALUE_EDIT == 1
	std::vector<WeightPairEdit> result_pairs_[MAXTHREADNUM];
	int isHeap[MAXTHREADNUM] = { 0 };
#endif	

public:
	StringJoinParallel() = default;

	StringJoinParallel(const std::vector<std::string>& data, int threshold, ui _maxHeapSize = 0)
		: workN(data.size()), D(threshold), PN(threshold + 1), work_dataset(data) {
#if DROP_EMPTY == 1
		printf("Drop empty strings for string(lev) join\n");
		auto wit = work_dataset.begin();
		for( ; wit != work_dataset.end(); ) {
			if((*wit).empty())
				wit = work_dataset.erase(wit);
			else
				++ wit;
		}
		workN = work_dataset.size();
#endif
		for(const auto &wit : work_dataset)
			if(strCount.find(wit) == strCount.end())
				strCount.insert(wit);
		for(int i = 0; i < workN; i++) {
			workMaxDictLen = (int)work_dataset[i].size() > workMaxDictLen ? (int)work_dataset[i].size() 
							 : workMaxDictLen;
			workMinDictLen = (int)work_dataset[i].size() < workMinDictLen ? (int)work_dataset[i].size()
							 : workMinDictLen;
			avgDictLen += work_dataset[i].size();
		}
		std::sort(work_dataset.begin(), work_dataset.end(), StringJoinUtil::strLessT);

		minDictLen = workMinDictLen;
		maxDictLen = workMaxDictLen;
		N = workN;
		avgDictLen /= workN;

		printf("Work size: %zu\tQuery size: %zu\tAvg length: %lu\n", work_dataset.size(), query_dataset.size(), avgDictLen);
		printf("Min work record's size: %u\tMax work record's size: %u\n", workMinDictLen, workMaxDictLen);
		printf("Number of tokens: %ld\n", strCount.size());

#if APPROXIMATE == 1
		sharePrefix = avgDictLen / 10 + 1;
		if(sharePrefix == 0) ++ sharePrefix;
		printf("Trigger approximate join on sharing prefix: %d\n", sharePrefix);
#endif

	maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;
#if MAINTAIN_VALUE_EDIT == 1
	for(int tid = 0; tid < MAXTHREADNUM; tid++)
		result_pairs_[tid].reserve(maxHeapSize);
#endif
	}

	StringJoinParallel(const std::vector<std::string>& work, const std::vector<std::string>& query, int threshold, 
					   ui _maxHeapSize = 0)
		: workN(work.size()), queryN(query.size()), D(threshold), PN(threshold + 1), 
		work_dataset(work), query_dataset(query) {
#if DROP_EMPTY == 1
		printf("Drop empty strings for string(lev) join\n");
		// work
		auto wit = work_dataset.begin();
		for( ; wit != work_dataset.end(); ) {
			if((*wit).empty())
				wit = work_dataset.erase(wit);
			else
				++ wit;
		}
		workN = work_dataset.size();
		// query
		auto qit = query_dataset.begin();
		for( ; qit != query_dataset.end(); ) {
			if((*qit).empty())
				qit = query_dataset.erase(qit);
			else
				++ qit;
		}
		queryN = query_dataset.size();
#endif
		for(const auto &wit : work_dataset)
			if(strCount.find(wit) == strCount.end())
				strCount.insert(wit);
		for(const auto &qit : query_dataset)
			if(strCount.find(qit) == strCount.end())
				strCount.insert(qit);
		for(int i = 0; i < workN; i++) {
			workMaxDictLen = (int)work_dataset[i].size() > workMaxDictLen ? (int)work_dataset[i].size() 
							 : workMaxDictLen;
			workMinDictLen = (int)work_dataset[i].size() < workMinDictLen ? (int)work_dataset[i].size()
							 : workMinDictLen;
			avgDictLen += work_dataset[i].size();
		}
		for(int i = 0; i < queryN; i++) {
			queryMaxDictLen = (int)query_dataset[i].size() > queryMaxDictLen ? (int)query_dataset[i].size() 
							 : queryMaxDictLen;
			queryMinDictLen = (int)query_dataset[i].size() < queryMinDictLen ? (int)query_dataset[i].size()
							 : queryMinDictLen;
			avgDictLen += query_dataset[i].size();
		}
		std::sort(work_dataset.begin(), work_dataset.end(), StringJoinUtil::strLessT);
		std::sort(query_dataset.begin(), query_dataset.end(), StringJoinUtil::strLessT);

		minDictLen = std::min(workMinDictLen, queryMinDictLen);
		maxDictLen = std::max(workMaxDictLen, queryMaxDictLen);
		N = std::max(workN, queryN);
		avgDictLen /= (workN + queryN);

		printf("Work size: %zu\tQuery size: %zu\tAvg length: %lu\n", work_dataset.size(), query_dataset.size(), avgDictLen);
		printf("Min work record's size: %u\tMax work record's size: %u\n", workMinDictLen, workMaxDictLen);
		printf("Min query record's size: %u\tMax query record's size: %u\n", queryMinDictLen, queryMaxDictLen);
		printf("Number of tokens: %ld\n", strCount.size());

#if APPROXIMATE == 1
			sharePrefix = avgDictLen - 2 * D;
			if(sharePrefix == 0) ++ sharePrefix;
			printf("Trigger approximate join on sharing prefix: %d\n", sharePrefix);
#endif

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;
#if MAINTAIN_VALUE_EDIT == 1
		for(int tid = 0; tid < MAXTHREADNUM; tid++)
			result_pairs_[tid].reserve(maxHeapSize);
#endif
	}

	~StringJoinParallel() {
		for (int lp = 0; lp < PN; lp++) {
			delete[] partPos[lp];
			delete[] partLen[lp];
			delete[] invLists[lp];
			delete[] partIndex[lp];
		}
		delete[] partPos[PN];

		for(ui i = 0; i < MAXTHREADNUM; i++) {
			for (int lp = maxDictLen; lp >= 0; lp--) {
				delete[] matrix[i][lp];
				delete[] _matrix[i][lp];
			}
			delete[] matrix[i];
			delete[] _matrix[i];
			delete[] quickRef[i];
			delete[] hv[i];
		}

		delete[] dist;
		delete[] power;
		delete[] partPos;
		delete[] partLen;
		delete[] invLists;
		delete[] partIndex;
		delete[] workInvSC;
		delete[] queryInvSC;
		
		std::cout << "destructor" << std::endl << std::flush;
	}

	StringJoinParallel(const StringJoinParallel& other) = delete;
	StringJoinParallel(StringJoinParallel&& other) = delete;

public:
	// pre-process
	void init();
	void prepareSelf();
	void prepareRS();

	// verify
	bool verifyLeftPartSelf(int xid, int yid, int xlen, int ylen, int Tau, int tid, int sharing=0) {
		const auto &workString = work_dataset[xid];
		const auto &queryString = work_dataset[yid];
		auto &currM = matrix[tid];

		for (int i = sharing + 1; i <= xlen; i++) {
			valid[tid] = 0;

			if (i <= left[tid]) {
				currM[i][D - i] = i;
				valid[tid] = 1;
			}
			
			int val1 = i - left[tid];
			int val2 = i + right[tid];
			int lowerBound = std::max(val1, 1);
			int upperBound = std::min(val2, ylen);

			for (int j = lowerBound; j <= upperBound; j++) {
				int val = j - i + D;

				if (workString[i - 1] == queryString[j - 1])
					currM[i][val] = currM[i - 1][val];
				else 
					currM[i][val] = StringJoinUtil::min(currM[i - 1][val],
						j - 1 >= val1 ? currM[i][val - 1] : D,
						j + 1 <= val2 ? currM[i - 1][val + 1] : D) + 1;

				if (abs(xlen - ylen - i + j) + currM[i][val] <= Tau) 
					valid[tid] = 1;
			}
			
			if (!valid[tid]) 
				return false;
		}

		return currM[xlen][ylen - xlen + D] <= Tau;
	}

	bool verifyRightPartSelf(int xid, int yid, int xlen, int ylen, int xpos, int ypos, int Tau, int tid, int sharing=0) {
		const auto &workString = work_dataset[xid];
		const auto &queryString = work_dataset[yid];
		auto &currM = _matrix[tid];

		for (int i = sharing + 1; i <= xlen; i++) {
			valid[tid] = 0;

			if (i <= _left[tid]) {
				currM[i][D - i] = i;
				valid[tid] = 1;
			}

			int val1 = i - _left[tid];
			int val2 = i + _right[tid];
			int lowerBound = std::max(val1, 1);
			int upperBound = std::min(val2, ylen);

			for (int j = lowerBound; j <= upperBound; j++) {
				int val = j - i + D;

				if (workString[xpos + i - 1] == queryString[ypos + j - 1])
					currM[i][val] =  currM[i - 1][val];
				else 
                    currM[i][val] = StringJoinUtil::min(currM[i - 1][val],
						j - 1 >= val1 ? currM[i][val - 1] : D,
						j + 1 <= val2 ? currM[i - 1][val + 1] : D) + 1;

				if (abs(xlen - ylen - i + j) + currM[i][val] <= Tau) 
					valid[tid] = 1;
			}
			
			if (!valid[tid]) 
				return false;
		}

		return currM[xlen][ylen - xlen + D] <= Tau;
	}

	bool verifyLeftPartRS(int xid, int yid, int xlen, int ylen, int Tau, int tid, int sharing=0) {
#if TIMER_ON == 1
		timeval begin, end;
		gettimeofday(&begin, NULL);
#endif 
		const auto &workString = work_dataset[xid];
		const auto &queryString = query_dataset[yid];
		auto &currM = matrix[tid];

		for (int i = sharing + 1; i <= xlen; i++) {
			valid[tid] = 0;

			if (i <= left[tid]) {
				currM[i][D - i] = i;
				valid[tid] = 1;
			}
			
			int val1 = i - left[tid];
			int val2 = i + right[tid];
			int lowerBound = std::max(val1, 1);
			int upperBound = std::min(val2, ylen);

			for (int j = lowerBound; j <= upperBound; j++) {
				int val = j - i + D;

				if (workString[i - 1] == queryString[j - 1])
					currM[i][val] = currM[i - 1][val];
				else
					currM[i][val] = StringJoinUtil::min(currM[i - 1][val],
						j - 1 >= val1 ? currM[i][val - 1] : D,
						j + 1 <= val2 ? currM[i - 1][val + 1] : D) + 1;

				if (abs(xlen - ylen - i + j) + currM[i][val] <= Tau) 
					valid[tid] = 1;
			}
			
			if (!valid[tid]) 
				return false;
		}

#if TIMER_ON == 1
		gettimeofday(&end, NULL);
		double elapsedTime = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
		verifyingCost[tid] += elapsedTime;
#endif

		return currM[xlen][ylen - xlen + D] <= Tau;
	}

	bool verifyRightPartRS(int xid, int yid, int xlen, int ylen, int xpos, int ypos, int Tau, int tid, int sharing=0) {
#if TIMER_ON == 1
		timeval begin, end;
		gettimeofday(&begin, NULL);
#endif 
		const auto &workString = work_dataset[xid];
		const auto &queryString = query_dataset[yid];
		auto &currM = _matrix[tid];

		for (int i = sharing + 1; i <= xlen; i++) {
			valid[tid] = 0;

			if (i <= _left[tid]) {
				currM[i][D - i] = i;
				valid[tid] = 1;
			}

			int val1 = i - _left[tid];
			int val2 = i + _right[tid];
			int lowerBound = std::max(val1, 1);
			int upperBound = std::min(val2, ylen);

			for (int j = lowerBound; j <= upperBound; j++) {
				int val = j - i + D;

				if (workString[xpos + i - 1] == queryString[ypos + j - 1])
					currM[i][val] = currM[i - 1][val];
				else
                    currM[i][val] = StringJoinUtil::min(currM[i - 1][val],
						j - 1 >= val1 ? currM[i][val - 1] : D,
						j + 1 <= val2 ? currM[i - 1][val + 1] : D) + 1;

				if (abs(xlen - ylen - i + j) + currM[i][val] <= Tau) 
					valid[tid] = 1;
			}
			
			if (!valid[tid]) 
				return false;
		}

#if TIMER_ON == 1
		gettimeofday(&end, NULL);
		double elapsedTime = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
		verifyingCost[tid] += elapsedTime;
#endif

		return currM[xlen][ylen - xlen + D] <= Tau;
	}

	// proposed in extended version in TODS'13
	bool iterativeVerifyLeftPartRS(int xid, int yid, int stPos, int xlen, int ylen, int wlen, int taul, int partId) {
		int xidx = xlen - 1;
		int yidx = ylen - 1;
		while(work_dataset[xid][xidx] == query_dataset[yid][yidx] && yidx >= 0 && xidx >= stPos) {
			-- xidx;
			-- yidx;
		}

		// the second from last segment must have length >= 1
		if(xidx <= stPos)
			return true;
		int addPos = stPos;
		int addLen = xidx - stPos;
		int addId = partId - 1;
		int delta = std::abs(ylen - addLen);
		int selectStart = std::max(addPos - addId, addPos + delta - (taul - addId));
		int selectEnd = std::min(addPos + addId, addPos + delta + (taul - addId));
		bool found = false;

		for (int i = selectStart; stPos <= selectEnd; i++) {
			bool equal = true;
			for(int l = 0; l < addLen; l++) {
				if(work_dataset[xid][i + l] != query_dataset[yid][addPos + l]) {
					equal = false;
					break;
				}
			}
			if(equal) {
				found = true;
				if(i == selectStart && addId >= 1)
					return iterativeVerifyLeftPartRS(xid, yid, partPos[addId - 1][wlen], addPos, selectStart, wlen, taul - 1, addId);
				else
					return true;
			}
		}

		if(found == false)
			return false;

		return true;
	}

	bool iterativeVerifyRightPartRS(int xid, int yid, int xpos, int ypos, int xlen, int ylen, int partId, int tid) {
		return true;
	}

	// join
	void selfJoin(std::vector<std::pair<int, int>> &finalPairs);
	void RSJoin(std::vector<std::pair<int, int>> &finalPairs);
	// check
	void checkSelfResults() const;
    void printDebugInfo(int currLen) const;
};


class ExactJoinParallel
{
public:
	ExactJoinParallel() = default;
	~ExactJoinParallel() = default;
	ExactJoinParallel(const ExactJoinParallel &other) = delete;
	ExactJoinParallel(ExactJoinParallel &&other) = delete;

public:
	static void exactJoinRS(const std::vector<std::string> &colA, const std::vector<std::string> &colB,
                         	std::vector<std::pair<int, int>> &pairs, ui _maxHeapSize = 0) {
		ui sizeA = colA.size();
		ui sizeB = colB.size();

		std::unordered_map<ui, std::vector<ui>> indexA;
		std::unordered_map<ui, std::vector<ui>> indexB;

		std::vector<std::pair<int, int>> tempPairs[MAXTHREADNUM];
		int eralyTerminated[MAXTHREADNUM] = { 0 };
		ui maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;

		for (ui j = 0; j < sizeA; j++) {
			ui length = colA[j].length();
			// skip empty
#if DROP_EMPTY == 1
			if (length == 0)
				continue;
#endif
			indexA[length].emplace_back(j);
		}
		for (ui j = 0; j < sizeB; j++) {
			ui length = colB[j].length();
			// skip empty
#if DROP_EMPTY == 1
			if (length == 0)
				continue;
#endif
			indexB[length].emplace_back(j);
		}

		for (auto &itA : indexA) {
			ui bucketSizeA = itA.second.size();
			ui bucketSizeB = indexB[itA.first].size();
			const auto &bucketB = indexB[itA.first];
		#pragma omp parallel for
			for (ui ii = 0; ii < bucketSizeA; ii++) {
				int tid = omp_get_thread_num();
				if(eralyTerminated[tid] == 1)
					continue;

				for(ui jj = 0; jj < bucketSizeB; jj++) 
					if(colA[itA.second[ii]] == colB[bucketB[jj]])
						tempPairs[tid].emplace_back(ii, jj);
				
				if(tempPairs[tid].size() >= maxHeapSize)
					eralyTerminated[tid] = 1;
			}
		}

		for (const auto &tp : tempPairs)
			pairs.insert(pairs.end(), tp.begin(), tp.end());
	}

	static void exactJoinSelf(const std::vector<std::string> &col, std::vector<std::pair<int, int>> &pairs, 
							  ui _maxHeapSize = 0) {
		ui size = col.size();

		std::unordered_map<ui, std::vector<ui>> index;
		std::vector<std::pair<int, int>> tempPairs[MAXTHREADNUM];
		int eralyTerminated[MAXTHREADNUM] = { 0 };
		ui maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;

		for (ui j = 0; j < size; j++) {
			ui length = col[j].length();
			// skip empty
#if DROP_EMPTY == 1
			if (length == 0)
				continue;
#endif
			index[length].emplace_back(j);
		}

		for(const auto &it : index) {
			ui bucketSize = it.second.size();
	#pragma omp parallel for
			for(ui ii = 0; ii < bucketSize; ii++) {
				int tid = omp_get_thread_num();
				if(eralyTerminated[tid] == 1)
					continue;

				for(ui jj = ii + 1; jj < bucketSize; jj++) 
					if(col[it.second[ii]] == col[it.second[jj]])
						tempPairs[tid].emplace_back(ii, jj);
				
				if(tempPairs[tid].size() >= maxHeapSize)
					eralyTerminated[tid] = 1;
			}
		}

		for (const auto &tp : tempPairs)
			pairs.insert(pairs.end(), tp.begin(), tp.end());
	}
};


#endif // _STRING_JOIN_PARALLEL_H_