/*
 * author: Dong Deng 
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _STRING_JOIN_H_
#define _STRING_JOIN_H_

#include "joinutil.h"
#include "index.h"
#include "simfunc.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>


class StringJoin
{
public:
    using InvLists = std::unordered_map<uint64_t, std::vector<int>>;

public:
	int workMinDictLen{19260817};
	int workMaxDictLen{0};
	int queryMinDictLen{19260817};
	int queryMaxDictLen{0};
	int maxDictLen{0};
	int minDictLen{0};
	int workN{0};
	int queryN{0};
	int N{0};
	int D{0}; 							            // threshold
	int PN{0};                                      // part num
	int hashNumber{31};                             // prime number for hashing
	int modNumber{1000000007};                      // prime mod number for hashing
	std::vector<std::string> work_dataset;		    // work dataset
	std::vector<std::string> query_dataset;         // query dataset, null if self-join
	std::vector<std::pair<int, int>> pairs;         // reuslt pairs

	uint64_t candNum{0};  				            // number of candidate pairs
	uint64_t veriNum{0};  				            // number of pairs passed left verification
	uint64_t listNum{0};  				            // number of inverted list matched
	uint64_t realNum{0};  				            // number of results
	std::vector<int> results;   				    // results
	bool valid;            				            // used for early termination
	int left, right;       				            // verify range of left part: [D - left, D + right]
	int _left, _right;     				            // verify range of right part: [D - _left, D + _right]

	int **matrix{nullptr};         	 	            // matrix used to verify left parts
	int **_matrix{nullptr};        		            // matrix used to verify right parts
	bool *quickRef{nullptr};       		            // record results to avoid duplicate verification
	int **partLen{nullptr};        		            // record length of all partitions
	int **partPos{nullptr};        		            // record start position of all partitions
	int *dist{nullptr};            		            // id of string with different length to its former one
	uint64_t *power{nullptr};      		            // the power of 131, used to hash segment and substring
	InvLists **invLists{nullptr}; 		            // inverted index of different string length and different segments
	std::vector<PIndex> **partIndex{nullptr};       // the index record with pair of substring and segment should be checked

	ui maxHeapSize{0};
#if MAINTAIN_VALUE_EDIT == 1
	std::vector<WeightPairEdit> result_pairs_;
	int isHeap = 0;
#endif

public:
	StringJoin() = default;

	StringJoin(const std::vector<std::string>& data, int threshold, ui _maxHeapSize = 0)
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
		for(int i = 0; i < workN; i++) {
			workMaxDictLen = (int)work_dataset[i].size() > workMaxDictLen ? (int)work_dataset[i].size() 
							 : workMaxDictLen;
			workMinDictLen = (int)work_dataset[i].size() < workMinDictLen ? (int)work_dataset[i].size()
							 : workMinDictLen;
		}
		std::sort(work_dataset.begin(), work_dataset.end(), StringJoinUtil::strLessT);

		minDictLen = workMinDictLen;
		maxDictLen = workMaxDictLen;
		N = workN;

		printf("Work size: %zu\tQuery size: %zu\n", work_dataset.size(), query_dataset.size());
		printf("Min work record's size: %d\tMax work record's size: %d\n", workMinDictLen, workMaxDictLen);

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;
#if MAINTAIN_VALUE_EDIT == 1
		result_pairs_.reserve(maxHeapSize);
#endif
	}

	StringJoin(const std::vector<std::string>& work, const std::vector<std::string>& query, int threshold, 
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
		for(int i = 0; i < workN; i++) {
			workMaxDictLen = (int)work_dataset[i].size() > workMaxDictLen ? (int)work_dataset[i].size() 
							 : workMaxDictLen;
			workMinDictLen = (int)work_dataset[i].size() < workMinDictLen ? (int)work_dataset[i].size()
							 : workMinDictLen;
		}
		for(int i = 0; i < queryN; i++) {
			queryMaxDictLen = (int)query_dataset[i].size() > queryMaxDictLen ? (int)query_dataset[i].size() 
							 : queryMaxDictLen;
			queryMinDictLen = (int)query_dataset[i].size() < queryMinDictLen ? (int)query_dataset[i].size()
							 : queryMinDictLen;
		}
		std::sort(work_dataset.begin(), work_dataset.end(), StringJoinUtil::strLessT);
		std::sort(query_dataset.begin(), query_dataset.end(), StringJoinUtil::strLessT);

		minDictLen = std::min(workMinDictLen, queryMinDictLen);
		maxDictLen = std::max(workMaxDictLen, queryMaxDictLen);
		N = std::max(workN, queryN);

		printf("Work size: %zu\tQuery size: %zu\n", work_dataset.size(), query_dataset.size());
		printf("Min work record's size: %d\tMax work record's size: %d\n", workMinDictLen, workMaxDictLen);
		printf("Min query record's size: %d\tMax query record's size: %d\n", queryMinDictLen, queryMaxDictLen);

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;
#if MAINTAIN_VALUE_EDIT == 1
		result_pairs_.reserve(maxHeapSize);
#endif
	}

	~StringJoin() {
		for (int lp = 0; lp < PN; lp++) {
			delete[] partPos[lp];
			delete[] partLen[lp];
			delete[] invLists[lp];
			delete[] partIndex[lp];
		}
		delete[] partPos[PN];

		for (int lp = maxDictLen; lp >= 0; lp--) {
			delete[] matrix[lp];
			delete[] _matrix[lp];
		}

		delete[] matrix;
		delete[] _matrix;
		delete[] dist;
		delete[] power;
		delete[] partPos;
		delete[] partLen;
		delete[] invLists;
		delete[] partIndex;
		delete[] quickRef;

		std::cout << "destructor" << std::endl << std::flush;
	}
    
	StringJoin(const StringJoin& other) = delete;
	StringJoin(StringJoin&& other) = delete;

public:
	// Pre-process
	void init();
	void prepareSelf();
	void prepareRS();

	// verify
	bool verifyLeftPartSelf(int xid, int yid, int xlen, int ylen, int Tau);
	bool verifyRightPartSelf(int xid, int yid, int xlen, int ylen, int xpos, int ypos, int Tau);
	bool verifyLeftPartRS(int xid, int yid, int xlen, int ylen, int Tau);
	bool verifyRightPartRS(int xid, int yid, int xlen, int ylen, int xpos, int ypos, int Tau);

	// join
	void selfJoin(std::vector<std::pair<int, int>> &finalPairs);
	void RSJoin(std::vector<std::pair<int, int>> &finalPairs);

	// check
	void checkSelfResults() const;
	void printDebugInfo(int currLen) const;
};


class ExactJoin
{
public:
	ExactJoin() = default;
	~ExactJoin() = default;
	ExactJoin(const ExactJoin &other) = delete;
	ExactJoin(ExactJoin &&other) = delete;

public:
	static void exactJoinRS(const std::vector<std::string> &colA, const std::vector<std::string> &colB,
                         std::vector<std::pair<int, int>> &pairs, ui _maxHeapSize = 0) {
		ui sizeA = colA.size();
		ui sizeB = colB.size();

		std::unordered_map<ui, std::vector<ui>> indexA;
		std::unordered_map<ui, std::vector<ui>> indexB;
		ui maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;

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
			for (ui ii = 0; ii < bucketSizeA; ii++) {
				for (ui jj = 0; jj < bucketSizeB; jj++)
					if (colA[itA.second[ii]] == colB[bucketB[jj]])
						pairs.emplace_back(ii, jj);
				if(pairs.size() > maxHeapSize)
					return;
			}
		}
	}

	static void exactJoinSelf(const std::vector<std::string> &col, std::vector<std::pair<int, int>> &pairs, 
							  ui _maxHeapSize = 0) {
		ui size = col.size();
		std::unordered_map<ui, std::vector<ui>> index;
		ui maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;

		for (ui j = 0; j < size; j++) {
			ui length = col[j].length();
			// skip empty
#if DROP_EMPTY == 1
			if (length == 0)
				continue;
#endif
			index[length].emplace_back(j);
		}

		for (auto &it : index) {
			ui bucketSize = it.second.size();
			for (ui ii = 0; ii < bucketSize; ii++) {
				for (ui jj = ii + 1; jj < bucketSize; jj++)
					if (col[it.second[ii]] == col[it.second[jj]])
						pairs.emplace_back(ii, jj);
				if(pairs.size() > maxHeapSize)
					return;
			}
		}
	}
};


#endif // _STRING_JOIN_H_