/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _OVLPRSJOIN_H_
#define _OVLPRSJOIN_H_

#include "config.h"
#include "type.h"
#include "index.h"
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <string.h>
#include <queue>
#include <inttypes.h>
#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <sys/sysinfo.h>
#include <chrono>
#include <assert.h>

#define BRUTEFORCE 0
#define REPORT_INDEX 0
#define REPORT_BINARY 0

/*
 * Overlap join according to SIGMOD2018: Overlap join with theoritical guaranteen
 * for sequential overlap joiner, the input dataset will not have a large size
 * thus, it is okay for taking all of them as small sets
 * so we only implement the heap-based method
 */

class OvlpRSJoin;
class OvlpSelfJoin;

struct combination2;
struct combination1
{
public:
    int N{0};
    int id{0};
    bool completed{false};
    std::vector<int> curr;

    combination1(int d, int beg, const OvlpRSJoin &joiner);
	combination1(int d, int beg, const OvlpSelfJoin &joiner);

    int getlastcurr(const OvlpRSJoin &joiner);
	int getlastcurr(const OvlpSelfJoin &joiner);

    // compute next combination_test1
    void next(const OvlpRSJoin &joiner);
	void next(const OvlpSelfJoin &joiner);

    void print(const OvlpRSJoin &joiner) const;
	void print(const OvlpSelfJoin &joiner) const;

    bool stepback(const int i, const OvlpRSJoin &joiner);
	bool stepback(const int i, const OvlpSelfJoin &joiner);

    void binary(const combination1 &value, const OvlpSelfJoin &joiner);
    void binary(const combination2 &value, const OvlpRSJoin &joiner);

	// test unit
	bool ifsame(const std::vector<ui> &data, const OvlpRSJoin &joiner);
};

struct combination2
{
public:
    int N{0};
    int id{0};
    bool completed{false};
    std::vector<int> curr;

    combination2(int d, int beg, const OvlpRSJoin &joiner);

    int getlastcurr(const OvlpRSJoin &joiner);

    // compute next combination2
    void next(const OvlpRSJoin &joiner);

    void print(const OvlpRSJoin &joiner) const;

    bool stepback(const int i, const OvlpRSJoin &joiner);

    void binary(const combination2 &value, const OvlpRSJoin &joiner);
    void binary(const combination1 &value, const OvlpRSJoin &joiner);

	// test unit
	bool ifsame(const std::vector<ui> &data, const OvlpRSJoin &joiner);
};


class OvlpSelfJoin
{
public:
    int n{0};
    int c{0};
    ui total_eles{0};

    std::vector<std::vector<ui>> records; // two sets
	std::vector<std::vector<ui>> datasets; // two working datasets
	std::vector<double> recWeights;
	std::vector<double> wordwt;
    std::vector<std::pair<int, int>> idmap_records;
    std::vector<std::vector<std::pair<int, int>>> ele_lists;
    std::vector<std::pair<int, int>> result_pairs;
	std::vector<int> heap;
	std::vector<combination1> combs;
	ui maxHeapSize{0};
#if MAINTAIN_VALUE_OVLP == 1
	bool isWeightedComp{false};
	// only build heap when the res size is greater than MAX
	std::vector<WeightPair> result_pairs_;
	int isHeap{0};
#endif

    int64_t candidate_num{0};
    int64_t result_num{0};

    void overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs);
    void small_case(int L, int R, std::vector<std::pair<int, int>> &finalPairs);

    OvlpSelfJoin(const std::vector<std::vector<ui>> &sorted_records, const std::vector<double> &_recWeights, 
				 const std::vector<double> _wordwt, ui _maxHeapSize = 0, bool _isWeightedComp = false)
	: records(sorted_records), recWeights(_recWeights), wordwt(_wordwt) {
        // reset everything
        c = 0;
        result_num = 0;
        candidate_num = 0;

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;
#if MAINTAIN_VALUE_OVLP == 1
		isWeightedComp = _isWeightedComp;
		result_pairs_.reserve(maxHeapSize);
#endif
    }

public:
	bool comp_comb1(const int a, const int b) {
		// cout << c << " ";
		auto & c1 = combs[a];
		auto & c2 = combs[b];
		for (ui i = 0; i < c; i++) {
			if (datasets[c1.id][c1.curr[i]] > datasets[c2.id][c2.curr[i]])
				return false;
			else if (datasets[c1.id][c1.curr[i]] < datasets[c2.id][c2.curr[i]])
				return true;
		}
		return c1.id > c2.id;
	}

	double weightedOverlapCoeff(int id1, int id2) {
		std::vector<ui> res;
		std::set_intersection(records[id1].begin(), records[id1].end(), 
						      records[id2].begin(), records[id2].end(), 
						      std::back_inserter(res));
		double ovlp = 0.0;
		for(const auto &e : res)
			ovlp += wordwt[e];
		return ovlp / std::min(recWeights[id1], recWeights[id2]);
		// return ovlp;
	}

	double overlapCoeff(int id1, int id2) {
		std::vector<ui> res;
		std::set_intersection(records[id1].begin(), records[id1].end(), 
						      records[id2].begin(), records[id2].end(), 
						      back_inserter(res));
		
		double ovlp = res.size() * 1.0;
		return ovlp / std::min(records[id1].size(), records[id2].size()) * 1.0;
		// return ovlp;
	}

public:
	// build heap for combination
	bool build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				int L, std::vector<int> &heap, std::vector<combination1> &combs, int &heap_size);
};


class OvlpRSJoin 
{
public:
    int n1{0}, n2{0}; // R S sizes
	int c{0}; // threshold
	ui total_eles{0};
    
    std::vector<std::vector<ui>> records1, records2; // two sets
	std::vector<std::vector<ui>> datasets1, datasets2; // two working datasets
	std::vector<double> recWeights1, recWeights2;
	std::vector<double> wordwt;
    std::vector<std::pair<int, int>> idmap_records1, idmap_records2;
    std::vector<std::vector<std::pair<int, int>>> ele_lists1, ele_lists2;
    std::vector<std::pair<int, int>> result_pairs;
	std::vector<int> heap1, heap2;
	std::vector<combination1> combs1;
	std::vector<combination2> combs2; // comb	
	ui maxHeapSize{0};
#if MAINTAIN_VALUE_OVLP == 1
	bool isWeightedComp{false};
	// only build heap when the res size is greater than MAX
	std::vector<WeightPair> result_pairs_;
	int isHeap{0};
#endif

    int64_t candidate_num;
    int64_t result_num;

    void overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs);
    void small_case(int L1, int R1, int L2, int R2, std::vector<std::pair<int, int>> &finalPairs);

    bool if_external_IO = false;
    std::string resultPair_storePath;

    OvlpRSJoin(const std::vector<std::vector<ui>> &sorted_records_1, const std::vector<std::vector<ui>> &sorted_records_2, 
			   const std::vector<double> &_recWeights1, const std::vector<double> &_recWeights2, 
			   const std::vector<double> &_wordwt, ui _maxHeapSize = 0, bool _isWeightedComp = false)
	: records1(sorted_records_1), records2(sorted_records_2), recWeights1(_recWeights1), recWeights2(_recWeights2), 	
	wordwt(_wordwt) {
        // reset everything
        c = 0;
		result_num = 0;
		candidate_num = 0;

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;
#if MAINTAIN_VALUE_OVLP == 1
		isWeightedComp = _isWeightedComp;
		result_pairs_.reserve(maxHeapSize);
#endif
    }

    void set_external_store(const std::string &_resPair_path){
        if_external_IO = true;
        resultPair_storePath = _resPair_path;
    }

    // void save_idmap(string _resPair_path);

public:
	bool comp_comb1(const int a, const int b) {
		// cout << c << " ";
		auto & c1 = combs1[a];
		auto & c2 = combs1[b];
		for (ui i = 0; i < c; i++) {
			if (datasets1[c1.id][c1.curr[i]] > datasets1[c2.id][c2.curr[i]])
				return false;
			else if (datasets1[c1.id][c1.curr[i]] < datasets1[c2.id][c2.curr[i]])
				return true;
		}
		return c1.id > c2.id;
	}
	bool comp_comb2(const int a, const int b) {
		auto & c1 = combs2[a];
		auto & c2 = combs2[b];
		for (ui i = 0; i < c; i++) {
			if (datasets2[c1.id][c1.curr[i]] > datasets2[c2.id][c2.curr[i]])
				return false;
			else if (datasets2[c1.id][c1.curr[i]] < datasets2[c2.id][c2.curr[i]])
				return true;
		}
		return c1.id > c2.id;
	}

	double weightedOverlapCoeff(int id1, int id2) {
		std::vector<ui> res;
		std::set_intersection(records1[id1].begin(), records1[id1].end(), 
						      records2[id2].begin(), records2[id2].end(), 
						      std::back_inserter(res));
		double ovlp = 0.0;
		for(const auto &e : res)
			ovlp += wordwt[e];
		return ovlp / std::min(recWeights1[id1], recWeights2[id2]);
		// return ovlp;
	}

	double overlapCoeff(int id1, int id2) {
		std::vector<ui> res;
		std::set_intersection(records1[id1].begin(), records1[id1].end(), 
						      records2[id2].begin(), records2[id2].end(), 
						      back_inserter(res));
		
		double ovlp = res.size() * 1.0;
		return ovlp / std::min(records1[id1].size(), records2[id2].size()) * 1.0;
		// return ovlp;
	}

public:
	// build heap for combination1
	bool build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				int L, std::vector<int> &heap, std::vector<combination1> &combs, int &heap_size);
	// build heap for combination2
	bool build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				int L, std::vector<int> &heap, std::vector<combination2> &combs, int &heap_size);
};



class OvlpUtil
{
public:
	OvlpUtil() = default;
	~OvlpUtil() = default;
	OvlpUtil(const OvlpUtil& other) = delete;
	OvlpUtil(OvlpUtil&& other) = delete;

public:
	static bool comp_int(const int a, const int b) {
		return a > b;
	}
	static bool comp_pair(const std::pair<int, int> &p1, const int val) {
		return p1.first < val;
	}
	static bool is_equal(const combination1 & c1, const combination1 & c2, 
							  const OvlpRSJoin &joiner) {
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets1[c1.id][c1.curr[i]] != joiner.datasets1[c2.id][c2.curr[i]])
				return false;
		}
		return true;
	}
	static bool is_equal(const combination1 & c1, const combination1 & c2, 
							  const OvlpSelfJoin &joiner) {
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets[c1.id][c1.curr[i]] != joiner.datasets[c2.id][c2.curr[i]])
				return false;
		}
		return true;
	}
	static bool is_equal(const combination2 & c1, const combination2 & c2, 
							  const OvlpRSJoin &joiner) {
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets2[c1.id][c1.curr[i]] != joiner.datasets2[c2.id][c2.curr[i]])
				return false;
		}
		return true;
	}
	static int compare(const combination1 & c1, const combination2 & c2,
							 const OvlpRSJoin &joiner) {
		// cout << joiner.c << ' ';
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets1[c1.id][c1.curr[i]] > joiner.datasets2[c2.id][c2.curr[i]])
				return 1;
			else if (joiner.datasets1[c1.id][c1.curr[i]] < joiner.datasets2[c2.id][c2.curr[i]])
				return -1;
		}
		return 0;
	}

public:
	static void removeShort(const std::vector<std::vector<ui>> &records, std::unordered_map<ui, std::vector<int>> &ele, 
							const OvlpRSJoin &joiner);
	// Remove "widows" from a hash map based on another hash map.
	// This function removes key-value pairs from the unordered_map 'ele' 
	// if the key doesn't exist in another unordered_map 'ele_other'.
	static void removeWidow(std::unordered_map<ui, std::vector<int>> &ele, const std::unordered_map<ui, std::vector<int>> &ele_other);
	static void transform(std::unordered_map<ui, std::vector<int>> &ele, const std::vector<std::pair<int, int>> &eles, 
               			  std::vector<std::pair<int, int>> &idmap, std::vector<std::vector<std::pair<int, int>>> &ele_lists,
               			  std::vector<std::vector<ui>> &dataset, const ui total_eles, const int n, const OvlpRSJoin &joiner);
};



#endif // OVLPRSJOIN
