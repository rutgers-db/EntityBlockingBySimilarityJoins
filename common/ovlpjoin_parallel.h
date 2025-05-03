/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _OVLP_JOIN_PARALLEL_H_
#define _OVLP_JOIN_PARALLEL_H_

#include "config.h"
#include "type.h"
#include "index.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <queue>
#include <array>
#include <chrono>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <assert.h>
#include <omp.h>

class OvlpRSJoinParallel;
class OvlpSelfJoinParallel;

struct combination_p2;
struct combination_p1
{
public:
    int N{0};
    int id{0};
    bool completed{false};
    std::vector<int> curr;

    combination_p1(int d, int beg, const OvlpRSJoinParallel& joiner);
	combination_p1(int d, int beg, const OvlpSelfJoinParallel &joiner);

    inline int getlastcurr(const OvlpRSJoinParallel& joiner);
	inline int getlastcurr(const OvlpSelfJoinParallel &joiner);

    // compute next combination_p1
    void next(const OvlpRSJoinParallel &joiner);
    void print(const OvlpRSJoinParallel &joiner) const;
    bool stepback(const int i, const OvlpRSJoinParallel &joiner);
	bool stepback(const int i, const OvlpSelfJoinParallel &joiner);
    void binary(const combination_p1 &value, const OvlpSelfJoinParallel &joiner);
    void binary(const combination_p2 &value, const OvlpRSJoinParallel &joiner);

	// test unit
	bool ifsame(const std::vector<ui> &data, const OvlpRSJoinParallel &joiner);
};

struct combination_p2
{
public:
    int N{0};
    int id{0};
    bool completed{false};
    std::vector<int> curr;

    combination_p2(int d, int beg, const OvlpRSJoinParallel &joiner);

    inline int getlastcurr(const OvlpRSJoinParallel &joiner);

    // compute next combination_p2
    void next(const OvlpRSJoinParallel &joiner);
    void print(const OvlpRSJoinParallel &joiner) const;
    bool stepback(const int i, const OvlpRSJoinParallel &joiner);
    void binary(const combination_p2 &value, const OvlpRSJoinParallel &joiner);
    void binary(const combination_p1 &value, const OvlpRSJoinParallel &joiner);

	// test unit
	bool ifsame(const std::vector<ui> &data, const OvlpRSJoinParallel &joiner);
};


/*
  This is a class that uses overlapjoin for two datasets (Set R and set S) to join them
  We gonna implement it in a parelled method
*/
class OvlpRSJoinParallel
{
public:
    int n1{0}, n2{0}; // R S sizes
	int c{0}; // threshold
	ui total_eles{0};
    
    std::vector<std::vector<ui>> records1, records2;                           // two sets
	std::vector<std::vector<ui>> datasets1, datasets2;                         // two working datasets
	std::vector<double> recWeights1, recWeights2;
	std::vector<double> wordwt;
    std::vector<std::pair<int, int>> idmap_records1, idmap_records2;
    std::vector<std::vector<std::pair<int, int>>> ele_lists1, ele_lists2;
    std::vector<std::pair<int, int>> result_pairs[MAXTHREADNUM];
	std::vector<int> heap1[MAXTHREADNUM], heap2[MAXTHREADNUM];
	std::vector<combination_p1> combs1[MAXTHREADNUM];
	std::vector<combination_p2> combs2[MAXTHREADNUM];                          // comb	
	ui maxHeapSize{0};
#if MAINTAIN_VALUE_OVLP == 1
	bool isWeightedComp{false};
	// only build heap when the res size is greater than MAX
	std::vector<WeightPair> result_pairs_[MAXTHREADNUM];
	int isHeap[MAXTHREADNUM] = { 0 };
#endif

    int64_t candidate_num;
    int64_t result_num;

    void overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs);
    void small_case(int L1, int R1, int L2, int R2, std::vector<std::pair<int, int>> &finalPairs);

    bool if_external_IO = false;
    std::string resultPair_storePath;

    OvlpRSJoinParallel(const std::vector<std::vector<ui>> &sorted_records_1, const std::vector<std::vector<ui>> &sorted_records_2, 
					   const std::vector<double> &rec1wt, const std::vector<double> &rec2wt, const std::vector<double> &_wordwt, 
					   ui _maxHeapSize = 0, bool _isWeightedComp = false)
	: records1(sorted_records_1), records2(sorted_records_2), recWeights1(rec1wt), recWeights2(rec2wt), wordwt(_wordwt) {
        // reset everything
        c = 0;
		result_num = 0;
		candidate_num = 0;

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;
#if MAINTAIN_VALUE_OVLP == 1
		isWeightedComp = _isWeightedComp;
        for(int tid = 0; tid < MAXTHREADNUM; tid++)
            result_pairs_[tid].reserve(maxHeapSize);
#endif
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


    void set_external_store(const std::string &_resPair_path){
        if_external_IO = true;
        resultPair_storePath = _resPair_path;
    }

public:
	bool comp_comb1(const int a, const int b, int tid) {
		// cout << c << " ";
		auto & c1 = combs1[tid][a];
		auto & c2 = combs1[tid][b];
		for (int i = 0; i < c; i++) {
			if (datasets1[c1.id][c1.curr[i]] > datasets1[c2.id][c2.curr[i]])
				return false;
			else if (datasets1[c1.id][c1.curr[i]] < datasets1[c2.id][c2.curr[i]])
				return true;
		}
		return c1.id > c2.id;
	}
	bool comp_comb2(const int a, const int b, int tid) {
		auto & c1 = combs2[tid][a];
		auto & c2 = combs2[tid][b];
		for (int i = 0; i < c; i++) {
			if (datasets2[c1.id][c1.curr[i]] > datasets2[c2.id][c2.curr[i]])
				return false;
			else if (datasets2[c1.id][c1.curr[i]] < datasets2[c2.id][c2.curr[i]])
				return true;
		}
		return c1.id > c2.id;
	}

public:
	// build heap for combination_p1
	bool build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				int L, std::vector<int> &heap, std::vector<combination_p1> &combs, int &heap_size, 
					int tid);
	// build heap for combination_p2
	bool build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				int L, std::vector<int> &heap, std::vector<combination_p2> &combs, int &heap_size, 
					int tid);
};


class OvlpSelfJoinParallel
{
public:
    int n1{0}; // size
	int c{0}; // threshold
	ui total_eles{0};
	int earlyTerminated[MAXTHREADNUM] = { 0 };
    
    std::vector<std::vector<ui>> records; // set
	std::vector<std::vector<ui>> datasets; // working datasets
	std::vector<double> weights;
	std::vector<double> wordwt;
    std::vector<std::pair<int, int>> idmap_records;
    std::vector<std::vector<std::pair<int, int>>> ele_lists;
    std::vector<std::pair<int, int>> result_pairs[MAXTHREADNUM];
	std::vector<int> heap[MAXTHREADNUM];
	std::vector<combination_p1> combs[MAXTHREADNUM]; // comb	
	std::unordered_set<int> random_ids;
	std::vector<std::pair<int, int>> buck;
	ui maxHeapSize{0};
#if MAINTAIN_VALUE_OVLP == 1
	bool isWeightedComp{false};
	// only build heap when the res size is greater than MAX
	std::vector<WeightPair> result_pairs_[MAXTHREADNUM];
	int isHeap[MAXTHREADNUM] = { 0 };
#endif
    int64_t candidate_num{0};
    int64_t result_num{0};
	int64_t list_cost{0};
	double heap_cost{0.0};
	double binary_cost{0.0};
	uint64_t heap_op{0};
	int64_t large_cost{0};
	int64_t large_est_cost{0};
	int alive_id{0};

    void overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs);
    void small_case(int L, int R, std::vector<std::pair<int, int>> &finalPairs);
	// large case
	int64_t small_estimate(int L, int R);
	int64_t large_estimate(int L, int R);
	int divide(int nL);
	int estimate();
	void large_case(int L, int R, std::vector<std::pair<int, int>> &finalPairs);

    bool if_external_IO = false;
    std::string resultPair_storePath;

    OvlpSelfJoinParallel(const std::vector<std::vector<ui>> &sorted_records, const std::vector<double> &recwt, 
						 const std::vector<double> &_wordwt, ui _maxHeapSize = 0, bool _isWeightedComp = false)
	: records(sorted_records), weights(recwt), wordwt(_wordwt) {
        // reset everything
        c = 0;
		result_num = 0;
		candidate_num = 0;

		maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;
#if MAINTAIN_VALUE_OVLP == 1
		isWeightedComp = _isWeightedComp;
        for(int tid = 0; tid < MAXTHREADNUM; tid++)
            result_pairs_[tid].reserve(maxHeapSize);
#endif
    }

	double weightedOverlapCoeff(int id1, int id2) {
		std::vector<ui> res;
		std::set_intersection(records[id1].begin(), records[id1].end(), 
						      records[id2].begin(), records[id2].end(), 
						      std::back_inserter(res));
		double ovlp = 0.0;
		for(const auto &e : res)
			ovlp += wordwt[e];
		// return ovlp / std::min(weights[id1], weights[id2]);
		return ovlp;
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

    void set_external_store(const std::string &_resPair_path){
        if_external_IO = true;
        resultPair_storePath = _resPair_path;
    }


public:
	bool comp_comb1(const int a, const int b, int tid) {
		// cout << c << " ";
		auto & c1 = combs[tid][a];
		auto & c2 = combs[tid][b];
		for (int i = 0; i < c; i++) {
			if (datasets[c1.id][c1.curr[i]] > datasets[c2.id][c2.curr[i]])
				return false;
			else if (datasets[c1.id][c1.curr[i]] < datasets[c2.id][c2.curr[i]])
				return true;
		}
		return c1.id > c2.id;
	}

public:
	// build heap for combination_p1
	bool build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				int L, std::vector<int> &heap, std::vector<combination_p1> &combs, int &heap_size, 
					int tid);
};



class OvlpUtilParallel
{
public:
	OvlpUtilParallel() = default;
	~OvlpUtilParallel() = default;
	OvlpUtilParallel(const OvlpUtilParallel& other) = delete;
	OvlpUtilParallel(OvlpUtilParallel&& other) = delete;

public:
	static bool comp_int(const int a, const int b) {
		return a > b;
	}
	static bool comp_pair(const std::pair<int, int> &p1, const int val) {
		return p1.first < val;
	}
	static bool is_equal(const combination_p1 & c1, const combination_p1 & c2, 
						 const OvlpRSJoinParallel &joiner) {
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets1[c1.id][c1.curr[i]] != joiner.datasets1[c2.id][c2.curr[i]])
				return false;
		}
		return true;
	}
	static bool is_equal(const combination_p1 & c1, const combination_p1 & c2, 
						 const OvlpSelfJoinParallel &joiner) {
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets[c1.id][c1.curr[i]] != joiner.datasets[c2.id][c2.curr[i]])
				return false;
		}
		return true;
	}
	static bool is_equal(const combination_p2 & c1, const combination_p2 & c2, 
						 const OvlpRSJoinParallel &joiner) {
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets2[c1.id][c1.curr[i]] != joiner.datasets2[c2.id][c2.curr[i]])
				return false;
		}
		return true;
	}
	static int compare(const combination_p1 & c1, const combination_p2 & c2,
					   const OvlpRSJoinParallel &joiner) {
		// cout << joiner.c << ' ';
		for (int i = 0; i < joiner.c; i++) {
			if (joiner.datasets1[c1.id][c1.curr[i]] > joiner.datasets2[c2.id][c2.curr[i]])
				return 1;
			else if (joiner.datasets1[c1.id][c1.curr[i]] < joiner.datasets2[c2.id][c2.curr[i]])
				return -1;
		}
		return 0;
	}
	static int64_t nchoosek(int64_t n, int64_t k) {
		if (k == 0) return 1;
		return (n * nchoosek(n - 1, k - 1)) / k;
	}

public:
	static void removeShort(const std::vector<std::vector<ui>> &records, std::unordered_map<ui, std::vector<int>> &ele, 
							const OvlpRSJoinParallel &joiner);
	// Remove "widows" from a hash map based on another hash map.
	// This function removes key-value pairs from the unordered_map 'ele' 
	// if the key doesn't exist in another unordered_map 'ele_other'.
	static void removeWidow(std::unordered_map<ui, std::vector<int>> &ele, const std::unordered_map<ui, std::vector<int>> &ele_other);
	static void transform(std::unordered_map<ui, std::vector<int>> &ele, const std::vector<std::pair<int, int>> &eles, 
               			  std::vector<std::pair<int, int>> &idmap, std::vector<std::vector<std::pair<int, int>>> &ele_lists,
               			  std::vector<std::vector<ui>> &dataset, const ui total_eles, const int n, const OvlpRSJoinParallel &joiner);
};



#endif // _OVLP_JOIN_PARALLEL_H_
