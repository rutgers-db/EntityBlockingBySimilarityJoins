/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _INDEX_H_
#define _INDEX_H_

#include "common/config.h"
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>

/*
 * weighted pair in heap
 */
struct WeightPair
{
    int id1{0};
    int id2{0};
    double val{0.0};

    WeightPair() = default;
    WeightPair(int _id1, int _id2, double _val)
    : id1(_id1), id2(_id2), val(_val) { }

    // min-heap for top k largest
	// make_heap uses < by default, 
	// the entity on top of the heap receives false in every < comparsion
    bool operator<(const WeightPair& rhs) const {
        return this->val > rhs.val;
    }
};

struct WeightPairEdit
{
    int id1{0};
    int id2{0};
    int val{0};

    WeightPairEdit() = default;
    WeightPairEdit(int _id1, int _id2, int _val)
    : id1(_id1), id2(_id2), val(_val) { }

    // min-heap for top k largest
	// make_heap uses < by default, 
	// the entity on top of the heap receives false in every < comparsion
    bool operator<(const WeightPairEdit& rhs) const {
        return this->val < rhs.val;
    }
};

/*
 * Set join
 */
struct SetJoinParelledIndex
{
	// index for partitions and one deletions
	// There are two dimensions:
	// 1st: (Pointer Level) Indicates the range
	// 2nd: (External Vector Level) Indicates the pid (partition ID)
	// Inner vector stores the vector of rid (record ID)
	std::vector<std::vector<unsigned int>> *parts_rids{nullptr}; // <rid>
	std::vector<std::vector<unsigned int>> *ods_rids{nullptr};   // <rid>

	// The index pointers
	std::vector<std::vector<unsigned int>> *parts_index_hv{nullptr};
	std::vector<std::vector<unsigned int>> *od_index_hv{nullptr};
	std::vector<std::vector<unsigned int>> *parts_index_offset{nullptr};
	std::vector<std::vector<unsigned int>> *od_index_offset{nullptr};

	// Mention: if the cnt is greater than the limit of TokenLen, we will just treat it as the maximum value of TokenLen
	std::vector<std::vector<TokenLen>> *parts_index_cnt{nullptr};
	std::vector<std::vector<TokenLen>> *od_index_cnt{nullptr};

	// memory released in functions
	SetJoinParelledIndex() = default;
	~SetJoinParelledIndex() = default;
	SetJoinParelledIndex(const SetJoinParelledIndex& other) = delete;
	SetJoinParelledIndex(SetJoinParelledIndex&& other) = delete;
};


/*
 * String join
 */
struct PIndex 
{
	int stPos{0};    // start position of substring
	int Lo{0};       // start position of segment
	int partLen{0};  // substring/segment length
	int len{0};      // length of indexed string

	PIndex() = default;
	PIndex(int _s, int _o, int _p, int _l)
		: stPos(_s), Lo(_o), partLen(_p), len(_l) { }
	~PIndex() = default;
};

struct hashValue 
{
	int id{0};          // id of string in dataset
	int stPos{0};       // start position of string to hash
	int enPos{0};       // end position of string to hash
	uint64_t value{0};  // hash value

	hashValue(): id(-1), stPos(-1), enPos(-1), value(0) { }
	hashValue(int in_id, int in_stPos, int in_enPos, uint64_t in_value)
		: id(in_id), stPos(in_stPos), enPos(in_enPos), value(in_value) { }
	hashValue(int in_id, int in_stPos, int in_enPos, const std::string &str)
		: id(in_id), stPos(in_stPos), enPos(in_enPos) {
		value = strHash(str, stPos, enPos - stPos + 1);
	}

	// calcuate hash value of string incrementally
	void update(int newid, int newStPos, int newEnPos, const std::string &str, 
				uint64_t *power) {
		// update information if string id is different
		if (newid != id) {
			id = newid;
			enPos = newEnPos;
			stPos = newStPos;
			value = strHash(str, stPos, enPos - stPos + 1);
		} 
		else {
			// check if it can update hash value incrementally
			if (newStPos == stPos + 1) {
				value = value - str[stPos] * power[enPos - stPos];
				stPos = newStPos;
			}
			if (newEnPos == enPos + 1) {
				value = (value * stringHashNumber + str[newEnPos]);
				enPos = newEnPos;
			}
		}
	}

	// Hash
	uint64_t strHash(const std::string &str, int stPos, int len) {
		uint64_t __h = 0;
		int i = 0;
		while (i < len) 
			__h = (__h * stringHashNumber + str[stPos + i++]);
		return __h;
	}
};


/*
 * dsu for clustering pairs
 */
struct DSU
{
	std::vector<int> fa;

	DSU(int _size): fa(_size) { std::iota(fa.begin(), fa.end(), 0); }

	int find(int x) {
		return fa[x] == x ? x : fa[x] = find(fa[x]);
	}

	void unite(int x, int y) {
		fa[find(x)] = find(y);
	}
};

#endif // _INDEX_H_