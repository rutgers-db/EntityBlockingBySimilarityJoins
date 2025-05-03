/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _SETJOIN_H_
#define _SETJOIN_H_

#include "config.h"
#include "type.h"
#include "joinutil.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstdio>
#include <string.h>
#include <inttypes.h>
#include <assert.h>


// SetJoin, single thread
/*
 * adaptive grouping mechanism is only suitable for threshold bigger than 0.5
 * since the \alpha is within [0.5, 1]
 * and that's reason why we have a seg fault when threshold is smaller than 0.5
 * thus we just adopt length filter
 */
class SetJoin 
{
public:
	double overlap_cost = 0;
	double allocation_cost = 0;
	double index_cost = 0;

public:
    bool ifRS = false;
	// join func
	SimFuncType simFType{SimFuncType::JACCARD};
    double (SetJoin::*weightedFunc)(ui, ui) = nullptr;
    double (SetJoin::*normalFunc)(ui, ui) = nullptr;
    bool (SetJoin::*overlapFunc)(int, int, int, int, int) = nullptr;
	// index
	std::vector<std::pair<int, int>> cacheVec;
	std::vector<std::vector<std::pair<int, int>>> indexVecs;

public:
    double det; // delta in vldb2016-setjoin
    uint64_t resultNum = 0;
    uint64_t candidateNum = 0;
    uint64_t lengthSum = 0;
    uint64_t listlens = 0;

    int prime_exp[MAX_LINE_LENGTH];
    std::vector<std::vector<ui>> dataset_all;
	std::vector<std::vector<ui>> work_dataset;
    std::vector<std::vector<ui>> query_dataset;
    std::vector<double> work_weights;
    std::vector<double> query_weights;
    std::vector<double> wordwt;
	// bucket for empty tokenzied sets
	std::vector<ui> workEmpty, queryEmpty;
    std::vector<std::pair<int, int>> result_pairs;
    // io
    std::string simP_file_path;
    // heap
    ui maxHeapSize{0};
#if MAINTAIN_VALUE == 1
    bool isWeightedComp{false}; // use the weighted version sim funcs
    std::vector<WeightPair> result_pairs_;
    int isHeap = 0;
#endif

    struct invertedList {
        int vec_no, cnt;
        std::pair<int, int> cache[CACHE_SIZE];

        std::vector<std::pair<int, int>>& getVector(SetJoin *joiner) const {
            if (cnt <= CACHE_SIZE) {
                joiner->cacheVec.assign(cache, cache + cnt);
                return joiner->cacheVec;
            } else
                return joiner->indexVecs[vec_no];
        }

        void add(std::pair<int, int> data, SetJoin *joiner) {
            if (cnt < CACHE_SIZE) 
				cache[cnt++] = data;
            else {
                if (CACHE_SIZE == cnt) {
                    joiner->indexVecs.push_back(std::vector<std::pair<int, int>>());
                    vec_no = joiner->indexVecs.size() - 1;
                    joiner->indexVecs[vec_no].assign(cache, cache + CACHE_SIZE);
                }
                ++cnt;
                joiner->indexVecs[vec_no].push_back(data);
            }
        }
    };

    struct invIndexStruct {
        unsigned long long list_no;
        int* oneList;

        invIndexStruct() : list_no(0) {}
    };

    std::vector<invertedList> indexLists;
    
public:
	SetJoin() = default;
    SetJoin(const std::vector<std::vector<ui>> &sorted_records, const std::vector<double> &recwt,
            const std::vector<double> &_wordwt, std::string _sim_pairs_filepath, double _det, 
            ui _maxHeapSize = 0, bool _isWeightedComp = false)
    : ifRS(false), work_dataset(sorted_records), work_weights(recwt), wordwt(_wordwt), simP_file_path(_sim_pairs_filepath) {   
        indexVecs.clear();
        cacheVec.clear();
        cacheVec.resize(CACHE_SIZE);

		ui min_records_size = sorted_records.front().size();
		ui max_records_size = sorted_records.back().size();
		
#if RESIZE_DATA == 1
        if(_det <= 0.4 && max_records_size >= 55) {
            resizeData(work_dataset);
            max_records_size = work_dataset.back().size();
        }
#endif

		printf("Work size: %zu\n", work_dataset.size());
		printf("Min record's size: %u\tMax record's size: %u\n", min_records_size, max_records_size);

        maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;
#if MAINTAIN_VALUE == 1
        isWeightedComp = _isWeightedComp;
        result_pairs_.reserve(maxHeapSize);
#endif
    }
	// RS-join
	SetJoin(const std::vector<std::vector<ui>> &work_records, const std::vector<std::vector<ui>> &query_records, 
            const std::vector<double> &workwt, const std::vector<double> &querywt, const std::vector<double> &_wordwt,
			const std::string &_sim_pairs_filepath, double _det, ui _maxHeapSize = 0, bool _isWeightedComp = false)
    : ifRS(true), work_dataset(work_records), query_dataset(query_records), work_weights(workwt), query_weights(querywt), 
    wordwt(_wordwt), simP_file_path(_sim_pairs_filepath) {
        indexVecs.clear();
        cacheVec.clear();
        cacheVec.resize(CACHE_SIZE);

		ui min_work_size = work_dataset.front().size();
		ui min_query_size = query_dataset.front().size();
		ui max_work_size = work_dataset.back().size();
		ui max_query_size = query_dataset.back().size();

#if RESIZE_DATA == 1
        if(_det <= 0.4 && max_work_size >= 55) {
            resizeData(work_dataset);
            max_work_size = work_dataset.back().size();
        }
        if(_det <= 0.4 && max_query_size >= 55) {
            resizeData(query_dataset);
            max_query_size = query_dataset.back().size();
        }
#endif

		printf("Work size: %zu\tQuery size: %zu\n", work_dataset.size(), query_dataset.size());
		printf("Min work record's size: %u\tMax work record's size: %u\n", min_work_size, max_work_size);
		printf("Min query record's size: %u\tMax query record's size: %u\n", min_query_size, max_query_size);

        maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE_SERIAL : _maxHeapSize;
#if MAINTAIN_VALUE == 1
        isWeightedComp = _isWeightedComp;
        result_pairs_.reserve(maxHeapSize);
#endif
	}

    ~SetJoin() {
        cacheVec.clear();
        indexVecs.clear();
    }

public:
	void loadDataset(const std::vector<std::vector<ui>> &records, std::string file) {
		simP_file_path = file;
        dataset_all = records;
		work_dataset.resize(dataset_all.size());
        indexVecs.clear();
        cacheVec.clear();
        cacheVec.resize(CACHE_SIZE);
	}
	void prepare(const std::vector<std::vector<ui>> &offsets, ui column) {
		ui size = dataset_all.size();
		for(ui i = 0; i < size; i++)
			work_dataset[i].clear();
		for(ui i = 0; i < size; i++) {
			ui prefix = offsets[i][column];
			ui suffix = offsets[i][column + 1];
			// for(ui j = prefix; j < suffix; j++) 
			// 	work_dataset[i].emplace_back(dataset_all[i][j]);
			work_dataset[i].assign(dataset_all[i].begin() + prefix, 
								   dataset_all[i].begin() + suffix);
		}
	}

	void resizeData(std::vector<std::vector<ui>> &dataset) {
         // [\sigma - 2 * sd, \sigma + 2 * sd]
        int unempty = 0;
        double avergaeSize = 0.0;
        for(const auto &rec : dataset) {
            if(rec.empty()) continue;
            avergaeSize += (double)rec.size() * 1.0;
            ++ unempty;
        }
        avergaeSize = avergaeSize / (unempty * 1.0);
        long double sd = 0.0;
        for(const auto &rec : dataset) {
            if(rec.empty()) continue;
            sd += (rec.size() * 1.0 - avergaeSize) * (rec.size() * 1.0 - avergaeSize);
        }
        sd = sd / (unempty * 1.0);
        printf("Resize dataset on 1sd, mean: %.1lf\tsd: %.1Lf\n", avergaeSize, sd);
        int bound = ceil(avergaeSize + 1.0 * sd - 1e-5);
        for(auto &rec : dataset)
            if((int)rec.size() > bound)
                rec.resize(bound);
    } 

    // weighted sim funcs
    double weightedOverlap(ui x, ui y) {
        const auto &records1 = ifRS ? query_dataset[x] : work_dataset[x];
        const auto &records2 = work_dataset[y];

        std::vector<ui> res;
        set_intersection(records1.begin(), records1.end(), 
                         records2.begin(), records2.end(), 
                         std::back_inserter(res));

        double ovlp = 0.0;
        for(const auto &e : res)
            ovlp += wordwt[e];

        return ovlp;
    }

    double weightedJaccard(ui x, ui y) {
        double ovlp = weightedOverlap(x, y);
        double rw1 = ifRS ? query_weights[x] : work_weights[x];
        double rw2 = work_weights[y];

        assert(std::abs(rw1) > 1e-7 && std::abs(rw2) > 1e-7);

        return ovlp / (rw1 + rw2 - ovlp);
    }
    double jaccard(ui x, ui y) {
        const auto &records1 = ifRS ? query_dataset[x] : work_dataset[x];
        const auto &records2 = work_dataset[y];

        std::vector<ui> res;
        set_intersection(records1.begin(), records1.end(), 
                         records2.begin(), records2.end(), 
                         std::back_inserter(res));
        int ovlp = (int)res.size();
        
        return ovlp * 1.0 / (records1.size() + records2.size() - ovlp) * 1.0;
    }

    double weightedCosine(ui x, ui y) {
        double ovlp = weightedOverlap(x, y);
        double rw1 = ifRS ? query_weights[x] : work_weights[x];
        double rw2 = work_weights[y];

        assert(std::abs(rw1) > 1e-7 && std::abs(rw2) > 1e-7);

        return ovlp / sqrt(rw1 * rw2);
    }
    double cosine(ui x, ui y) {
        const auto &records1 = ifRS ? query_dataset[x] : work_dataset[x];
        const auto &records2 = work_dataset[y];

        std::vector<ui> res;
        set_intersection(records1.begin(), records1.end(), 
                         records2.begin(), records2.end(), 
                         std::back_inserter(res));
        int ovlp = (int)res.size();
        
        return ovlp * 1.0 / sqrt(records1.size() * records2.size()) * 1.0;
    }

    double weightedDice(ui x, ui y) {
        double ovlp = weightedOverlap(x, y);
        double rw1 = ifRS ? query_weights[x] : work_weights[x];
        double rw2 = work_weights[y];

        assert(std::abs(rw1) > 1e-7 && std::abs(rw2) > 1e-7);

        return 2.0 * ovlp / (rw1 + rw2);
    }
    double dice(ui x, ui y) {
        const auto &records1 = ifRS ? query_dataset[x] : work_dataset[x];
        const auto &records2 = work_dataset[y];

        std::vector<ui> res;
        set_intersection(records1.begin(), records1.end(), 
                         records2.begin(), records2.end(), 
                         std::back_inserter(res));
        int ovlp = (int)res.size();
        
        return ovlp * 2.0 / (records1.size() + records2.size()) * 1.0;
    }

	// check overlap between two sets
    bool overlap(int x, int y, int posx = 0, int posy = 0, int current_overlap = 0);
	bool overlapRS(int x, int y, int posx = 0, int posy = 0, int current_overlap = 0);
	// Self-join
    void setSelfJoin(double threshold, std::vector<std::pair<int, int>>& sim_pairs); 
	// RS-join
	void setRSJoin(double threshold, std::vector<std::pair<int, int>>& sim_pairs);
};



#endif