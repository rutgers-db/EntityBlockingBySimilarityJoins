/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _SETJOINPARELLED_H_
#define _SETJOINPARELLED_H_

#include "config.h"
#include "type.h"
#include "joinutil.h"
#include "index.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <queue>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>


class SetJoinParallel
{
public:
	bool ifRS = false;
    int earlyTerminated[MAXTHREADNUM] = { 0 };
    int earlyTerminatedEmpty[MAXTHREADNUM] = { 0 };

    // join func
	SimFuncType simFType{SimFuncType::JACCARD};
    std::string typeMap[3] = {"Jaccard", "Cosine", "Dice"};
    double (SetJoinParallel::*weightedFunc)(ui, ui) = nullptr;
    double (SetJoinParallel::*normalFunc)(ui, ui) = nullptr;
    bool (SetJoinParallel::*overlapFunc)(ui, ui, int, int, int) = nullptr;

    double det;
    uint64_t resultNum = 0;    // Number of result pairs found
    uint64_t candidateNum = 0; // Number of candidate pairs considered
    uint64_t listlens = 0;
    ui maxIndexPartNum{0}; // Maximum index partition number

    // Dataset containing records to be joined
    std::vector<std::vector<ui>> work_dataset;
	std::vector<std::vector<ui>> query_dataset;
    std::vector<double> work_weights;
    std::vector<double> query_weights;
    std::vector<double> wordwt;
    // Bucket for empty
    std::vector<ui> workEmpty;
    std::vector<ui> queryEmpty;
    // length
    std::vector<ui> workLength;

    // Parameters related to calculation and dataset
    double coe{0.0};
    double coePart{0.0};
    double ALPHA{0.0};
    ui work_n{0};       // Number of records in the dataset
	ui query_n{0};
    ui work_maxSize{0}, work_minSize{0}; // Maximum size of the records
	ui query_maxSize{0}, query_minSize{0};
    ui maxHeapSize{0};

    // Array to store result pairs for each thread
    std::vector<std::pair<int, int>> result_pairs[MAXTHREADNUM];
    std::vector<std::pair<int, int>> emptyPairs[MAXTHREADNUM];
#if MAINTAIN_VALUE == 1
    bool isWeightedComp{false}; // use the weighted version sim funcs
    std::vector<WeightPair> result_pairs_[MAXTHREADNUM];
    int isHeap[MAXTHREADNUM] = { 0 };
#endif

    // Recording time cost of different part
    double index_cost;
    double search_cost;
    double hashInFind_cost[MAXTHREADNUM];
    double mem_cost[MAXTHREADNUM];
    double find_cost[MAXTHREADNUM];
    double alloc_cost[MAXTHREADNUM];
    double verif_cost[MAXTHREADNUM];

private:
	// Index
	SetJoinParelledIndex invertedIndex;

    int *prime_exp;    // Array for storing prime numbers, presumably for hashing
    bool **quickRef2D; // 2D quick reference array
    bool **negRef2D;   // 2D negative reference array

    // Vectors for storing range information(the groups that based on the size of documents)
    std::vector<std::pair<ui, ui>> range;
    std::vector<ui> range_st;
    std::vector<int> range_id, rangeIdQuery, rangeQueryAdd;
    
	// the precalculated hashvalue key for the partitions and ondeletions
	std::vector<std::vector<ui>> parts_keys, partsKeysQuery;
	std::vector<std::vector<ui>> onedelete_keys, oneDeleteKeysQuery;
	std::vector<std::vector<ui>> odkeys_st, oDKeysStQuery; // Stores position of one deletion information

    // vectors needed when allocate by greedy heap method
    std::vector<ui> invPtrArr[MAXTHREADNUM];
    std::vector<ui> intPtrArr[MAXTHREADNUM];
    std::vector<std::vector<ui>> onePtrArr[MAXTHREADNUM];
    std::vector<std::pair<int, ui>> valuesArr[MAXTHREADNUM]; // <value, loc>
    std::vector<ui> scoresArr[MAXTHREADNUM];

public:
    // interchangeable value
    // A must be query and B must be work
    bool flagIC{false}; // indicate whether considering interchangeable value
    std::vector<int> grpIdA, grpIdB; // default to -1
    std::vector<std::vector<int>> groupA, groupB;
    std::vector<ui> revIdMapA, revIdMapB;
    std::vector<ui> idMapA, idMapB;
    // copy from "fast_group"
    double **featureValueCache{nullptr};
    int *discreteCacheIdx{nullptr};

public:
	// Self-join
    SetJoinParallel(const std::vector<std::vector<ui>> &sorted_records, const std::vector<double> &recwt, 
                    const std::vector<double> &_wordwt, double _det, ui _maxHeapSize = 0, 
                    bool _isWeightedComp = false) 
	: work_dataset(sorted_records), work_weights(recwt), wordwt(_wordwt), work_n(work_dataset.size()), 
      work_maxSize(work_dataset.back().size()), work_minSize(work_dataset.front().size()) { 
#if RESIZE_DATA == 1
        if(_det <= 0.4 && work_maxSize >= 55) {
            resizeData(work_dataset);
            work_maxSize = work_dataset.back().size();
        }
#endif
        printf("Work size: %u\n", work_n);
		printf("Min record's size: %u\tMax record's size: %u\n", work_minSize, work_maxSize);

        maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;
#if MAINTAIN_VALUE == 1
        isWeightedComp = _isWeightedComp;
        for(int tid = 0; tid < MAXTHREADNUM; tid++)
            result_pairs_[tid].reserve(maxHeapSize);
#endif
    }

	// RS-join
	SetJoinParallel(const std::vector<std::vector<ui>> &work_records, const std::vector<std::vector<ui>> &query_records, 
                    const std::vector<double> &workwt, const std::vector<double> &querywt, 
                    const std::vector<double> &_wordwt, double _det, ui _maxHeapSize = 0, 
                    bool  _isWeightedComp = false) 
	: ifRS(true), work_dataset(work_records), query_dataset(query_records), 
    work_weights(workwt), query_weights(querywt), wordwt(_wordwt),
    work_n(work_dataset.size()), query_n(query_dataset.size()), 
    work_maxSize(work_dataset.back().size()), work_minSize(work_dataset.front().size()),
    query_maxSize(query_dataset.back().size()), query_minSize(query_dataset.front().size()) { 
#if RESIZE_DATA == 1
        if(_det <= 0.4 && work_maxSize >= 55) {
            resizeData(work_dataset);
            work_maxSize = work_dataset.back().size();
        }
        if(_det <= 0.4 && query_maxSize >= 55) {
            resizeData(query_dataset);
            query_maxSize = query_dataset.back().size();
        }
#endif
        printf("Work size: %u\tQuery size: %u\n", work_n, query_n);
		printf("Min work record's size: %u\tMax work record's size: %u\n", work_minSize, work_maxSize);
		printf("Min query record's size: %u\tMax query record's size: %u\n", query_minSize, query_maxSize);

        maxHeapSize = _maxHeapSize == 0 ? MAX_PAIR_SIZE : _maxHeapSize;
#if MAINTAIN_VALUE == 1
        isWeightedComp = _isWeightedComp;
        for(int tid = 0; tid < MAXTHREADNUM; tid++)
            result_pairs_[tid].reserve(maxHeapSize);
#endif
    }

	// index's memory released in functions
    ~SetJoinParallel() = default;

public:
    // Output the Parameters
    void showPara() const {
        std::string st = "";
        switch(simFType) {
            case SimFuncType::JACCARD : st = "jaccard"; break;
            case SimFuncType::COSINE : st = "cosine"; break;
            case SimFuncType::DICE : st = "dice"; break;
        }
        printf("type: %s det: %.4lf coe: %.8lf ALPHA: %.8lf maxIndexPartNum: %u \n", st.c_str(), det, coePart, ALPHA,
               maxIndexPartNum);
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

    void reportTimeCost() {
        double total_hash_cost = 0;
        double total_memeory_cost = 0;
        double total_find_cost = 0;
        double total_alloc_cost = 0;
        double total_verif_cost = 0;
        double sum;
        for (int i = 0; i < MAXTHREADNUM; i++) {
            total_hash_cost += hashInFind_cost[i];
            total_memeory_cost += mem_cost[i];
            total_find_cost += find_cost[i];
            total_alloc_cost += alloc_cost[i];
            total_verif_cost += verif_cost[i];
        }
        sum = total_hash_cost + total_memeory_cost + total_find_cost + total_alloc_cost + total_verif_cost;
        total_hash_cost = total_hash_cost / sum * search_cost;
        total_memeory_cost = total_memeory_cost / sum * search_cost;
        total_find_cost = total_find_cost / sum * search_cost;
        total_alloc_cost = total_alloc_cost / sum * search_cost;
        total_verif_cost = total_verif_cost / sum * search_cost;

        printf("|index_cost| total_hash_cost| total_memeory_cost| find_cost| alloc_cost| verif_cost|\n");
        printf("|%f|%f|%f|%f|%f|%f|\n", index_cost, total_hash_cost, total_memeory_cost, total_find_cost, total_alloc_cost, total_verif_cost);
    }

    void reportLargestGroup() {
        std::vector<unsigned long long> range_size(range.size());
        for (ui i = 0; i < range_id.size(); i++) {
            range_size[range_id[i]] += work_dataset[i].size();
        }

        double total_size = 0;
        double max_size = 0;
        for (auto &size : range_size) {
            total_size += size;
            max_size = std::max(max_size, (double)size);
        }

        printf("Average Range size: %.3f Maximum Range size ratio %.3f \n", total_size / range_size.size(), max_size / total_size);
    }
    
    // Function to get the total number of result pairs found by all threads
    unsigned long long getResultPairsAmount() {
        unsigned long long pairs_amount = 0;
        for (int i = 0; i < MAXTHREADNUM; i++) {
            pairs_amount += result_pairs[i].size();
        }
        return pairs_amount;
    }

    void mergeResults(std::vector<std::pair<int, int>> &finalPairs) {
#if APPEND_EMPTY == 1
        if(!ifRS) {
#pragma omp parallel for
            for(ui i = 0; i < workEmpty.size(); i++) {
                int tid = omp_get_thread_num();
                if(earlyTerminatedEmpty[tid] == 1)
                    continue;

                for(ui j = i + 1; j < workEmpty.size(); j++)
                    emptyPairs[tid].emplace_back(i, j);
                if(emptyPairs[tid].size() > MAX_EMPTY_SIZE)
                    earlyTerminatedEmpty[tid] = 1;
            }
        }
        else {
#pragma omp parallel for
            for(ui j = 0; j < workEmpty.size(); j++) {
                int tid = omp_get_thread_num();
                if(earlyTerminatedEmpty[tid] == 1)
                    continue;

                for(ui i = 0; i < queryEmpty.size(); i++)
                    emptyPairs[tid].emplace_back(i, j);
                if(emptyPairs[tid].size() > MAX_EMPTY_SIZE)
                    earlyTerminatedEmpty[tid] = 1;
            }
        }
#endif

        std::cout << "Start merging" << std::endl << std::flush;
#if MAINTAIN_VALUE == 0
        for(int i = 0; i < MAXTHREADNUM; i++) {
            finalPairs.insert(finalPairs.end(), emptyPairs[i].begin(), emptyPairs[i].end());
            finalPairs.insert(finalPairs.end(), result_pairs[i].begin(), result_pairs[i].end());
        }
#elif MAINTAIN_VALUE == 1
        for(int i = 0; i < MAXTHREADNUM; i++) {
            finalPairs.insert(finalPairs.end(), emptyPairs[i].begin(), emptyPairs[i].end());
            // finalPairs.insert(finalPairs.end(), result_pairs_[i].begin(), result_pairs_[i].end());
            for(const auto &wp : result_pairs_[i])
                finalPairs.emplace_back(wp.id1, wp.id2);
        }
#endif


#if DEDUPLICATE == 1
        sort(finalPairs.begin(), finalPairs.end());
        auto it = unique(finalPairs.begin(), finalPairs.end());
        if(it != finalPairs.end()) {
            std::cerr << "Duplicate results: " << distance(it, finalPairs.end()) << std::endl;
            exit(1);
        }
#endif
        std::cout << workEmpty.size() << std::endl << std::flush;
    }

public:
    // sim funcs
    bool overlapSelf(ui x, ui y, int posx = 0, int posy = 0, int current_overlap = 0) {
        // Calculate required overlap based on a formula
        int require_overlap = 0;
        switch(simFType) {
            case SimFuncType::JACCARD : {
                require_overlap = ceil(det / (1 + det) * (int)(work_dataset[x].size() + work_dataset[y].size()) - EPS);
                break;
            }
            case SimFuncType::COSINE : {
                require_overlap = ceil(1.0 * det * sqrt(work_dataset[x].size() * work_dataset[y].size()) - EPS);
                break;
            }
            case SimFuncType::DICE : {
                require_overlap = ceil(0.5 * det * (int)(work_dataset[x].size() + work_dataset[y].size()) - EPS);
                break;
            }
        }

        // Loop through both sets to find overlap
        while (posx < (int)work_dataset[x].size() && posy < (int)work_dataset[y].size()) {
            // Check if remaining elements are sufficient for required overlap
            if ((int)work_dataset[x].size() - posx + current_overlap < require_overlap || (int)work_dataset[y].size() - posy + current_overlap < require_overlap) 
                return false;

            if (work_dataset[x][posx] == work_dataset[y][posy]) {
                current_overlap++;
                posx++;
                posy++;
            } else if (work_dataset[x][posx] < work_dataset[y][posy]) {
                posx++;
            } else {
                posy++;
            }
        }

        return current_overlap >= require_overlap;
    }

    bool overlapSelfIC(ui x, ui y, int posx = 0, int posy = 0, int current_overlap = 0) {
        ui revIdx = idMapA[x];
        ui revIdy = idMapA[y];
        int grpIdX = grpIdA[revIdx];
        int grpIdY = grpIdA[revIdy];

        if(grpIdX == -1 && grpIdY == -1) 
            return overlapSelf(x, y, posx, posy, current_overlap);
        else if(grpIdX != -1 && grpIdY == -1) {
            for(const auto &icid : groupA[grpIdX]) {
                bool success = overlapSelf(revIdMapA[icid], y, posx, posy, current_overlap);
                if(success)
                    return true;
            }
        }
        else if(grpIdX == -1 && grpIdY != -1) {
            for(const auto &icid: groupA[grpIdY]) {
                bool success = overlapSelf(x, revIdMapA[icid], posx, posy, current_overlap);
                if(success)
                    return true;
            }
        }
        else {
            int dcIdxX = discreteCacheIdx[grpIdX];
            int dcIdxY = discreteCacheIdx[grpIdY];
            double val = featureValueCache[dcIdxX][dcIdxY];
            return val >= det;
        }

        return false;
    }

	bool overlapRS(ui x, ui y, int posx = 0, int posy = 0, int current_overlap = 0) {
        // Calculate required overlap based on a formula
        int require_overlap = 0;
        switch(simFType) {
            case SimFuncType::JACCARD : {
                require_overlap = ceil(det / (1 + det) * (int)(query_dataset[x].size() + work_dataset[y].size()) - EPS);
                break;
            }
            case SimFuncType::COSINE : {
                require_overlap = ceil(1.0 * det * sqrt(query_dataset[x].size() * work_dataset[y].size()) - EPS);
                break;
            }
            case SimFuncType::DICE : {
                require_overlap = ceil(0.5 * det * (int)(query_dataset[x].size() + work_dataset[y].size()) - EPS);
                break;
            }
        }

        // Loop through both sets to find overlap
        while (posx < (int)query_dataset[x].size() && posy < (int)work_dataset[y].size()) {
            // Check if remaining elements are sufficient for required overlap
            if ((int)query_dataset[x].size() - posx + current_overlap < require_overlap || (int)work_dataset[y].size() - posy + current_overlap < require_overlap) 
                return false;

            if (query_dataset[x][posx] == work_dataset[y][posy]) {
                current_overlap++;
                posx++;
                posy++;
            } else if (query_dataset[x][posx] < work_dataset[y][posy]) {
                posx++;
            } else {
                posy++;
            }
        }
        return current_overlap >= require_overlap;
    }

    bool overlapRSIC(ui x, ui y, int posx = 0, int posy = 0, int current_overlap = 0) {
        ui revIdx = idMapA[x];
        ui revIdy = idMapB[y];
        int grpIdX = grpIdA[revIdx];
        int grpIdY = grpIdB[revIdy];

        if(grpIdX == -1 && grpIdY == -1) 
            return overlapRS(x, y, posx, posy, current_overlap);
        else if(grpIdX != -1 && grpIdY == -1) {
            for(const auto &icid : groupA[grpIdX]) {
                bool success = overlapRS(revIdMapA[icid], y, posx, posy, current_overlap);
                if(success)
                    return true;
            }
        }
        else if(grpIdX == -1 && grpIdY != -1) {
            for(const auto &icid: groupB[grpIdY]) {
                bool success = overlapRS(x, revIdMapB[icid], posx, posy, current_overlap);
                if(success)
                    return true;
            }
        }
        else {
            int dcIdxX = discreteCacheIdx[grpIdX];
            int dcIdxY = discreteCacheIdx[grpIdY];
            double val = featureValueCache[dcIdxX][dcIdxY];
            return val >= det;
        }

        return false;
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

public:
    // join steps
    // Function to build index
    void index(double threshold);

    // Function to find candidate and similar pairs using a greedy approach
    void GreedyFindCandidateAndSimPairs(const int &tid, const int indexLenGrp, const ui rid, 
										ui record_length, const std::vector<ui> &p_keys, 
										const std::vector<ui> &od_keys, const std::vector<ui> &odk_st);

    // Function to find similar pairs
    void findSimPairsSelf();
	void findSimPairsRS();
};


#endif