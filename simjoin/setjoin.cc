/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/setjoin.h"


// To have Jaccard(x, y) > \delta
// Over(x, y) > \frac{\delta}{1 + \delta} * (|X| + |Y|)
bool SetJoin::overlap(int x, int y, int posx, int posy, int current_overlap) 
{
	// \frac{\delta}{1 + \delta} * (|X| + |Y|)
    int require_overlap = 0;
    switch(simFType) {
        case SimFuncType::JACCARD : 
            require_overlap = ceil(det / (1 + det) * (int)(work_dataset[x].size() + work_dataset[y].size()) - EPS);
            break;
        case SimFuncType::COSINE :
            require_overlap = ceil(1.0 * det * sqrt(work_dataset[x].size() * work_dataset[y].size()) - EPS);
            break;
        case SimFuncType::DICE : 
            require_overlap = ceil(0.5 * det * (int)(work_dataset[x].size() + work_dataset[y].size()) - EPS);
            break;
    }

    while (posx < (int)work_dataset[x].size() && posy < (int)work_dataset[y].size()) {

        if ((int)work_dataset[x].size() - posx + current_overlap < require_overlap || 
                (int)work_dataset[y].size() - posy + current_overlap < require_overlap) return false; 

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


bool SetJoin::overlapRS(int x, int y, int posx, int posy, int current_overlap) 
{
	// \frac{\delta}{1 + \delta} * (|X| + |Y|)
    int require_overlap = 0;
    switch(simFType) {
        case SimFuncType::JACCARD : 
            require_overlap = ceil(det / (1 + det) * (int)(query_dataset[x].size() + work_dataset[y].size()) - EPS);
            break;
        case SimFuncType::COSINE : 
            require_overlap = ceil(1.0 * det * sqrt(query_dataset[x].size() * work_dataset[y].size()) - EPS);
            break;
        case SimFuncType::DICE : 
            require_overlap = ceil(0.5 * det * (int)(query_dataset[x].size() + work_dataset[y].size()) - EPS);
            break;
    }

    while (posx < (int)query_dataset[x].size() && posy < (int)work_dataset[y].size()) {

        if ((int)query_dataset[x].size() - posx + current_overlap < require_overlap || 
                (int)work_dataset[y].size() - posy + current_overlap < require_overlap) 
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



// Self-join
void SetJoin::setSelfJoin(double threshold, std::vector<std::pair<int, int>> &sim_pairs) 
{
#if MAINTAIN_VALUE == 1
    if(simFType == SimFuncType::JACCARD) {
        weightedFunc = &SetJoin::weightedJaccard;
        normalFunc = &SetJoin::jaccard;
    }
    else if(simFType == SimFuncType::COSINE) {
        weightedFunc = &SetJoin::weightedCosine;
        normalFunc = &SetJoin::cosine;
    }
    else if(simFType == SimFuncType::DICE) {
        weightedFunc = &SetJoin::weightedDice;
        normalFunc = &SetJoin::dice;
    }
#endif

    SetJoinUtil::printMemory();
    timeval allstart, allend;
    gettimeofday(&allstart, NULL);

	// wont overflow?
    prime_exp[0] = 1;
    for (int i = 1; i < MAX_LINE_LENGTH; ++i)
        prime_exp[i] = prime_exp[i - 1] * PRIME;

    std::string str;
    
    // for (auto &vec : dataset) sort(vec.begin(), vec.end());

    det = threshold;
    double coe = (1 - det) / (1 + det);
	double coePart = (1 - det) / det;
    if(simFType == SimFuncType::COSINE) 
        coePart = (1 - det * det) / (det * det);
    else if(simFType == SimFuncType::DICE)
        coePart = 2 * (1 - det) / det;

    double ALPHA = 1.0; // shouldn't it's in [0.5, 1]?
    // double maxAlpha = 1.0 / threshold + 0.01;
    // if(ALPHA > maxAlpha)
    //     ALPHA = maxAlpha;
    if(det < 0.9)
        ALPHA = 1.8;
    
    // bool print_result = false;
    
    int n = work_dataset.size();

    int maxSize = work_dataset.back().size();

    int tokenNum = 0;
    for (ui i = 0; i < work_dataset.size(); i++) {
        if(work_dataset[i].empty())
            continue;
        if (tokenNum < (int)work_dataset[i].back()) 
            tokenNum = work_dataset[i].back();
    }
    tokenNum += 1;

    // the universe U: 1st dimension: length + part, 2nd dimension, hashvalue
    std::unordered_map<int64_t, invIndexStruct> invIndex;
    std::unordered_map<int64_t, unsigned long long> oneIndex;
    // invIndex.set_empty_key(-1);
    // oneIndex.set_empty_key(-1);
    int invIndexSize = (1 - det) / det * lengthSum, oneIndexSize =  lengthSum;
    // invIndex.resize(invIndexSize);
    // oneIndex.resize(oneIndexSize);
    indexLists.resize(1);
    indexVecs.resize(1);
    indexLists.reserve(oneIndexSize + invIndexSize);
    indexVecs.reserve(n);

    int partNum = 0;  // partition num
    // int prevLen = 0;
    bool * quickRef = new bool[n];
    memset(quickRef, 0, sizeof(bool) * n);
    bool * negRef = new bool[n];
    memset(negRef, 0, sizeof(bool) * n);

    std::vector<int> candidates;

#if PART_COE == 0
    int maxIndexPartNum = floor(2 * coe * maxSize + EPS) + 1; 
#elif PART_COE == 1
	int maxIndexPartNum = floor(coePart * maxSize + EPS) + 1;
#endif
    std::vector<int> hashValues;
    std::vector<int> positions;
    std::vector<std::vector<int>> subquery;  // first part id second tokens
    std::vector<std::vector<int>> oneHashValues;
    std::vector<unsigned long long> invPtr;
    std::vector<unsigned long long> intPtr;
    std::vector<std::vector<unsigned long long>> onePtr;
    std::vector<std::pair<int, int>> values;	// <value, loc>
    std::vector<int> scores;

    subquery.resize(maxIndexPartNum);
    hashValues.resize(maxIndexPartNum);
    positions.resize(maxIndexPartNum);
    oneHashValues.resize(maxIndexPartNum);
    invPtr.resize(maxIndexPartNum);
    intPtr.resize(maxIndexPartNum);
    onePtr.resize(maxIndexPartNum);
    values.resize(maxIndexPartNum);
    scores.resize(maxIndexPartNum);

    int low = 0, high = 0;
	std::vector<std::pair<int, int>> range;

    for (int rid = 0; rid < n; rid++) {
        int len = work_dataset[rid].size();
        if(len==0) {
            workEmpty.emplace_back(rid);
            continue;
        }

        int indexPartNum;
        int prevIndexPartNum = 0; 
        int pos = 0;

        int indexLenLow = 0; // [\delta * s, s]
        switch(simFType) {
            case SimFuncType::JACCARD : indexLenLow = ceil(det * len - EPS); break;
            case SimFuncType::COSINE : indexLenLow = ceil(det * det * len - EPS); break;
            case SimFuncType::DICE : indexLenLow = ceil(det / (2 - det) * len - EPS); break;
        }

        for (int indexLenGrp = 0; indexLenGrp < (int)range.size(); ++indexLenGrp) {
            if (range[indexLenGrp].second < indexLenLow) 
				continue;

            int indexLen = range[indexLenGrp].first;
#if PART_COE == 0
            indexPartNum = floor(2 * coe * indexLen + EPS) + 1;
#elif PART_COE == 1
			indexPartNum = floor(coePart * indexLen + EPS) + 1;
#endif
            // split the query into multiple parts if prevIndexPartNum != indexPartNum
            // it means if the indexLenGrp's first range is not change
            // But does it possible?
            if (prevIndexPartNum != indexPartNum) {

                // clear subquery oneHashValues  hashValues
                for (int pid = 0; pid < indexPartNum; ++pid) {
                    subquery[pid].clear();
                    oneHashValues[pid].clear();
                    hashValues[pid] = 0;
                }
                
                // allocate the tokens into subquery and hash each part
                // That is the way how the query is splitted into indexPartNum
                for (auto &token : work_dataset[rid]) {
                    int pid = token % indexPartNum;
                    subquery[pid].push_back(token);
                    hashValues[pid] = PRIME * hashValues[pid] + token + 1;
                }

                // Dont know the position meaning?
                pos = 0;
                for (int pid = 0; pid < indexPartNum; pid++) {
                    positions[pid] = pos;
                    pos = pos + subquery[pid].size();
                }

                prevIndexPartNum = indexPartNum;
            }

            // Initialize onePtr and reserve each space
            for (int pid = 0; pid < indexPartNum; ++pid) {
                onePtr[pid].clear();
                onePtr[pid].reserve(subquery[pid].size());
            }

            auto alloc_st = SetJoinUtil::logTime();
            // Iterate each part find each part if there is identical part already exits in inverted list
            // Based on the above result, initialize the value and scores
            for (int pid = 0; pid < indexPartNum; pid++) {
                
                // A hash 
                int64_t lenPart = indexLen + pid * (maxSize + 1);

                int v1 = 0;
                auto invit = invIndex.find(PACK(lenPart, hashValues[pid]));
                if (invit != invIndex.end()) {
                    invPtr[pid] = invit->second.list_no;
                    v1 = -indexLists[invit->second.list_no].cnt;
                } else {
                    invPtr[pid] = 0;
                }
                values[pid] = std::make_pair(v1, pid);
                scores[pid] = 0;
            }
            int heap_cnt = indexPartNum;
            make_heap(values.begin(), values.begin() + heap_cnt);


            int cost = 0;
            int rLen = std::min(range[indexLenGrp].second, len);
            int Ha = floor((len - det * rLen) / (1 + det) + EPS);
            int Hb = floor((rLen - det * len) / (1 + det) + EPS);
            int maxH = Ha + Hb;
            switch(simFType) {
                case SimFuncType::JACCARD : maxH = floor(coe * (len + rLen) + EPS); break;
                case SimFuncType::COSINE : maxH = floor(len + rLen - 2 * det * sqrt(len * rLen) + EPS); break;
                case SimFuncType::DICE : maxH = floor((1 - det) * (len + rLen) + EPS); break;
            }
    
            // We need use greedy selection in maxH + 1 times
            for (int i = 0; i < maxH + 1; ++i) {
                auto sel = values.front();
                pop_heap(values.begin(), values.begin() + heap_cnt);
                int pid = sel.second;
                ++scores[pid];
                cost -= sel.first;
                int64_t lenPart = indexLen + pid * (maxSize + 1);

                if (scores[sel.second] == 1) {
                    if (invPtr[pid] != 0) {
                        auto &vec = indexLists[invPtr[pid]].getVector(this);
                        for (auto lit = vec.begin(); lit != vec.end(); lit++) {
                            int rLen = work_dataset[lit->first].size();
                            
                            int H = 0, Ha = 0, Hb = 0;
                            switch(simFType) {
                                case SimFuncType::JACCARD : 
                                    Ha = floor((len - det * rLen) / (1 + det) + EPS);
                                    Hb = floor((rLen - det * len) / (1 + det) + EPS);
                                    H = Ha + Hb;
                                    break;
                                case SimFuncType::COSINE : 
                                    Ha = floor(len - det * sqrt(len * rLen) + EPS);
                                    Hb = floor(rLen - det * sqrt(len * rLen) + EPS);
                                    H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                                    break;
                                case SimFuncType::DICE : 
                                    Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                                    Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                                    H = floor((1.0 - det) * (len + rLen) + EPS);
                                    break;
                            }

                            // If the current iteration index i is greater than this maximum allowable difference H
                            // current entry in the inverted list is skipped,
                            if (i > H) continue;
                            // position filter
                            if (negRef[lit->first] == false && quickRef[lit->first] == false)
                                candidates.push_back(lit->first);
                            if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
                                negRef[lit->first] = true;
                            else
                                quickRef[lit->first] = true;
                        }
                    }

                    // maintain heap
                    // Here is to find the next pair that insert to the heap
                    // Basically it is to consider the situation of the 1-deletion of the current part
                    int v2 = 0;

                    // search if the 1-deletion 
                    auto oneit = oneIndex.find(PACK(lenPart, hashValues[pid]));
                    if (oneit != oneIndex.end()) {
                        intPtr[pid] = oneit->second;
                        v2 -= indexLists[oneit->second].cnt;
                    } else {
                        intPtr[pid] = 0;
                    }


                    if (oneHashValues[pid].size() == 0) {
                        int mhv = 0, hv = hashValues[pid];
                        auto &sq = subquery[pid];
                        for (ui idx = 0; idx < sq.size(); idx++) {
                            int chv = hv + mhv * prime_exp[sq.size() - 1 - idx];
                            mhv = mhv * PRIME + sq[idx] + 1;
                            chv -= mhv * prime_exp[sq.size() - 1 - idx];
                            oneHashValues[pid].push_back(chv);
                        }
                    }
                    for (int id = 0; id < (int)oneHashValues[pid].size(); ++id) {
                        auto invit = invIndex.find(PACK(lenPart, oneHashValues[pid][id]));
                        if (invit != invIndex.end()) {
                            onePtr[pid].push_back(invit->second.list_no);
                            v2 -= indexLists[invit->second.list_no].cnt;
                        } else {
                            onePtr[pid].push_back(0);
                        }
                    }

                    values[heap_cnt - 1].first = v2;
                    push_heap(values.begin(), values.begin() + heap_cnt);
                } 
				else {
                    auto ov_st = SetJoinUtil::logTime();
                    // add candidates
                    if (intPtr[pid] != 0) {
                        auto &vec = indexLists[intPtr[pid]].getVector(this);
                        for (auto lit = vec.begin(); lit != vec.end(); lit++) {
                            int rLen = work_dataset[lit->first].size();
                            
                            int H = 0, Ha = 0, Hb = 0;
                            switch(simFType) {
                                case SimFuncType::JACCARD : 
                                    Ha = floor((len - det * rLen) / (1 + det) + EPS);
                                    Hb = floor((rLen - det * len) / (1 + det) + EPS);
                                    H = Ha + Hb;
                                    break;
                                case SimFuncType::COSINE : 
                                    Ha = floor(len - det * sqrt(len * rLen) + EPS);
                                    Hb = floor(rLen - det * sqrt(len * rLen) + EPS);
                                    H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                                    break;
                                case SimFuncType::DICE : 
                                    Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                                    Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                                    H = floor((1.0 - det) * (len + rLen) + EPS);
                                    break;
                            }

                            if (i > H) continue;
                            if (negRef[lit->first] == false && quickRef[lit->first] == false)
                                candidates.push_back(lit->first);
                            if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
                                negRef[lit->first] = true;
                            else
                                quickRef[lit->first] = true;
                        }
                    }

                    for (int id = 0; id < (int)onePtr[pid].size(); ++id) {
                        if (onePtr[pid][id] != 0) {
                            auto &vec = indexLists[onePtr[pid][id]].getVector(this);
                            for (auto lit = vec.begin(); lit != vec.end(); lit++) {
                                int rLen = work_dataset[lit->first].size();

                                int H = 0, Ha = 0, Hb = 0;
                                switch(simFType) {
                                    case SimFuncType::JACCARD : 
                                        Ha = floor((len - det * rLen) / (1 + det) + EPS);
                                        Hb = floor((rLen - det * len) / (1 + det) + EPS);
                                        H = Ha + Hb;
                                        break;
                                    case SimFuncType::COSINE : 
                                        Ha = floor(len - det * sqrt(len * rLen) + EPS);
                                        Hb = floor(rLen - det * sqrt(len * rLen) + EPS);
                                        H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                                        break;
                                    case SimFuncType::DICE : 
                                        Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                                        Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                                        H = floor((1.0 - det) * (len + rLen) + EPS);
                                        break;
                                }

                                if (i > H) continue;
                                if (negRef[lit->first] == false && quickRef[lit->first] == false)
                                    candidates.push_back(lit->first);
                                if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
                                    negRef[lit->first] = true;
                                else
                                    quickRef[lit->first] = true;
                            }
                        }
                    }
                    overlap_cost += SetJoinUtil::repTime(ov_st);
                    // maintain heap
                    --heap_cnt;
                }
            }
            allocation_cost += SetJoinUtil::repTime(alloc_st);
            listlens += cost;

            // clear candidates
            for (ui idx = 0; idx < candidates.size(); idx++) {
                if (negRef[candidates[idx]] == false && quickRef[candidates[idx]] == true) {
                    if (overlap(candidates[idx], rid) == true) {
                        assert(rid != candidates[idx]);
#if MAINTAIN_VALUE == 0
                        if(rid < candidates[idx])
                            sim_pairs.emplace_back(rid, candidates[idx]);
                        else
                            sim_pairs.emplace_back(candidates[idx], rid);
                        
                        if(sim_pairs.size() > maxHeapSize)
                            return;
#elif MAINTAIN_VALUE == 1
                        double val = isWeightedComp ? (this->*weightedFunc)(rid, candidates[idx]) 
                                                    : (this->*normalFunc)(rid, candidates[idx]);
                        if(result_pairs_.size() < maxHeapSize) {
                            if(rid < candidates[idx]) result_pairs_.emplace_back(rid, candidates[idx], val);
                            else result_pairs_.emplace_back(candidates[idx], rid, val);
                        }
                        else {
                            if(isHeap == 0) {
                                std::make_heap(result_pairs_.begin(), result_pairs_.end());
                                isHeap = 1;
                            }
                            if(result_pairs_[0].val < val) {
                                std::pop_heap(result_pairs_.begin(), result_pairs_.end());
                                result_pairs_.pop_back();
                                if(rid < candidates[idx]) result_pairs_.emplace_back(rid, candidates[idx], val);
                                else result_pairs_.emplace_back(candidates[idx], rid, val);
                                std::push_heap(result_pairs_.begin(), result_pairs_.end());
                            }
                        }
#endif
                        resultNum++;
                        // std::pair<int, int> sim_pair = std::make_pair(rid, candidates[idx]);
						// sim_pairs.emplace_back(rid, candidates[idx]);
                        // result_pairs.emplace_back(rid, candidates[idx]);
                    }
                    candidateNum++;
                }
                quickRef[candidates[idx]] = false;
                negRef[candidates[idx]] = false;
            }
            candidates.clear();
        }

        // indexing
        auto index_st = SetJoinUtil::logTime();
        // Here is to create a new group 
		
        if (len > high) {
            low = len;
            high = len * ALPHA;
#if PART_COE == 0
            partNum = floor(2 * coe * low + EPS) + 1;
#elif PART_COE == 1
			partNum = floor(coePart * low + EPS) + 1;
#endif
            range.emplace_back(low, high);
        } 

		hashValues.clear();
		oneHashValues.clear();
		subquery.clear();
		hashValues.resize(partNum);
		oneHashValues.resize(partNum);
		subquery.resize(partNum);
		for (auto &token : work_dataset[rid]) {
			int pid = token % partNum;
			auto &subrec = subquery[pid];
			subrec.push_back(token);
			hashValues[pid] = PRIME * hashValues[pid] + token + 1;
		}

        for (int pid = 0; pid < partNum; ++pid) {
            if (oneHashValues[pid].size() == 0) {
                int mhv = 0, hv = hashValues[pid];
                auto &subrec = subquery[pid];
                for (ui idx = 0; idx < subrec.size(); idx++) {
                    int chv = hv + mhv * prime_exp[subrec.size() - 1 - idx];
                    mhv = mhv * PRIME + subrec[idx] + 1;
                    chv -= mhv * prime_exp[subrec.size() - 1 - idx];
                    oneHashValues[pid].push_back(chv);
                }
            }
        }

        pos = 0;
        for (int pid = 0; pid < partNum; pid++) {
            int hv = hashValues[pid];
            auto &subrec = subquery[pid];
            int64_t lenPart = low + pid * (maxSize + 1);
            auto &inv = invIndex[PACK(lenPart, hv)];
            if (inv.list_no == 0) {
                indexLists.push_back(invertedList());
                inv.list_no = indexLists.size() - 1;
            }
            indexLists[inv.list_no].add(std::make_pair(rid, pos), this);
            // build one inverted index
            for (auto &chv : oneHashValues[pid]) {
                auto &one = oneIndex[PACK(lenPart, chv)];
                if (one == 0) {
                    indexLists.push_back(invertedList());
                    one = indexLists.size() - 1;
                }
                indexLists[one].add(std::make_pair(rid, pos), this);
            }

            pos = pos + subrec.size();
        }
		
        index_cost += SetJoinUtil::repTime(index_st);
    }

    // empty tokenized set
#if APPEND_EMPTY == 1
    ui totalEmpty = workEmpty.size();
    for(ui i = 0; i < totalEmpty; i++)
        for(ui j = i + 1; j < totalEmpty; j++)
            sim_pairs.emplace_back(j, i);
#endif

#if MAINTAIN_VALUE == 1
    for(const auto &p : result_pairs_)
        sim_pairs.emplace_back(p.id1, p.id2);
#endif

    gettimeofday(&allend, NULL);
    double all = allend.tv_sec - allstart.tv_sec + (allend.tv_usec - allstart.tv_usec) / 1e6;

    // print
    int v[6] = { 0 };
    for (auto &vec : indexVecs) {
        if (vec.size() < 6) ++v[vec.size()];
    }
    //fprintf(stderr, "vec len: ");
    //for (int i = 1; i < 6; ++i) fprintf(stderr, "(%d: %d) ", i, v[i]);
    fprintf(stderr, "total time: %.3fs\n", allend.tv_sec - allstart.tv_sec + (allend.tv_usec - allstart.tv_usec) / 1e6);
    fprintf(stderr, "%lu %lu %lu %.3f\n", resultNum, candidateNum, listlens, all);
    fprintf(stderr, "AllocCost %.3f  IndexCost: %.3f OverlapCost : %.3f\n",allocation_cost, index_cost, overlap_cost);

    unsigned long long pairs_amount = 0;
    for(auto const & vec : indexVecs){
        pairs_amount+=vec.size();
    }
    fprintf(stderr, "The amount of pairs in the indexVecs is: %llu\n", pairs_amount);
    SetJoinUtil::printMemory();

#ifdef WRITE_RESULT
    simp_ofs.close();
#endif

    delete[] quickRef;
    delete[] negRef;
} 


// RS-join
void SetJoin::setRSJoin(double threshold, std::vector<std::pair<int, int>> &sim_pairs)
{
#if MAINTAIN_VALUE == 1
    if(simFType == SimFuncType::JACCARD) {
        weightedFunc = &SetJoin::weightedJaccard;
        normalFunc = &SetJoin::jaccard;
    }
    else if(simFType == SimFuncType::COSINE) {
        weightedFunc = &SetJoin::weightedCosine;
        normalFunc = &SetJoin::cosine;
    }
    else if(simFType == SimFuncType::DICE) {
        weightedFunc = &SetJoin::weightedDice;
        normalFunc = &SetJoin::dice;
    }
#endif

	SetJoinUtil::printMemory();
    timeval allstart, allend;
    gettimeofday(&allstart, NULL);

	// overflow here
    prime_exp[0] = 1;
    for (int i = 1; i < MAX_LINE_LENGTH; ++i)
        prime_exp[i] = prime_exp[i - 1] * PRIME;

    std::string str;

    det = threshold;
    double coe = (1 - det) / (1 + det);
    double coePart = 0.0;
    switch(simFType) {
        case SimFuncType::JACCARD : coePart = (1 - det) / det; break;
        case SimFuncType::COSINE : coePart = (1.0 - det * det) / (det * det); break;
        case SimFuncType::DICE : coePart = 2.0 * (1.0 - det) / det; break;
    }
    // printf("%.4lf\n", coePart);

    double ALPHA = 1.0; 
    // double maxAlpha = 1.0 / threshold + 0.01;
    // ALPHA = 1.8;
    // if(ALPHA > maxAlpha)
    //     ALPHA = maxAlpha;
     if(det < 0.7)
        ALPHA = 1.8;
    
    // bool print_result = false;
    
    int n = work_dataset.size();
	int qn = query_dataset.size();

    int work_maxSize = work_dataset.back().size();
    int work_minSize = work_dataset[0].size();
	// int query_maxSize = query_dataset.back().size();

    int tokenNum = 0;
    for (ui i = 0; i < work_dataset.size(); i++) {
		if(work_dataset[i].empty())
			continue;
        if(work_dataset[i].size() >= MAX_LINE_LENGTH) {
            std::cerr << "Line length overflow: " << work_dataset[i].size() << std::endl;
            exit(1);
        }

        if (tokenNum < (int)work_dataset[i].back()) 
            tokenNum = work_dataset[i].back();
	}
    tokenNum += 1;

    // the universe U: 1st dimension: length + part, 2nd dimension, hashvalue
	int invIndexSize = (1 - det) / det * lengthSum, oneIndexSize = lengthSum;

    std::unordered_map<int64_t, invIndexStruct> invIndex;
    std::unordered_map<int64_t, unsigned long long> oneIndex;
    
    indexLists.resize(1);
    indexVecs.resize(1);
    indexLists.reserve(oneIndexSize + invIndexSize);
    indexVecs.reserve(n);

    int partNum = 0;  // partition num
    // int prevLen = 0;
	int maxN = std::max(n, qn);
    bool * quickRef = new bool[maxN];
    memset(quickRef, 0, sizeof(bool) * maxN);
    bool * negRef = new bool[maxN];
    memset(negRef, 0, sizeof(bool) * maxN);

#if PART_COE == 0
    int maxIndexPartNum = floor(2 * coe * work_maxSize + EPS) + 1; 
#elif PART_COE == 1
	int maxIndexPartNum = floor(coePart * work_maxSize + EPS) + 1; 
#endif

    std::vector<int> hashValues;
    std::vector<int> positions;
    std::vector<std::vector<int>> subquery;  // first part id second tokens
    std::vector<std::vector<int>> oneHashValues;
    std::vector<unsigned long long> invPtr;
    std::vector<unsigned long long> intPtr;
    std::vector<std::vector<unsigned long long>> onePtr;
    std::vector<std::pair<int, int>> values;	// <value, loc>
    std::vector<int> scores;

    subquery.resize(maxIndexPartNum);
    hashValues.resize(maxIndexPartNum);
    positions.resize(maxIndexPartNum);
    oneHashValues.resize(maxIndexPartNum);
    invPtr.resize(maxIndexPartNum);
    intPtr.resize(maxIndexPartNum);
    onePtr.resize(maxIndexPartNum);
    values.resize(maxIndexPartNum);
    scores.resize(maxIndexPartNum);

    int low = 0, high = 0, candidate_size_num = 0;
	std::vector<std::pair<int, int>> range;
	std::vector<int> candidates;

	// Build index on smaller table
	for(int rid = 0; rid < n; rid++) {
		int len = work_dataset[rid].size();
        if(len==0) {
            workEmpty.emplace_back(rid);
            continue;
        }

		auto index_st = SetJoinUtil::logTime();

        // Here is to create a new group 
#if BRUTE_FORCE == 1
        if (len > high) {
            low = len;
            high = len * ALPHA;
#if PART_COE == 0
            partNum = floor(2 * coe * low + EPS) + 1;
#elif PART_COE == 1
			partNum = floor(coePart * low + EPS) + 1;
#endif
            range.emplace_back(low, high);
        } 
#endif 
#if BRUTE_FORCE == 0
#if PART_COE == 0
        partNum = floor(2 * coe * len + EPS) + 1;
#elif PART_COE == 1
		partNum = floor(coePart * len + EPS) + 1;
#endif
#endif

		hashValues.clear();
		oneHashValues.clear();
		subquery.clear();
		hashValues.resize(partNum);
		oneHashValues.resize(partNum);
		subquery.resize(partNum);

		for (auto &token : work_dataset[rid]) {
			int pid = token % partNum;
			auto &subrec = subquery[pid];
			subrec.push_back(token);
			hashValues[pid] = (PRIME * hashValues[pid] + token + 1);
		}

        for (int pid = 0; pid < partNum; ++pid) {
            if (oneHashValues[pid].size() == 0) {
                int mhv = 0, hv = hashValues[pid];
                auto &subrec = subquery[pid];
                for (ui idx = 0; idx < subrec.size(); idx++) {
                    int chv = (hv + mhv * prime_exp[subrec.size() - 1 - idx]);
                    mhv = (mhv * PRIME + subrec[idx] + 1);
                    chv -= (mhv * prime_exp[subrec.size() - 1 - idx]);
                    oneHashValues[pid].push_back(chv);
                }
            }
        }

        int pos = 0;
        for (int pid = 0; pid < partNum; pid++) {
            int hv = hashValues[pid];
            auto &subrec = subquery[pid];
#if BRUTE_FORCE == 0
            int64_t lenPart = len + pid * (work_maxSize + 1);
#elif BRUTE_FORCE == 1
			int64_t lenPart = low + pid * (work_maxSize + 1);
#endif
            auto &inv = invIndex[PACK(lenPart, hv)];
            if (inv.list_no == 0) {
                indexLists.push_back(invertedList());
                inv.list_no = indexLists.size() - 1;
            }
            indexLists[inv.list_no].add(std::make_pair(rid, pos), this);
            // build one inverted index
            for (auto &chv : oneHashValues[pid]) {
                auto &one = oneIndex[PACK(lenPart, chv)];
                if (one == 0) {
                    indexLists.push_back(invertedList());
                    one = indexLists.size() - 1;
                }
                indexLists[one].add(std::make_pair(rid, pos), this);
            }

            pos = pos + subrec.size();
        }

		index_cost += SetJoinUtil::repTime(index_st);
	}
	
	printf("Finish building index for work records\n");


	// query
	hashValues.clear();
	oneHashValues.clear();
	subquery.clear();
	subquery.resize(maxIndexPartNum);
    hashValues.resize(maxIndexPartNum);
    oneHashValues.resize(maxIndexPartNum);

    for (int rid = 0; rid < qn; rid++) {
        int len = query_dataset[rid].size();
        if(len==0) {
            queryEmpty.emplace_back(rid);
            continue;
        }

        int indexPartNum;
        int prevIndexPartNum = 0;
        int pos = 0;

        int indexLenLow = 0, indexLenHigh = 0;
        switch(simFType) {
            case SimFuncType::JACCARD :
                indexLenLow = std::max(ceil(1.0 * det * len - EPS), work_minSize * 1.0); // [\delta * s, s]
		        indexLenHigh = std::min(floor((1.0 / det) * len + EPS), work_maxSize * 1.0);
                break;
            case SimFuncType::COSINE :
                indexLenLow = std::max(ceil(1.0 * det * det * len - EPS), work_minSize * 1.0);
                indexLenHigh = std::min(floor(1.0 / (det * det) * len + EPS), work_maxSize * 1.0);
                break;
            case SimFuncType::DICE :
                indexLenLow = std::max(ceil(1.0 * det / (2.0 - det) * len - EPS), work_minSize * 1.0);
                indexLenHigh = std::min(floor(1.0 * (2.0 - det) / det * len + EPS), work_maxSize * 1.0);
                break;
        }

#if BRUTE_FORCE == 0
		for(int ttemp = work_minSize; ttemp <= work_maxSize; ttemp ++) {
#elif BRUTE_FORCE == 1
        for (int indexLenGrp = 0; indexLenGrp < (int)range.size(); ++indexLenGrp) {
            if (range[indexLenGrp].second < indexLenLow || range[indexLenGrp].first > indexLenHigh) 
				continue;
#endif
#if BRUTE_FORCE == 0
			int indexLen = ttemp;
#elif BRUTE_FORCE == 1
            int indexLen = range[indexLenGrp].first;
#endif
			
#if PART_COE == 0
            indexPartNum = floor(2 * coe * indexLen + EPS) + 1;
#elif PART_COE == 1
			indexPartNum = floor(coePart * indexLen + EPS) + 1;
#endif
            // split the query into multiple parts if prevIndexPartNum != indexPartNum
            // it means if the indexLenGrp's first range is not change
            // But does it possible?
            if (prevIndexPartNum != indexPartNum) {

                // clear subquery oneHashValues  hashValues
                for (int pid = 0; pid < indexPartNum; ++pid) {
                    subquery[pid].clear();
                    oneHashValues[pid].clear();
                    hashValues[pid] = 0;
                }
                
                // allocate the tokens into subquery and hash each part
                // That is the way how the query is splitted into indexPartNum
                for (auto &token : query_dataset[rid]) {
                    int pid = token % indexPartNum;
                    subquery[pid].push_back(token);
                    hashValues[pid] = PRIME * hashValues[pid] + token + 1;
                }

                // Dont know the position meaning?
                pos = 0;
                for (int pid = 0; pid < indexPartNum; pid++) {
                    positions[pid] = pos;
                    pos = pos + subquery[pid].size();
                }

                prevIndexPartNum = indexPartNum;
            }

            // Initialize onePtr and reserve each space
            for (int pid = 0; pid < indexPartNum; ++pid) {
                onePtr[pid].clear();
                onePtr[pid].reserve(subquery[pid].size());
            }

            auto alloc_st = SetJoinUtil::logTime();
            // Iterate each part find each part if there is identical part already exits in inverted list
            // Based on the above result, initialize the value and scores
            for (int pid = 0; pid < indexPartNum; pid++) {
                
                // A hash 
                int64_t lenPart = indexLen + pid * (work_maxSize + 1);
                
                int v1 = 0;
                auto invit = invIndex.find(PACK(lenPart, hashValues[pid]));
                if (invit != invIndex.end()) {
                    invPtr[pid] = invit->second.list_no;
                    v1 = -indexLists[invit->second.list_no].cnt;
                } else {
                    invPtr[pid] = 0;
                }
                values[pid] = std::make_pair(v1, pid);
                scores[pid] = 0;
            }
            int heap_cnt = indexPartNum;
            make_heap(values.begin(), values.begin() + heap_cnt);


            int cost = 0;
#if BRUTE_FORCE == 0
			int rLen = ttemp;
#elif BRUTE_FORCE == 1
			int rLen = range[indexLenGrp].second;
			// int rLen = min(range[indexLenGrp].second, len);
#endif 
            int Ha = floor((len - det * rLen) / (1 + det) + EPS);
            int Hb = floor((rLen - det * len) / (1 + det) + EPS);
            int maxH = Ha + Hb;
            switch(simFType) {
                case SimFuncType::JACCARD : maxH = floor(coe * (len + rLen) + EPS); break;
                case SimFuncType::COSINE : maxH = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS); break;
                case SimFuncType::DICE : maxH = floor((1.0 - det) * (len + rLen) + EPS); break;
            }
                
#if BRUTE_FORCE == 0
			for(int pid = 0; pid < indexPartNum; pid ++) {
				if (invPtr[pid] != 0) {
					auto &vec = indexLists[invPtr[pid]].getVector(this);
					for (auto lit = vec.begin(); lit != vec.end(); lit++) {
						// if(lit->first == rid)
						// 	continue;

						int rLen = work_dataset[lit->first].size();
						int Ha = floor((len - det * rLen) / (1 + det) + EPS);
						int Hb = floor((rLen - det * len) / (1 + det) + EPS);
						int H = Ha + Hb; // maximum allowable difference
                        if(simFType == SimFuncType::COSINE)
                            H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                        else if(simFType == SimFuncType::DICE) {
                            Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                            Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                            H = floor((1.0 - det) * (len + rLen) + EPS);
                        }

						// If the current iteration index i is greater than this maximum allowable difference H
						// current entry in the inverted list is skipped,
						// if (i > H) continue;
						// position filter
						if (negRef[lit->first] == false && quickRef[lit->first] == false) {
							candidates.push_back(lit->first);
                        }
						if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
							negRef[lit->first] = true;
						else
							quickRef[lit->first] = true;
					}
				}
			}
#elif BRUTE_FORCE == 1
            // We need use greedy selection in maxH + 1 times
            for (int i = 0; i < maxH + 1; ++i) {
                auto sel = values.front();
                pop_heap(values.begin(), values.begin() + heap_cnt);
                int pid = sel.second;
                ++scores[pid];
                cost -= sel.first;
                int64_t lenPart = indexLen + pid * (work_maxSize + 1);

                if (scores[sel.second] == 1) {
                    if (invPtr[pid] != 0) {
                        auto &vec = indexLists[invPtr[pid]].getVector(this);
                        for (auto lit = vec.begin(); lit != vec.end(); lit++) {
							// if(lit->first == rid)
							// 	continue;

                            int rLen = work_dataset[lit->first].size();
                            int Ha = 0, Hb = 0, H = 0;
                            switch(simFType) {
                                case SimFuncType::JACCARD : 
                                    Ha = floor((len - det * rLen) / (1 + det) + EPS);
                                    Hb = floor((rLen - det * len) / (1 + det) + EPS);
                                    H = Ha + Hb; // maximum allowable difference
                                    break;
                                case SimFuncType::COSINE : 
                                    Ha = floor(len - det * sqrt(len * rLen) + EPS);
                                    Hb = floor(rLen - det * sqrt(len * rLen) + EPS);
                                    H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                                    break;
                                case SimFuncType::DICE : 
                                    Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                                    Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                                    H = floor((1.0 - det) * (len + rLen) + EPS);
                                    break;
                            }

                            // If the current iteration index i is greater than this maximum allowable difference H
                            // current entry in the inverted list is skipped,
                            if (i > H) continue;
                            // position filter
                            if (negRef[lit->first] == false && quickRef[lit->first] == false)
                                candidates.push_back(lit->first);
                            if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
                                negRef[lit->first] = true;
                            else
                                quickRef[lit->first] = true;
                        }
                    }

                    // maintain heap
                    // Here is to find the next pair that insert to the heap
                    // Basically it is to consider the situation of the 1-deletion of the current part
                    int v2 = 0;

                    // search if the 1-deletion 
                    auto oneit = oneIndex.find(PACK(lenPart, hashValues[pid]));
                    if (oneit != oneIndex.end()) {
                        intPtr[pid] = oneit->second;
                        v2 -= indexLists[oneit->second].cnt;
                    } else {
                        intPtr[pid] = 0;
                    }


                    if (oneHashValues[pid].size() == 0) {
                        int mhv = 0, hv = hashValues[pid];
                        auto &sq = subquery[pid];
                        for (ui idx = 0; idx < sq.size(); idx++) {
                            int chv = hv + mhv * prime_exp[sq.size() - 1 - idx];
                            mhv = mhv * PRIME + sq[idx] + 1;
                            chv -= mhv * prime_exp[sq.size() - 1 - idx];
                            oneHashValues[pid].push_back(chv);
                        }
                    }
                    for (int id = 0; id < (int)oneHashValues[pid].size(); ++id) {
                        auto invit = invIndex.find(PACK(lenPart, oneHashValues[pid][id]));
                        if (invit != invIndex.end()) {
                            onePtr[pid].push_back(invit->second.list_no);
                            v2 -= indexLists[invit->second.list_no].cnt;
                        } else {
                            onePtr[pid].push_back(0);
                        }
                    }

                    values[heap_cnt - 1].first = v2;
                    push_heap(values.begin(), values.begin() + heap_cnt);
                } 
				else {
                    auto ov_st = SetJoinUtil::logTime();
                    // add candidates
                    if (intPtr[pid] != 0) {
                        auto &vec = indexLists[intPtr[pid]].getVector(this);
                        for (auto lit = vec.begin(); lit != vec.end(); lit++) {
							if(lit->first == rid)
								continue;

                            int rLen = work_dataset[lit->first].size();
                            int Ha = 0, Hb = 0, H = 0;
                            switch(simFType) {
                                case SimFuncType::JACCARD : 
                                    Ha = floor((len - det * rLen) / (1 + det) + EPS);
                                    Hb = floor((rLen - det * len) / (1 + det) + EPS);
                                    H = Ha + Hb; // maximum allowable difference
                                    break;
                                case SimFuncType::COSINE : 
                                    Ha = floor(len - det * sqrt(len * rLen) + EPS);
                                    Hb = floor(rLen - det * sqrt(len * rLen) + EPS);
                                    H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                                    break;
                                case SimFuncType::DICE : 
                                    Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                                    Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                                    H = floor((1.0 - det) * (len + rLen) + EPS);
                                    break;
                            }

                            if (i > H) continue;
                            if (negRef[lit->first] == false && quickRef[lit->first] == false)
                                candidates.push_back(lit->first);
                            if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
                                negRef[lit->first] = true;
                            else
                                quickRef[lit->first] = true;
                        }
                    }

                    for (int id = 0; id < (int)onePtr[pid].size(); ++id) {
                        if (onePtr[pid][id] != 0) {
                            auto &vec = indexLists[onePtr[pid][id]].getVector(this);
                            for (auto lit = vec.begin(); lit != vec.end(); lit++) {
								if(lit->first == rid)
									continue;

                                int rLen = work_dataset[lit->first].size();
                                int Ha = 0, Hb = 0, H = 0;
                                switch(simFType) {
                                    case SimFuncType::JACCARD : 
                                        Ha = floor((len - det * rLen) / (1 + det) + EPS);
                                        Hb = floor((rLen - det * len) / (1 + det) + EPS);
                                        H = Ha + Hb; // maximum allowable difference
                                        break;
                                    case SimFuncType::COSINE : 
                                        Ha = floor(len - det * sqrt(len * rLen) + EPS);
                                        Hb = floor(rLen - det * sqrt(len * rLen) + EPS);
                                        H = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS);
                                        break;
                                    case SimFuncType::DICE : 
                                        Ha = floor(((2 - det) * len - det * rLen) / 2 + EPS);
                                        Hb = floor(((2 - det) * rLen - det * len) / 2 + EPS);
                                        H = floor((1.0 - det) * (len + rLen) + EPS);
                                        break;
                                }

                                if (i > H) continue;
                                if (negRef[lit->first] == false && quickRef[lit->first] == false)
                                	candidates.push_back(lit->first);	
                                if (positions[pid] - lit->second > Ha || lit->second - positions[pid] > Hb)
                                    negRef[lit->first] = true;
                                else
                                    quickRef[lit->first] = true;
                            }
                        }
                    }
                    overlap_cost += SetJoinUtil::repTime(ov_st);
                    // maintain heap
                    --heap_cnt;
                }
            }
#endif
			allocation_cost += SetJoinUtil::repTime(alloc_st);
            listlens += cost;

            // compute candidates
			candidate_size_num += candidates.size();
            for (ui idx = 0; idx < candidates.size(); idx++) {
                if (negRef[candidates[idx]] == false && quickRef[candidates[idx]] == true) {
                    if (overlapRS(rid, candidates[idx]) == true) {
#if MAINTAIN_VALUE == 0
                        sim_pairs.emplace_back(rid, candidates[idx]);
                        if(sim_pairs.size() > maxHeapSize)
                            return;
#elif MAINTAIN_VALUE == 1
                        double val = isWeightedComp ? (this->*weightedFunc)(rid, candidates[idx]) 
                                                    : (this->*normalFunc)(rid, candidates[idx]);
                        if(result_pairs_.size() < maxHeapSize)
                            result_pairs_.emplace_back(rid, candidates[idx], val);
                        else {
                            if(isHeap == 0) {
                                std::make_heap(result_pairs_.begin(), result_pairs_.end());
                                isHeap = 1;
                            }
                            if(result_pairs_[0].val < val) {
                                std::pop_heap(result_pairs_.begin(), result_pairs_.end());
                                result_pairs_.pop_back();
                                result_pairs_.emplace_back(rid, candidates[idx], val);
                                std::push_heap(result_pairs_.begin(), result_pairs_.end());
                            }
                        }
#endif
						resultNum ++;
						// sim_pairs.emplace_back(rid, candidates[idx]);
                    }
                    candidateNum++;
                }
                quickRef[candidates[idx]] = false;
                negRef[candidates[idx]] = false;
            }
            candidates.clear();
        }
	}

    // empty sets
#if APPEND_EMPTY == 1
    ui totalWorkEmpty = workEmpty.size();
    ui totalQueryEmpty = queryEmpty.size();
    for(ui i = 0; i < totalWorkEmpty; i++)
        for(ui j = 0; j < totalQueryEmpty; j++)
            sim_pairs.emplace_back(queryEmpty[j], workEmpty[i]);
#endif 

#if MAINTAIN_VALUE == 1
    for(const auto &p : result_pairs_)
        sim_pairs.emplace_back(p.id1, p.id2);
#endif

    gettimeofday(&allend, NULL);
    double all = allend.tv_sec - allstart.tv_sec + (allend.tv_usec - allstart.tv_usec) / 1e6;

	// Check if there is duplicate pairs.
	// Disable it when conducting self-join test.
	sort(sim_pairs.begin(), sim_pairs.end());
    auto sim_iter = unique(sim_pairs.begin(), sim_pairs.end());

    // duplicate
	if(sim_iter != sim_pairs.end()) {
        ui dup_count = 0;
        for(auto it = sim_iter; it != sim_pairs.end(); it++) {
#if OUTPUT_DUP == 1
            printf("%d %d ", it->first, it->second);
            printf("#");
            for(auto &fen : query_dataset[it->first])
                printf("%d ", fen);
            printf("#");
            for(auto &sen : work_dataset[it->second])
                printf("%d ", sen);
            printf("#\n");
#endif
            ++ dup_count;
        }
        printf("Results contain duplicate pairs: %u\n", dup_count);

        // deduplicate
        sim_pairs.erase(sim_iter, sim_pairs.end());
		exit(1);
	}

    // print
    int v[6] = { 0 };
    for (auto &vec : indexVecs) {
        if (vec.size() < 6) ++v[vec.size()];
    }
    fprintf(stderr, "total time: %.3fs\n", allend.tv_sec - allstart.tv_sec + (allend.tv_usec - allstart.tv_usec) / 1e6);
    fprintf(stderr, "%lu %lu %lu %.3f\n", resultNum, candidateNum, listlens, all);
    fprintf(stderr, "AllocCost %.3f  IndexCost: %.3f OverlapCost : %.3f\n",allocation_cost, index_cost, overlap_cost);

    unsigned long long pairs_amount = 0;
    for(auto const & vec : indexVecs) {
        pairs_amount+=vec.size();
    }
    fprintf(stderr, "The amount of pairs in the indexVecs is: %llu\n", pairs_amount);
    SetJoinUtil::printMemory();

#ifdef WRITE_RESULT
    simp_ofs.close();
#endif

    delete[] quickRef;
    delete[] negRef;
}