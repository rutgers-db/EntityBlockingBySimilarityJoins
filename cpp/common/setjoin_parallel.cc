/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/setjoin_parallel.h"


// Function to create the index of Partitions and one-deletion neighbors
void SetJoinParallel::index(double threshold) 
{
#if MAINTAIN_VALUE == 1
    if(simFType == SimFuncType::JACCARD) {
        weightedFunc = &SetJoinParallel::weightedJaccard;
        normalFunc = &SetJoinParallel::jaccard;
    }
    else if(simFType == SimFuncType::COSINE) {
        weightedFunc = &SetJoinParallel::weightedCosine;
        normalFunc = &SetJoinParallel::cosine;
    }
    else if(simFType == SimFuncType::DICE) {
        weightedFunc = &SetJoinParallel::weightedDice;
        normalFunc = &SetJoinParallel::dice;
    }
#endif

    if(!ifRS && !flagIC) overlapFunc = &SetJoinParallel::overlapSelf;
    else if(!ifRS && flagIC) overlapFunc = &SetJoinParallel::overlapSelfIC;
    else if(ifRS && !flagIC) overlapFunc = &SetJoinParallel::overlapRS;
    else overlapFunc = &SetJoinParallel::overlapRSIC;

    // Initilaize the Parameters
    det = threshold;

    // ALPHA = 1.0 / threshold + 0.01;
    // cosine has a different upper bound for ALPHA (1 / \alpha)
    double cosineALPHA = (2.0 * (1 - det * det) + det * det * det * det + 2 * sqrt(2) * det * det * sqrt(1 - det * det)) / (det * det);
    std::cout << "cosine ALPHA: " << cosineALPHA << std::endl;
    ALPHA = 1.0;
    if(ALPHA > cosineALPHA) {
        ALPHA = cosineALPHA - 0.02;
        std::cout << "new ALPHA" << ALPHA << std::endl;
    }
    if(det < 0.9)
        ALPHA = 1.8;

    coe = (1 - det) / (1 + det);
    coePart = (1 - det) / det;
    if(simFType == SimFuncType::COSINE) 
        coePart = (1 - det * det) / (det * det);
    else if(simFType == SimFuncType::DICE)
        coePart = 2 * (1 - det) / det;

    // the possible maximum part amount of the maximum record
    // maxIndexPartNum = floor(2 * coe * work_maxSize + EPS) + 1; 
    maxIndexPartNum = floor(coePart * work_maxSize + EPS) + 1;

	// show parameters
	showPara();

    // Know how many tokens there
    ui tokenNum = 0;
    for (size_t i = 0; i < work_dataset.size(); i++) {
        if(work_dataset[i].empty())
            continue;
        if (tokenNum < work_dataset[i].back()) 
			tokenNum = work_dataset[i].back();
    }
    if(ifRS) {
        for(size_t i = 0; i < query_dataset.size(); i++) {
            if(query_dataset[i].empty())
                continue;
            if(tokenNum < query_dataset[i].back())
                tokenNum = query_dataset[i].back();
        }
    }

    printf("The tokenNum is %u \n", tokenNum);

    // Initialize prime exponential array for hashing
	ui maxLineLength = std::max(work_maxSize, query_maxSize);
    prime_exp = new int[maxLineLength + 1];
    prime_exp[0] = 1;
    for (ui i = 1; i <= maxLineLength; ++i)
        prime_exp[i] = prime_exp[i - 1] * PRIME;

    // Initialize quick reference arrays
    quickRef2D = new bool *[MAXTHREADNUM];
    negRef2D = new bool *[MAXTHREADNUM];
    for (ui i = 0; i < MAXTHREADNUM; ++i) {
        quickRef2D[i] = new bool[work_n];
        memset(quickRef2D[i], 0, sizeof(bool) * work_n);
        negRef2D[i] = new bool[work_n];
        memset(negRef2D[i], 0, sizeof(bool) * work_n);
    }

    // Get the ranges(The group based on the record's size)
    // We group those size similar ones together
    range_id.resize(work_n);
    ui low = 0, high = 0;
    for (ui rid = 0; rid < work_n; rid++) {
        ui len = work_dataset[rid].size();
        if(len == 0)
            continue;

        if (len > high) {
            low = len;
            high = low * ALPHA;
            range.push_back(std::make_pair(low, high));
            range_st.push_back(rid);
            workLength.emplace_back(len);
        }
        range_id[rid] = range.size() - 1;
    }
    range_st.push_back(work_n);

    // Map the query set
    if(ifRS) {
        int rangeid = 0;
        rangeIdQuery.resize(query_n);
        fill(rangeIdQuery.begin(), rangeIdQuery.end(), -1);

        for(ui rid = 0; rid < query_n; rid++) {
            ui len = query_dataset[rid].size();
            while(len > range[rangeid].second && rangeid < (int)range.size() - 1)
                rangeid ++;
            if(len >= range[rangeid].first && len <= range[rangeid].second)
                rangeIdQuery[rid] = rangeid;
        }

        int qid = 0;
        rangeQueryAdd.resize(query_n);

        while(rangeIdQuery[qid] == -1) {
            if(ui(query_dataset[qid].size() / det) >= range[0].first)
                rangeQueryAdd[qid] = 0;
            qid ++;
        }
        qid = query_n - 1;
        while(rangeIdQuery[qid] == -1) {
            if(ui(query_dataset[qid].size() * det) <= range[range.size()-1].second)
                rangeQueryAdd[qid] = range.size() - 1;
            qid --;
        }
    }

    // Initialize subquery and hash value vectors
    std::vector<std::vector<TokenLen>> subquery[MAXTHREADNUM]; // first thread id second tokens
    for (ui i = 0; i < MAXTHREADNUM; i++) {
        subquery[i].resize(maxIndexPartNum);
    }

    // To store the keys of parts and the one deletion neighbors
    parts_keys.resize(work_n);
    partsKeysQuery.resize(query_n);
    onedelete_keys.resize(work_n);
    oneDeleteKeysQuery.resize(query_n);
    odkeys_st.resize(work_n);
    oDKeysStQuery.resize(query_n);

    // Calculate hash values for partitions and one-deletion neighbors
    auto calHash_st = SetJoinParallelUtil::logTime();
	// Parallel loop to calculate hash values
#pragma omp parallel for
    for (ui rid = 0; rid < work_n; rid++) {
        ui len = work_dataset[rid].size();
        if(len == 0)
            continue;

        const ui cur_low = range[range_id[rid]].first;
        // const ui partNum = floor(2 * coe * cur_low + EPS) + 1;
        const ui partNum = floor(coePart * cur_low + EPS) + 1;

        ui tid = omp_get_thread_num();
        subquery[tid].clear();
        parts_keys[rid].resize(partNum);
        subquery[tid].resize(partNum);

        // Attention: If the parition is empty, then its hashvalue is 0
        for (auto &token : work_dataset[rid]) {
            ui pid = token % partNum;
            auto &subrec = subquery[tid][pid];
            subrec.push_back(token);

            // get the keys of partitions (or get the hash values)
            parts_keys[rid][pid] = PRIME * parts_keys[rid][pid] + token + 1;
        }

        // for getting the keys
        onedelete_keys[rid].resize(len);
        odkeys_st[rid].resize(partNum + 1);
        TokenLen tmp_cnt = 0;
        for (ui pid = 0; pid < partNum; ++pid) {
            // int64_t lenPart = cur_low + pid * (work_maxSize + 1);
            ui mhv = 0, hv = parts_keys[rid][pid];
            auto &subrec = subquery[tid][pid];

            // we store the keys into one array for each record.
            // so we need to know the boundary of keys for onedeletions of each partition
            odkeys_st[rid][pid] = tmp_cnt;

            // Maybe the subrec.size() is 0 cause some partitions are empty
            for (ui idx = 0; idx < subrec.size(); idx++) {
                int chv = hv + mhv * prime_exp[subrec.size() - 1 - idx];
                mhv = mhv * PRIME + subrec[idx] + 1;
                chv -= mhv * prime_exp[subrec.size() - 1 - idx];

                // get the keys of one-deletion neighbors from the current partition
                // auto odkey = PACK(lenPart, chv);
                onedelete_keys[rid][tmp_cnt++] = chv;
            }
            // pos += subrec.size();
        }
        odkeys_st[rid][partNum] = tmp_cnt;
    }
    // calculate for query set
    if(ifRS) {
#pragma omp parallel for
        for (ui rid = 0; rid < query_n; rid++) {
            ui len = query_dataset[rid].size();
            if(len == 0 || rangeIdQuery[rid] == -1)
                continue;
            // if(rangeIdQuery[rid] == -1)
            //     continue;

            const ui cur_low = range[rangeIdQuery[rid]].first;
            const ui partNum = floor(coePart * cur_low + EPS) + 1;

            ui tid = omp_get_thread_num();
            subquery[tid].clear();
            partsKeysQuery[rid].resize(partNum);
            subquery[tid].resize(partNum);

            // Attention: If the parition is empty, then its hashvalue is 0
            for (auto &token : query_dataset[rid]) {
                ui pid = token % partNum;
                auto &subrec = subquery[tid][pid];
                subrec.push_back(token);

                // get the keys of partitions (or get the hash values)
                partsKeysQuery[rid][pid] = PRIME * partsKeysQuery[rid][pid] + token + 1;
            }

            // for getting the keys
            oneDeleteKeysQuery[rid].resize(len);
            oDKeysStQuery[rid].resize(partNum + 1);
            TokenLen tmp_cnt = 0;
            for (ui pid = 0; pid < partNum; ++pid) {
                // int64_t lenPart = cur_low + pid * (work_maxSize + 1);
                ui mhv = 0, hv = partsKeysQuery[rid][pid];
                auto &subrec = subquery[tid][pid];

                // we store the keys into one array for each record.
                // so we need to know the boundary of keys for onedeletions of each partition
                oDKeysStQuery[rid][pid] = tmp_cnt;

                // Maybe the subrec.size() is 0 cause some partitions are empty
                for (ui idx = 0; idx < subrec.size(); idx++) {
                    int chv = hv + mhv * prime_exp[subrec.size() - 1 - idx];
                    mhv = mhv * PRIME + subrec[idx] + 1;
                    chv -= mhv * prime_exp[subrec.size() - 1 - idx];

                    // get the keys of one-deletion neighbors from the current partition
                    // auto odkey = PACK(lenPart, chv);
                    oneDeleteKeysQuery[rid][tmp_cnt++] = chv;
                }
                // pos += subrec.size();
            }
            oDKeysStQuery[rid][partNum] = tmp_cnt;
        }
    }

    index_cost = SetJoinParallelUtil::repTime(calHash_st);
    std::cout << "Calculating Hash Values of Partitions and OneDeleteion Neighbors Time Cost: " << index_cost << std::endl;
    std::cout << "Now Let's building the inverted Index of these Paritions" << std::endl;

    auto sort_st = SetJoinParallelUtil::logTime();
    printf("There are %lu range groups in the setJoin\n", range.size());

    // Initialize index arrays
    invertedIndex.parts_rids = new std::vector<std::vector<ui>>[range.size()];
    invertedIndex.ods_rids = new std::vector<std::vector<ui>>[range.size()];
    invertedIndex.parts_index_hv = new std::vector<std::vector<ui>>[range.size()];
    invertedIndex.od_index_hv = new std::vector<std::vector<ui>>[range.size()];
    invertedIndex.parts_index_offset = new std::vector<std::vector<ui>>[range.size()];
    invertedIndex.od_index_offset = new std::vector<std::vector<ui>>[range.size()];
    invertedIndex.parts_index_cnt = new std::vector<std::vector<TokenLen>>[range.size()];
    invertedIndex.od_index_cnt = new std::vector<std::vector<TokenLen>>[range.size()];

    // Building Inverted Index for the hashed partitions and one-deletion neghbors
#pragma omp parallel for
    for (ui i = 0; i < range.size(); i++) {
        auto const &rid_st = range_st[i];
        auto const &rid_ed = range_st[i + 1];
        const TokenLen partNum = parts_keys[rid_st].size();

        // Allocate Memory for index
        invertedIndex.parts_rids[i].resize(partNum);
        invertedIndex.ods_rids[i].resize(partNum);
        invertedIndex.parts_index_hv[i].resize(partNum);
        invertedIndex.od_index_hv[i].resize(partNum);
        invertedIndex.parts_index_offset[i].resize(partNum);
        invertedIndex.parts_index_cnt[i].resize(partNum);
        invertedIndex.od_index_offset[i].resize(partNum);
        invertedIndex.od_index_cnt[i].resize(partNum);

        // Build Index
        for (TokenLen pid = 0; pid < partNum; pid++) {
            ui parts_amount = rid_ed - rid_st;
            ui one_deletion_amount = 0;
            // First Index Partitions

            auto &cur_rids = invertedIndex.parts_rids[i][pid];
            cur_rids.reserve(parts_amount);

            for (ui rid = rid_st; rid < rid_ed; rid++) {
                cur_rids.emplace_back(rid);
                one_deletion_amount += odkeys_st[rid][pid + 1] - odkeys_st[rid][pid];
            }

            // sort the parts_rid[i][pid]
            sort(cur_rids.begin(), cur_rids.end(), [&pid, this](const ui &rid_1, const ui &rid_2) {
                if (parts_keys[rid_1][pid] == parts_keys[rid_2][pid])
                    return rid_1 < rid_2;
                return parts_keys[rid_1][pid] < parts_keys[rid_2][pid];
            });

            // Build the index pointers for the cur_rids
            ui prev_hv = parts_keys[cur_rids[0]][pid];
            TokenLen tmp_cnt = 1;
            ui ofs = 0;
            for (ui j = 1; j < cur_rids.size(); j++) {
                auto &cur_hv = parts_keys[cur_rids[j]][pid];
                if (cur_hv != prev_hv) {
                    invertedIndex.parts_index_hv[i][pid].emplace_back(prev_hv);
                    invertedIndex.parts_index_offset[i][pid].emplace_back(ofs);
                    invertedIndex.parts_index_cnt[i][pid].emplace_back(tmp_cnt);
                    prev_hv = cur_hv;
                    ofs = j;
                    tmp_cnt = 0;
                }
                if (tmp_cnt < std::numeric_limits<TokenLen>::max()) // the maximum of TokenLen
                    tmp_cnt++;
            }
            // We need to emplace_back more time out of the loop
            // because the last hash value do not have next hash value to trigger "cur_hv != prev_hv"
            invertedIndex.parts_index_hv[i][pid].emplace_back(prev_hv);
            invertedIndex.parts_index_offset[i][pid].emplace_back(ofs);
            invertedIndex.parts_index_cnt[i][pid].emplace_back(tmp_cnt);

            // Now build the index for od key
            if (one_deletion_amount == 0) // it means that in current range group, current partition, all of them are empty, we don't have elements partitioned in this partition
                continue;

            auto &cur_ods_rids = invertedIndex.ods_rids[i][pid];
            cur_ods_rids.resize(one_deletion_amount);

           	iota(cur_ods_rids.begin(), cur_ods_rids.end(), 0); // cur_rids temporarily filled with 0 to one_deletion_amount-1

            // temporary vector for the sorting the array
            // vector<ui> tmp_rid(one_deletion_amount);
            std::vector<ui> tmp_rid;
            tmp_rid.reserve(one_deletion_amount);
            std::vector<TokenLen> tmp_od_locs;
            tmp_od_locs.reserve(one_deletion_amount);
            for (ui rid = rid_st; rid < rid_ed; rid++) {
                auto const &od_loc_st = odkeys_st[rid][pid];
                auto const &od_loc_ed = odkeys_st[rid][pid + 1];
                for (auto od_loc = od_loc_st; od_loc < od_loc_ed; od_loc++) {
                    tmp_rid.emplace_back(rid);
                    tmp_od_locs.emplace_back(od_loc);
                }
            }

            // sort the parts_rid[i][pid]
            sort(cur_ods_rids.begin(), cur_ods_rids.end(), [&tmp_rid, &tmp_od_locs, &pid, this](const ui &id1, const ui &id2) {
                auto const &rid_1 = tmp_rid[id1];
                auto const &rid_2 = tmp_rid[id2];
                auto const &od_loc_1 = tmp_od_locs[id1];
                auto const &od_loc_2 = tmp_od_locs[id2];
                if (onedelete_keys[rid_1][od_loc_1] == onedelete_keys[rid_2][od_loc_2])
                    return rid_1 < rid_2;
                return onedelete_keys[rid_1][od_loc_1] < onedelete_keys[rid_2][od_loc_2];
            });

            // Build the index pointers for the cur_rids
            auto const &rid = tmp_rid[cur_ods_rids[0]];
            auto const &od_loc = tmp_od_locs[cur_ods_rids[0]];
            prev_hv = onedelete_keys[rid][od_loc];
            tmp_cnt = 1;
            ofs = 0;
            for (ui j = 1; j < cur_ods_rids.size(); j++) {
                auto const &_rid = tmp_rid[cur_ods_rids[j]];
                auto const &_od_loc = tmp_od_locs[cur_ods_rids[j]];
                auto const &cur_hv = onedelete_keys[_rid][_od_loc];
                if (cur_hv != prev_hv) {
                    invertedIndex.od_index_hv[i][pid].emplace_back(prev_hv);
                    invertedIndex.od_index_offset[i][pid].emplace_back(ofs);
                    invertedIndex.od_index_cnt[i][pid].emplace_back(tmp_cnt);
                    prev_hv = cur_hv;
                    ofs = j;
                    tmp_cnt = 0;
                }
                if (tmp_cnt < std::numeric_limits<TokenLen>::max()) // the maximum of TokenLen
                    tmp_cnt++;
            }
            invertedIndex.od_index_hv[i][pid].emplace_back(prev_hv);
            invertedIndex.od_index_offset[i][pid].emplace_back(ofs);
            invertedIndex.od_index_cnt[i][pid].emplace_back(tmp_cnt);

            // convert the id to rid
            for (ui j = 0; j < cur_ods_rids.size(); j++) {
                cur_ods_rids[j] = tmp_rid[cur_ods_rids[j]];
            }
        }
    }
    index_cost += SetJoinParallelUtil::repTime(sort_st);
    std::cout << "Sorting them And Partition Time Cost: " << SetJoinParallelUtil::repTime(sort_st) << std::endl;

    // cout << "Indexing Partitions and OneDeleteion Neighbors Time Cost: " << SetJoinParallelUtil::repTime(index_st) << endl;

    SetJoinParallelUtil::printMemory();
}


void SetJoinParallel::GreedyFindCandidateAndSimPairs(const int &tid, const int indexLenGrp, const ui rid, 
													 ui record_length, const std::vector<ui> &p_keys, 
													 const std::vector<ui> &od_keys, const std::vector<TokenLen> &odk_st) 
{
    auto mem_st = SetJoinParallelUtil::logTime();
    size_t totalRange = range.size();

    auto const indexPartNum = p_keys.size();
    assert(indexPartNum == odk_st.size() - 1);
    // if(indexPartNum != odk_st.size()-1) {
    //     printf("Error in %zu %zu %zu %zu\n", p_keys.size(), indexPartNum, odk_st.size(), odk_st.size()-1);
    //     exit(1);
    // }
    ui const len = record_length;
    // auto const indexLen = range[indexLenGrp].first;

    // Prepare thread-local storage for various data structures
    std::vector<ui> candidates;
    auto &invPtr = invPtrArr[tid];
    auto &intPtr = intPtrArr[tid];
    auto &onePtr = onePtrArr[tid];
    auto &values = valuesArr[tid];
    auto &scores = scoresArr[tid];
    invPtr.resize(indexPartNum);
    intPtr.resize(indexPartNum);
    onePtr.resize(indexPartNum);
    values.resize(indexPartNum);
    scores.resize(indexPartNum);
    auto negRef = negRef2D[tid];
    auto quickRef = quickRef2D[tid];
#if MAINTAIN_VALUE == 0
    auto &cur_result_pairs = result_pairs[tid];
#elif MAINTAIN_VALUE == 1
    auto &cur_result_pairs_ = result_pairs_[tid];
#endif

    // Initialize onePtr and reserve space for each part
    for (ui pid = 0; pid < indexPartNum; ++pid) {
        onePtr[pid].clear();
        assert(odk_st[pid + 1] >= odk_st[pid]);
        onePtr[pid].reserve(odk_st[pid + 1] - odk_st[pid]);
    }

    mem_cost[tid] += SetJoinParallelUtil::repTime(mem_st);

    // Iterate each part find each part if there is identical part already exits in inverted list
    // Based on the above result, initialize the value and scores
    auto find_st = SetJoinParallelUtil::logTime();
    for (TokenLen pid = 0; pid < indexPartNum; pid++) {
        // A hash
        // int64_t lenPart = indexLen + pid * (maxSize + 1);
        int v1 = 0;
        const auto &pkey = p_keys[pid];
        const auto &cur_hvs = invertedIndex.parts_index_hv[indexLenGrp][pid];
        auto invit = std::lower_bound(cur_hvs.begin(), cur_hvs.end(), pkey);
        // auto invit = lower_bound(parts_index[indexLenGrp].begin(), parts_index[indexLenGrp].end(), HashOfsCnt(pkey, 0, 0));
        if (invit != cur_hvs.end() && *invit == pkey) {
            auto dis = invit - cur_hvs.begin();
            invPtr[pid] = dis;
            v1 = -invertedIndex.parts_index_cnt[indexLenGrp][pid][dis];
        } else {
            invPtr[pid] = UINT_MAX;
        }
        values[pid] = std::make_pair(v1, pid);
        scores[pid] = 0;
    }
    find_cost[tid] += SetJoinParallelUtil::repTime(find_st);

    auto alloc_st = SetJoinParallelUtil::logTime();
    // Prepare heap for greedy selection
    ui heap_cnt = indexPartNum;
    std::make_heap(values.begin(), values.begin() + heap_cnt);

    // Initialize some variables for the greedy selection
    int cost = 0;
    // ui rLen = min(range[indexLenGrp].second, len);
    ui rLen = range[indexLenGrp].second;
    ui Ha = floor((len - det * rLen) / (1 + det) + EPS);
    ui Hb = floor((rLen - det * len) / (1 + det) + EPS);
    ui maxH = Ha + Hb;
    switch(simFType) {
        case SimFuncType::JACCARD : maxH = floor(coe * (len + rLen) + EPS); break;
        case SimFuncType::COSINE : maxH = floor(len + rLen - 2.0 * det * sqrt(len * rLen) + EPS); break;
        case SimFuncType::DICE : maxH = floor((1.0 - det) * (len + rLen) + EPS); break;
    }

    // We need use greedy selection in maxH + 1 times
    for (ui i = 0; i < maxH + 1; ++i) {
        // cout << i << " " << maxH << " " << heap_cnt << " " << indexPartNum << endl << flush;
        auto sel = values.front();
        std::pop_heap(values.begin(), values.begin() + heap_cnt);
        auto pid = sel.second;
        auto const pid_pos = (int)odk_st[pid];
        ++scores[pid];
        cost -= sel.first;
        auto const &pkey = p_keys[pid];

        if (scores[sel.second] == 1) {
            if (invPtr[pid] != UINT_MAX) {
                // Iterate the candidates that shares the same partition
                auto ofs_st = invertedIndex.parts_index_offset[indexLenGrp][pid][invPtr[pid]];
                auto ofs_ed = ofs_st + invertedIndex.parts_index_cnt[indexLenGrp][pid][invPtr[pid]];
                for (auto ofs = ofs_st; ofs < ofs_ed; ofs++) {
                    // auto const &hrp = parts_arr[indexLenGrp][ofs];
                    auto const tmp_rid = invertedIndex.parts_rids[indexLenGrp][pid][ofs];
                    if (tmp_rid >= rid && !ifRS) 
                        break;

                    int tmp_pos = odkeys_st[tmp_rid][pid];
                    ui rLen = work_dataset[tmp_rid].size();

                    int H = 0, Ha = 0, Hb = 0;
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
                    if ((int)i > H) continue;
                    // position filter
                    if (negRef[tmp_rid] == false && quickRef[tmp_rid] == false)
                        candidates.push_back(tmp_rid);

                    // We need to let them be int in case of negative minus result
                    if (pid_pos - tmp_pos > Ha || tmp_pos - pid_pos > Hb)
                        negRef[tmp_rid] = true;
                    else
                        quickRef[tmp_rid] = true;
                }
            }

            // maintain heap
            // Here is to find the next pair that insert to the heap
            // Basically it is to consider the situation of the 1-deletion of the current part
            int v2 = 0;

            // search if the 1-deletion
            const auto &cur_od_hvs = invertedIndex.od_index_hv[indexLenGrp][pid];
            auto oneit = lower_bound(cur_od_hvs.begin(), cur_od_hvs.end(), pkey);
            // auto oneit = oneIndex.find(PACK(lenPart, hashValues[pid]));
            if (oneit != cur_od_hvs.end() && *oneit == pkey) {
                auto dis = oneit - cur_od_hvs.begin();
                intPtr[pid] = oneit - cur_od_hvs.begin();
                v2 -= invertedIndex.od_index_cnt[indexLenGrp][pid][dis];
            } else {
                intPtr[pid] = UINT_MAX;
            }

            auto const &id_st = odk_st[pid];
            auto const &id_ed = odk_st[pid + 1];
            for (auto id = id_st; id < id_ed; id++) {
                auto const &odk = od_keys[id];
                const auto &cur_hvs = invertedIndex.parts_index_hv[indexLenGrp][pid];
                auto invit = lower_bound(cur_hvs.begin(), cur_hvs.end(), odk);
                if (invit != cur_hvs.end() && *invit == odk) {
                    auto dis = invit - cur_hvs.begin();
                    onePtr[pid].push_back(dis);
                    v2 -= invertedIndex.parts_index_cnt[indexLenGrp][pid][dis];
                } else {
                    onePtr[pid].push_back(UINT_MAX);
                }
            }

            values[heap_cnt - 1].first = v2;
            std::push_heap(values.begin(), values.begin() + heap_cnt);
        } 
        else {
            // auto ov_st = SetJoinParallelUtil::logTime();
            // add candidates
            if (intPtr[pid] != UINT_MAX) {
                auto ofs_st = invertedIndex.od_index_offset[indexLenGrp][pid][intPtr[pid]];
                auto ofs_ed = ofs_st + invertedIndex.od_index_cnt[indexLenGrp][pid][intPtr[pid]];
                // auto ofs_st = intPtr[pid].ofs;
                // auto ofs_ed = intPtr[pid].ofs + intPtr[pid].cnt;
                for (auto ofs = ofs_st; ofs < ofs_ed; ofs++) {
                    auto const tmp_rid = invertedIndex.ods_rids[indexLenGrp][pid][ofs];
                    if (tmp_rid >= rid && !ifRS) 
                        break;

                    int tmp_pos = odkeys_st[tmp_rid][pid];
                    // Attention here is the ods_arr
                    // auto const &hrp = ods_arr[indexLenGrp][ofs];
                    ui rLen = work_dataset[tmp_rid].size();

                    int H = 0, Ha = 0, Hb = 0;
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
                    if ((int)i > H) continue;
                    // position filter
                    if (negRef[tmp_rid] == false && quickRef[tmp_rid] == false)
                        candidates.push_back(tmp_rid);

                    // We need to let them be int in case of negative minus result
                    // auto const hrp_pos = (int)hrp.pos;
                    if (pid_pos - tmp_pos > Ha || tmp_pos - pid_pos > Hb)
                        negRef[tmp_rid] = true;
                    else
                        quickRef[tmp_rid] = true;
                }
            }

            for (int id = 0; id < (int)onePtr[pid].size(); ++id) {
                if (onePtr[pid][id] != UINT_MAX) {
                    auto ofs_st = invertedIndex.parts_index_offset[indexLenGrp][pid][onePtr[pid][id]];
                    auto ofs_ed = ofs_st + invertedIndex.parts_index_cnt[indexLenGrp][pid][onePtr[pid][id]];
                    // auto ofs_st = onePtr[pid][id].ofs;
                    // auto ofs_ed = onePtr[pid][id].ofs + onePtr[pid][id].cnt;
                    for (auto ofs = ofs_st; ofs < ofs_ed; ofs++) {
                        // auto const &hrp = parts_arr[indexLenGrp][ofs];
                        auto const tmp_rid = invertedIndex.parts_rids[indexLenGrp][pid][ofs];
                        if (tmp_rid >= rid && !ifRS) 
                            break;

                        int tmp_pos = odkeys_st[tmp_rid][pid];
                        ui rLen = work_dataset[tmp_rid].size();

                        int H = 0, Ha = 0, Hb = 0;
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
                        if ((int)i > H) continue;
                        // position filter
                        if (negRef[tmp_rid] == false && quickRef[tmp_rid] == false)
                            candidates.push_back(tmp_rid);

                        // We need to let them be int in case of negative minus result
                        // auto const hrp_pos = (int)hrp.pos;
                        if (pid_pos - tmp_pos > Ha || tmp_pos - pid_pos > Hb)
                            negRef[tmp_rid] = true;
                        else
                            quickRef[tmp_rid] = true;
                    }
                }
            }
            // overlap_cost += SetJoinParallelUtil::repTime(ov_st);
            // maintain heap
            --heap_cnt;
        }
    }
    alloc_cost[tid] += SetJoinParallelUtil::repTime(alloc_st);

    auto verif_st = SetJoinParallelUtil::logTime();
    // Clear candidates and update global results
    for (ui idx = 0; idx < candidates.size(); idx++) {
        if (negRef[candidates[idx]] == false && quickRef[candidates[idx]] == true) {
            if ((this->*overlapFunc)(rid, candidates[idx], 0, 0, 0) == true) {
                // resultNum++;
#if MAINTAIN_VALUE == 0
                if(ifRS) 
                    result_pairs[tid].emplace_back(rid, candidates[idx]);
                else {
                    assert(rid != candidates[idx]);
                    if(rid < candidates[idx])
                        result_pairs[tid].emplace_back(rid, candidates[idx]);
                    else
                        result_pairs[tid].emplace_back(candidates[idx], rid);
                }
#elif MAINTAIN_VALUE == 1
                if(ifRS) {
                    double val = isWeightedComp ? (this->*weightedFunc)(rid, candidates[idx]) 
                                                : (this->*normalFunc)(rid, candidates[idx]);
                    if(cur_result_pairs_.size() < maxHeapSize)
                        cur_result_pairs_.emplace_back(rid, candidates[idx], val);
                    else {
                        if(isHeap[tid] == 0) {
                            std::make_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
                            isHeap[tid] = 1;
                        }
                        if(cur_result_pairs_[0].val < val) {
                            std::pop_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
                            cur_result_pairs_.pop_back();
                            cur_result_pairs_.emplace_back(rid, candidates[idx], val);
                            std::push_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
                        }
                    }
                }
                else {
                    assert(rid != candidates[idx]);
                    double val = isWeightedComp ? (this->*weightedFunc)(rid, candidates[idx]) 
                                                : (this->*normalFunc)(rid, candidates[idx]);
                    if(cur_result_pairs_.size() < maxHeapSize) {
                        if(rid < candidates[idx]) cur_result_pairs_.emplace_back(rid, candidates[idx], val);
                        else cur_result_pairs_.emplace_back(candidates[idx], rid, val);
                    }
                    else {
                        if(isHeap[tid] == 0) {
                            std::make_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
                            isHeap[tid] = 1;
                        }
                        if(cur_result_pairs_[0].val < val) {
                            std::pop_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
                            cur_result_pairs_.pop_back();
                            if(rid < candidates[idx]) cur_result_pairs_.emplace_back(rid, candidates[idx], val);
                            else cur_result_pairs_.emplace_back(candidates[idx], rid, val);
                            std::push_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
                        }
                    }
                }
#endif
            }
            // candidateNum++;
        }
        quickRef[candidates[idx]] = false;
        negRef[candidates[idx]] = false;
    }
    candidates.clear();
    verif_cost[tid] += SetJoinParallelUtil::repTime(verif_st);

#if MAINTAIN_VALUE == 0
    if(result_pairs[tid].size() >= maxHeapSize)
        earlyTerminated[tid] = 1;
#endif
}


void SetJoinParallel::findSimPairsSelf() 
{
    auto find_st = SetJoinParallelUtil::logTime();
    std::cout << "Start finding similar pairs " << std::endl;

    // Initialize thread-local storage for sub-queries, partition keys, one-deletion keys, and one-deletion key starts
    std::vector<std::vector<TokenLen>> subquery[MAXTHREADNUM];
    std::vector<ui> p_keys[MAXTHREADNUM];
    std::vector<ui> od_keys[MAXTHREADNUM];
    std::vector<TokenLen> odk_st[MAXTHREADNUM];
    // Resize the thread-local storage vectors
    for (int i = 0; i < MAXTHREADNUM; i++) {
        subquery[i].resize(maxIndexPartNum);
        p_keys[i].resize(maxIndexPartNum);
        od_keys[i].resize(work_maxSize);
        odk_st[i].resize(maxIndexPartNum);
    }

    // Allocation and calculate candidates
#pragma omp parallel for
    for (int rid = 0; rid < (int)work_n; rid++) {
        int len = work_dataset[rid].size();
        if (len == 0)
            continue;

        // Get the thread ID for OpenMP
        const int tid = omp_get_thread_num();
        // We only need to access the indexLenGrp that current record belongs to and the previous group
        // But the previous group we have not created the partition key,etc for the current record
        // In this case we need generate them now

#if MAINTAIN_VALUE == 0
        if(earlyTerminated[tid] == 1)
            continue;
#endif
        int indexLenLow = 0, indexLenHigh = 0; // [\delta * s, s]
        switch(simFType) {
            case SimFuncType::JACCARD : 
                indexLenLow = std::max(ceil(1.0 * det * len - EPS), work_minSize * 1.0); 
		        indexLenHigh = std::min(floor((1.0 / det) * len + EPS), work_maxSize * 1.0);
                break;
            case SimFuncType::COSINE:
                indexLenLow = std::max(ceil(1.0 * det * det * len - EPS), work_minSize * 1.0);
                indexLenHigh = std::min(floor(1.0 / (det * det) * len + EPS), work_maxSize * 1.0);
                break;
            case SimFuncType::DICE : 
                indexLenLow = std::max(ceil(1.0 * det / (2.0 - det) * len - EPS), work_minSize * 1.0);
                indexLenHigh = std::min(floor(1.0 * (2.0 - det) / det * len + EPS), work_maxSize * 1.0);
                break;
        }

        int indexLenGrpLow = 0;
        switch(simFType) {
            case SimFuncType::JACCARD : 
                indexLenGrpLow = std::max(0, range_id[rid] - 1);
                break;
            case SimFuncType::COSINE : 
                indexLenGrpLow = std::max(0, range_id[rid] - 2);
                break;
            case SimFuncType::DICE : 
                int halfLen = floor(len / 2 + EPS);
                auto halfIt = lower_bound(workLength.begin(), workLength.end(), (ui)halfLen);

                if(*halfIt == (ui)halfLen) {
                    int idx = distance(workLength.begin(), halfIt);
                    indexLenGrpLow = std::max(0, idx - 1);
                }
                else if(*halfIt > (ui)halfLen) {
                    int idx = distance(workLength.begin(), halfIt) - 1;
                    if(idx < 0)
                        indexLenGrpLow = 0;
                    else
                        indexLenGrpLow = std::max(0, idx - 1);
                }
                else
                    indexLenGrpLow = 0;
                break;
        }

        // for (int indexLenGrp = indexLenGrpLow; indexLenGrp <= range_id[rid]; ++indexLenGrp) {
        for (int indexLenGrp = 0; indexLenGrp < (int)range.size(); ++indexLenGrp) {
            // if (range[indexLenGrp].second < (ui)indexLenLow || range[indexLenGrp].first > (ui)indexLenHigh) {
                // printf("current len: %d, current range: %d, rid range: %d %d %d, current low: %d, current high: %d, indexLow: %d, indexHigh: %d\n", 
                //         len, indexLenGrp, range_id[rid], range[range_id[rid]].first, range[range_id[rid]].second, range[indexLenGrp].first, 
                //         range[indexLenGrp].second, indexLenLow, indexLenHigh);
                // exit(1);
            //    continue;
            // }
            // if(range[indexLenGrp].first > (ui)indexLenHigh)
            //     break;
            if(range[indexLenGrp].first > (ui)len)
                break;
            if(range[indexLenGrp].second < (ui)indexLenLow)
                continue;
            // If the current length group is not the one the record belongs to, we need to recalculate partition keys, etc.
            if (indexLenGrp != range_id[rid]) {
                auto hash_st = SetJoinParallelUtil::logTime();

                const int &indexLen = range[indexLenGrp].first;
                // const int indexPartNum = floor(2 * coe * indexLen + EPS) + 1;
                const int indexPartNum = floor(coePart * indexLen + EPS) + 1;
                p_keys[tid].resize(indexPartNum);
                od_keys[tid].resize(len);
                odk_st[tid].resize(indexPartNum + 1);

                // clear subquery oneHashValues  hashValues
                for (int pid = 0; pid < indexPartNum; ++pid) {
                    subquery[tid][pid].clear();
                    p_keys[tid][pid] = 0;
                }

                // allocate the tokens into subquery and hash each part
                // That is the way how the query is splitted into indexPartNum
                for (auto &token : work_dataset[rid]) {
                    ui pid = token % indexPartNum;
                    subquery[tid][pid].push_back(token);
                    p_keys[tid][pid] = PRIME * p_keys[tid][pid] + token + 1;
                }

                TokenLen pos = 0;
                TokenLen tmp_cnt = 0;
                for (int pid = 0; pid < indexPartNum; ++pid) {
                    // int64_t lenPart = indexLen + pid * (work_maxSize + 1);
                    ui mhv = 0, hv = p_keys[tid][pid];
                    auto &subrec = subquery[tid][pid];

                    // we store the keys into one array for each record.
                    // so we need to know the boundary of keys for onedeletions of each partition
                    odk_st[tid][pid] = tmp_cnt;
                    for (ui idx = 0; idx < subrec.size(); idx++) {
                        ui chv = hv + mhv * prime_exp[subrec.size() - 1 - idx];
                        mhv = mhv * PRIME + subrec[idx] + 1;
                        chv -= mhv * prime_exp[subrec.size() - 1 - idx];

                        // get the keys of one-deletion neighbors from the current partition
                        // auto odkey = PACK(lenPart, chv);
                        od_keys[tid][tmp_cnt++] = chv;
                    }
                    pos += subrec.size();
                }
                odk_st[tid][indexPartNum] = tmp_cnt;

                hashInFind_cost[tid] += SetJoinParallelUtil::repTime(hash_st);
                // Now all the information we have got
                if(!flagIC)
                    GreedyFindCandidateAndSimPairs(tid, indexLenGrp, rid, work_dataset[rid].size(), 
                                                p_keys[tid], od_keys[tid], odk_st[tid]);
                else
                    GreedyFindCandidateAndSimPairs(tid, indexLenGrp, rid, work_dataset[rid].size(), 
                                                p_keys[tid], od_keys[tid], odk_st[tid]);
            } else {
                // We have calculated them if current indexLenGroup is the group this record belongs to the
                const auto &existing_p_keys = parts_keys[rid];
                const auto &existing_od_keys = onedelete_keys[rid];
                const auto &existing_odk_st = odkeys_st[rid];
                GreedyFindCandidateAndSimPairs(tid, indexLenGrp, rid, work_dataset[rid].size(), 
											   existing_p_keys, existing_od_keys, existing_odk_st);
            }
        }
    }

    fprintf(stderr, "%lu %lu %lu \n", resultNum, candidateNum, listlens);
    search_cost = SetJoinParallelUtil::repTime(find_st);

    // handle empty
    for(ui i = 0; i < work_n; i++) {
        if(work_dataset[i].size() > 0)
            break;
        workEmpty.emplace_back(i);
    }

    // release memory
    delete[] prime_exp;
    delete[] invertedIndex.parts_rids;
    delete[] invertedIndex.ods_rids;
    delete[] invertedIndex.parts_index_hv;
    delete[] invertedIndex.od_index_hv;
    delete[] invertedIndex.parts_index_offset;
    delete[] invertedIndex.od_index_offset;
    delete[] invertedIndex.parts_index_cnt;
    delete[] invertedIndex.od_index_cnt;

    for (int i = 0; i < MAXTHREADNUM; ++i) {
        delete[] quickRef2D[i];
        delete[] negRef2D[i];
    }
    delete[] quickRef2D;
    delete[] negRef2D;

    std::cout << "Finding Similar Pairs Time Cost: " << search_cost << std::endl;
}


void SetJoinParallel::findSimPairsRS() 
{
    auto find_st = SetJoinParallelUtil::logTime();
    std::cout << "Start finding similar pairs " << std::endl;

    // Initialize thread-local storage for sub-queries, partition keys, one-deletion keys, and one-deletion key starts
    std::vector<std::vector<TokenLen>> subquery[MAXTHREADNUM];
    std::vector<ui> p_keys[MAXTHREADNUM];
    std::vector<ui> od_keys[MAXTHREADNUM];
    std::vector<TokenLen> odk_st[MAXTHREADNUM];
    // Resize the thread-local storage vectors
    for (int i = 0; i < MAXTHREADNUM; i++) {
        subquery[i].resize(maxIndexPartNum);
        p_keys[i].resize(maxIndexPartNum);
        od_keys[i].resize(query_maxSize);
        odk_st[i].resize(maxIndexPartNum);
    }

    // Allocation and calculate candidates
#pragma omp parallel for
    for (int rid = 0; rid < (int)query_n; rid++) {
        int len = query_dataset[rid].size();
        if (len == 0)
            continue;

        // Get the thread ID for OpenMP
        const int tid = omp_get_thread_num();
        // We only need to access the indexLenGrp that current record belongs to and the previous group
        // But the previous group we have not created the partition key,etc for the current record
        // In this case we need generate them now

        int indexLenLow = std::max(ceil(1.0 * det * len - EPS), work_minSize * 1.0); // [\delta * s, s]
		int indexLenHigh = std::min(floor((1.0 / det) * len + EPS), work_maxSize * 1.0);
        if(simFType == SimFuncType::COSINE) {
            indexLenLow = std::max(ceil(1.0 * det * det * len - EPS), work_minSize * 1.0);
            indexLenHigh = std::min(floor(1.0 / (det * det) * len + EPS), work_maxSize * 1.0);
            // if(indexLenLow != std::max(ceil(1.0 * det * len - EPS), work_minSize * 1.0)) 
            //     printf("\033[1;31mFound!\033[0m\n");
        }
        else if(simFType == SimFuncType::DICE) {
            indexLenLow = std::max(ceil(1.0 * det / (2.0 - det) * len - EPS), work_minSize * 1.0);
            indexLenHigh = std::min(floor(1.0 * (2.0 - det) / det * len + EPS), work_maxSize * 1.0);
        }

        int rangeId = rangeIdQuery[rid] == -1 ? rangeQueryAdd[rid] : rangeIdQuery[rid];
        int indexLenGrpLow = std::max(0, rangeId - 1);
        int indexLenGrpHigh = std::min(int(range.size() - 1), rangeId + 1);
        if(simFType == SimFuncType::COSINE) {
            indexLenGrpLow = std::max(0, rangeId - 2);
            indexLenGrpHigh = std::min(int(range.size() - 1), rangeId + 2);
        }
        else if(simFType == SimFuncType::DICE) {
            // left part
            int halfLen = floor(len / 2 + EPS);
            auto halfIt = lower_bound(workLength.begin(), workLength.end(), (ui)halfLen);

            if(*halfIt == (ui)halfLen) {
                int idx = distance(workLength.begin(), halfIt);
                indexLenGrpLow = std::max(0, idx - 1);
            }
            else if(*halfIt > (ui)halfLen) {
                int idx = distance(workLength.begin(), halfIt) - 1;
                if(idx < 0)
                    indexLenGrpLow = 0;
                else
                    indexLenGrpLow = std::max(0, idx - 1);
            }
            else
                indexLenGrpLow = 0;

            // right part
            int doubleLen = ceil(len * 2 + EPS);
            auto doubleIt = lower_bound(workLength.begin(), workLength.end(), (ui)doubleLen);

            if(doubleIt == workLength.end())
                indexLenGrpHigh = int(range.size() - 1);
            else {
                int idx = distance(workLength.begin(), doubleIt);
                indexLenGrpHigh = std::min(int(range.size() - 1), idx + 1);
            }
        }

        // printf("%d %d %d %u\n", indexLenGrpLow, indexLenGrpHigh, rid, query_n);
        // for (int indexLenGrp = indexLenGrpLow; indexLenGrp <= indexLenGrpHigh; ++indexLenGrp) {
        for (int indexLenGrp = 0; indexLenGrp < (int)range.size(); ++indexLenGrp) {
            // if (range[indexLenGrp].second < (ui)indexLenLow || range[indexLenGrp].first > (ui)indexLenHigh) 
			// 	continue;
            if(range[indexLenGrp].first > (ui)indexLenHigh)
                break;
            if(range[indexLenGrp].second < (ui)indexLenLow)
                continue;
            // If the current length group is not the one the record belongs to, we need to recalculate partition keys, etc.
            if (indexLenGrp != rangeIdQuery[rid]) {
                auto hash_st = SetJoinParallelUtil::logTime();

                const int &indexLen = range[indexLenGrp].first;
                // const int indexPartNum = floor(2 * coe * indexLen + EPS) + 1;
                const int indexPartNum = floor(coePart * indexLen + EPS) + 1;
                p_keys[tid].resize(indexPartNum);
                od_keys[tid].resize(len);
                odk_st[tid].resize(indexPartNum + 1);

                // clear subquery oneHashValues  hashValues
                for (int pid = 0; pid < indexPartNum; ++pid) {
                    subquery[tid][pid].clear();
                    p_keys[tid][pid] = 0;
                }

                // allocate the tokens into subquery and hash each part
                // That is the way how the query is splitted into indexPartNum
                for (auto &token : query_dataset[rid]) {
                    ui pid = token % indexPartNum;
                    subquery[tid][pid].push_back(token);
                    p_keys[tid][pid] = PRIME * p_keys[tid][pid] + token + 1;
                }

                TokenLen pos = 0;
                TokenLen tmp_cnt = 0;
                for (int pid = 0; pid < indexPartNum; ++pid) {
                    // int64_t lenPart = indexLen + pid * (work_maxSize + 1);
                    ui mhv = 0, hv = p_keys[tid][pid];
                    auto &subrec = subquery[tid][pid];

                    // we store the keys into one array for each record.
                    // so we need to know the boundary of keys for onedeletions of each partition
                    odk_st[tid][pid] = tmp_cnt;
                    for (ui idx = 0; idx < subrec.size(); idx++) {
                        ui chv = hv + mhv * prime_exp[subrec.size() - 1 - idx];
                        mhv = mhv * PRIME + subrec[idx] + 1;
                        chv -= mhv * prime_exp[subrec.size() - 1 - idx];

                        // get the keys of one-deletion neighbors from the current partition
                        // auto odkey = PACK(lenPart, chv);
                        od_keys[tid][tmp_cnt++] = chv;
                    }
                    pos += subrec.size();
                }
                odk_st[tid][indexPartNum] = tmp_cnt;

                hashInFind_cost[tid] += SetJoinParallelUtil::repTime(hash_st);

                // Now all the information we have got
                // cout << "here" << endl << flush;
                GreedyFindCandidateAndSimPairs(tid, indexLenGrp, rid, query_dataset[rid].size(),
											   p_keys[tid], od_keys[tid], odk_st[tid]);
            } else {
                // We have calculated them if current indexLenGroup is the group this record belongs to the
                const auto &existing_p_keys = partsKeysQuery[rid];
                const auto &existing_od_keys = oneDeleteKeysQuery[rid];
                const auto &existing_odk_st = oDKeysStQuery[rid];

                GreedyFindCandidateAndSimPairs(tid, indexLenGrp, rid, query_dataset[rid].size(), 
				 							   existing_p_keys, existing_od_keys, existing_odk_st);
            }
        }
    }

    // handle empty
    for(ui i = 0; i < work_n; i++) {
        if(work_dataset[i].size() > 0)
            break;
        workEmpty.emplace_back(i);
    }
    for(ui i = 0; i < query_n; i++) {
        if(query_dataset[i].size() > 0)
            break;
        queryEmpty.emplace_back(i);
    }

    // fprintf(stderr, "%lu %lu %lu \n", resultNum, candidateNum, listlens);
    search_cost = SetJoinParallelUtil::repTime(find_st);
    std::cout << "Finding Similar Pairs Time Cost: " << search_cost << std::endl;

    // release memory
    delete[] prime_exp;
    delete[] invertedIndex.parts_rids;
    delete[] invertedIndex.ods_rids;
    delete[] invertedIndex.parts_index_hv;
    delete[] invertedIndex.od_index_hv;
    delete[] invertedIndex.parts_index_offset;
    delete[] invertedIndex.od_index_offset;
    delete[] invertedIndex.parts_index_cnt;
    delete[] invertedIndex.od_index_cnt;

    for (int i = 0; i < MAXTHREADNUM; ++i) {
        delete[] quickRef2D[i];
        delete[] negRef2D[i];
    }
    delete[] quickRef2D;
    delete[] negRef2D;
}