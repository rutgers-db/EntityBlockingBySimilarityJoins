/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/ovlpjoin_parallel.h"


/*
 * Util
 */

// build heap for combination_p1
bool OvlpRSJoinParallel::build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				 			 	int L, std::vector<int> &heap, std::vector<combination_p1> &combs, int &heap_size, int tid) 
{

	// get the number of large sets in the inverted list
	// int size = distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, comp_pair));
	// cause they are all treat as small sets
	int size = 0;

	// return false if there is no small set
	if ((int)vec.size() < size + 1) return false;  

	// initialize heap and combination_tests
	heap.clear();
	combs.clear();
	for (auto i = size; i < (int)vec.size(); i++) {
		// remove if there are not enough (>= c) elements left
		if ((int)(dataset[vec[i].first].size()) - 1 - vec[i].second < c) continue;
		heap.push_back(heap_size++);
		combs.push_back(combination_p1(vec[i].first, vec[i].second, *this));
	}

	if (heap_size == 0) return false;

	make_heap(heap.begin(), heap.end(), std::bind(&OvlpRSJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));

	return true;
}


// build heap for combination_p2
bool OvlpRSJoinParallel::build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				 			 	int L, std::vector<int> &heap, std::vector<combination_p2> &combs, int &heap_size, int tid) 
{
	// get the number of large sets in the inverted list
	// int size = distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, comp_pair));
	// cause they are all treat as small sets
	int size = 0;

	// return false if there is no small set
	if ((int)vec.size() < size + 1) return false;  

	// initialize heap and combination_tests
	heap.clear();
	combs.clear();
	for (auto i = size; i < (int)vec.size(); i++) {
		// remove if there are not enough (>= c) elements left
		if ((int)(dataset[vec[i].first].size()) - 1 - vec[i].second < c) continue;
		heap.push_back(heap_size++);
		combs.push_back(combination_p2(vec[i].first, vec[i].second, *this));
	}

	if (heap_size == 0) return false;

	make_heap(heap.begin(), heap.end(), std::bind(&OvlpRSJoinParallel::comp_comb2, this, std::placeholders::_1, std::placeholders::_2, tid));
	
	return true;
}


void OvlpUtilParallel::removeShort(const std::vector<std::vector<ui>> &records, std::unordered_map<ui, std::vector<int>> &ele, 
							 	   const OvlpRSJoinParallel &joiner) 
{
	for (int i = 0; i < (int)records.size(); i++) {
		if ((int)records[i].size() < joiner.c) 
			continue;
		for (ui j = 0; j < records[i].size(); j++)
			ele[records[i][j]].push_back(i);
	}
}


// Remove "widows" from a hash map based on another hash map.
// This function removes key-value pairs from the unordered_map 'ele' 
// if the key doesn't exist in another unordered_map 'ele_other'.
void OvlpUtilParallel::removeWidow(std::unordered_map<ui, std::vector<int>> &ele, const std::unordered_map<ui, std::vector<int>> &ele_other) 
{
	// Start an iterator at the beginning of 'ele'
	auto eit = ele.begin();
	
	// Iterate over the entire 'ele' unordered_map
	while (eit != ele.end()) {
		
		// If the current key doesn't exist in 'ele_other'
		if (ele_other.find(eit->first) == ele_other.end())
		
		// If the key is not found in 'ele_other', erase it from 'ele'
		eit = ele.erase(eit);
		else
		// If the key is found in 'ele_other', continue to the next key-value pair in 'ele'
		++eit;
	}
}


void OvlpUtilParallel::transform(std::unordered_map<ui, std::vector<int>> &ele, const std::vector<std::pair<int, int>> &eles, 
               					 std::vector<std::pair<int, int>> &idmap, std::vector<std::vector<std::pair<int, int>>> &ele_lists,
               					 std::vector<std::vector<ui>> &dataset, const ui total_eles, const int n, const OvlpRSJoinParallel &joiner) 
{
	dataset.resize(n);

	// the numbers in dataset is from large to small
	// the frequency in dataset is from small to large
	for (ui i = 0; i < eles.size(); ++i) {
		for (auto j = ele[eles[i].first].begin(); j != ele[eles[i].first].end(); j++)
			dataset[*j].push_back(total_eles - i - 1);
	}

	for (auto i = 0; i < n; i++)
		if ((int)dataset[i].size() < joiner.c) 
			dataset[i].clear();

	for (auto i = 0; i < n; i++)
		idmap.emplace_back(i, dataset[i].size());

	sort(idmap.begin(), idmap.end(), [] (const std::pair<int, int> & a, const std::pair<int, int> & b) {
		return a.second > b.second;
	});
	sort(dataset.begin(), dataset.end(), [] (const std::vector<ui>& a, const std::vector<ui>& b) {
		return a.size() > b.size();
	});
	std::cout << " largest set: " << dataset.front().size() << " smallest set: " << dataset.back().size() << std::endl;

	// build index
	ele_lists.resize(total_eles);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < (int)dataset[i].size(); j++)
			ele_lists[dataset[i][j]].emplace_back(i, j);

	std::cout << "Build ele_list" << std::endl;
	// cout << flush;
}




/*
 * Join
 */
void OvlpRSJoinParallel::small_case(int L1, int R1, int L2, int R2, std::vector<std::pair<int, int>> &finalPairs) 
{
	if (L1 >= R1) return;
	if (L2 >= R2) return;
	--c;

	timeval beg, mid, mid1, end;
	gettimeofday(&beg, NULL);

	std::cout << " number of small sets: " << R1 - L1 << " and " << R2 - L2 << std::endl;

	std::vector<std::vector<int>> res_lists1[MAXTHREADNUM];
	std::vector<std::vector<int>> res_lists2[MAXTHREADNUM];

	gettimeofday(&mid, NULL);

	// FILE *fp = fopen("buffer/heap_size.txt", "w");

	// int id1 = 0; 
	// int id2 = 0;
	int64_t turn = 0;
	int64_t buildTurn = 0;

#pragma omp parallel for
	for (int idx = total_eles - 1; idx >= 0; idx--) {

		int heap_size1 = 0;
		int heap_size2 = 0;
		ui totalZero = 0;

		int tid = omp_get_thread_num();
		auto &curHeap1 = heap1[tid];
		auto &curHeap2 = heap2[tid];
		auto &curCombs1 = combs1[tid];
		auto &curCombs2 = combs2[tid];

		if (!build_heap(ele_lists1[idx], datasets1, L1, curHeap1, curCombs1, heap_size1, tid)) continue;
		if (!build_heap(ele_lists2[idx], datasets2, L2, curHeap2, curCombs2, heap_size2, tid)) continue;
		++ buildTurn;

		bool pop1 = true;
		bool pop2 = true;

		// pop heaps
		// heap1: S, heap2: R
		std::vector<int> inv_list1;
		std::vector<int> inv_list2;

		do {
			// bool found1 = false;
			// bool found2 = false;
			turn ++;

			if (pop1) pop_heap(curHeap1.begin(), curHeap1.begin() + heap_size1--, std::bind(&OvlpRSJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));
			else  pop1 = true;
			if (pop2) pop_heap(curHeap2.begin(), curHeap2.begin() + heap_size2--, std::bind(&OvlpRSJoinParallel::comp_comb2, this, std::placeholders::_1, std::placeholders::_2, tid));
			else  pop2 = true;

			switch (OvlpUtilParallel::compare(curCombs1[curHeap1[heap_size1]], curCombs2[curHeap2[heap_size2]], *this)) {
				case 1 : {	
#if BRUTEFORCE_COMB == 0
					curCombs1[curHeap1[heap_size1]].binary(curCombs2[curHeap2[heap_size2]], *this);
#elif BRUTEFORCE_COMB == 1
					curCombs1[curHeap1[heap_size1]].next(*this);
#endif
					if (curCombs1[curHeap1[heap_size1]].completed) {
						curHeap1.pop_back();
					} else {
						std::push_heap(curHeap1.begin(), curHeap1.end(), std::bind(&OvlpRSJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));
						++heap_size1;
					}
					pop2 = false;
					break;
				}

				case -1 : {
#if BRUTEFORCE_COMB == 0
					curCombs2[curHeap2[heap_size2]].binary(curCombs1[curHeap1[heap_size1]], *this);
#elif BRUTEFORCE_COMB == 1
					curCombs2[curHeap2[heap_size2]].next(*this);
#endif
					if (curCombs2[curHeap2[heap_size2]].completed) {
						curHeap2.pop_back();
						// if(turn == 34) cout << "pop here" << endl;
					} else {
						std::push_heap(curHeap2.begin(), curHeap2.end(), std::bind(&OvlpRSJoinParallel::comp_comb2, this, std::placeholders::_1, std::placeholders::_2, tid));
						++heap_size2;
					}
					pop1 = false;
					break;
				}

				case 0 : {
					inv_list1.clear();
					inv_list2.clear();

					totalZero ++;

					inv_list1.push_back(curCombs1[curHeap1[heap_size1]].id);
		
					while (heap_size1 > 0 && OvlpUtilParallel::is_equal(curCombs1[curHeap1[heap_size1]], curCombs1[curHeap1.front()], *this)) {
						inv_list1.push_back(curCombs1[curHeap1.front()].id);
						pop_heap(curHeap1.begin(), curHeap1.begin() + heap_size1, std::bind(&OvlpRSJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));
						--heap_size1;
					} 

					inv_list2.push_back(curCombs2[curHeap2[heap_size2]].id);
	
					while (heap_size2 > 0 && OvlpUtilParallel::is_equal(curCombs2[curHeap2[heap_size2]], curCombs2[curHeap2.front()], *this)) {
						inv_list2.push_back(curCombs2[curHeap2.front()].id);
						pop_heap(curHeap2.begin(), curHeap2.begin() + heap_size2, std::bind(&OvlpRSJoinParallel::comp_comb2, this, std::placeholders::_1, std::placeholders::_2, tid));
						--heap_size2;
					} 

#if LIMIT_INV_SIZE == 1
					if (inv_list1.size() > MAX_INV_SIZE)
						inv_list1.resize(MAX_INV_SIZE);
					if (inv_list2.size() > MAX_INV_SIZE)
						inv_list2.resize(MAX_INV_SIZE);
#endif

					res_lists1[tid].push_back(std::move(inv_list1));
					res_lists2[tid].push_back(std::move(inv_list2));

					if (heap_size1 == 0 && heap_size2 == 0) break;

					for (auto i = heap_size1; i < (int)curHeap1.size(); ++i) {
#if REPORT_BINARY == 1
						printf("Before binary\n");
						curCombs1[curHeap1[i]].print(*this);
#endif
#if BRUTEFORCE_COMB == 0
						curCombs1[curHeap1[i]].binary(curCombs2[curHeap2.front()], *this);
#elif BRUTEFORCE_COMB == 1
						curCombs1[curHeap1[i]].next(*this);
#endif
#if REPORT_BINARY == 1
						printf("After binary\n");
						curCombs1[curHeap1[i]].print(*this);
#endif
					}

					for (auto i = heap_size2; i < (int)curHeap2.size(); ++i) {
#if REPORT_BINARY == 1
						printf("Before binary\n");
						curCombs2[curHeap2[i]].print(*this);
#endif
#if BRUTEFORCE_COMB == 0
						curCombs2[curHeap2[i]].binary(curCombs1[curHeap1.front()], *this);
#elif BRUTEFORCE_COMB == 1
						curCombs2[curHeap2[i]].next(*this);
#endif
#if REPORT_BINARY == 1
						printf("After binary\n");
						curCombs2[curHeap2[i]].print(*this);
#endif
					}

					int comp_num1 = 0;
					for (auto i = heap_size1; i < (int)curHeap1.size(); ++i) {
						if (curCombs1[curHeap1[i]].completed)
						++comp_num1;
						else if (comp_num1 > 0)
						curHeap1[i - comp_num1] = curHeap1[i];
					}

					int comp_num2 = 0;
					for (auto i = heap_size2; i < (int)curHeap2.size(); ++i) {
						if (curCombs2[curHeap2[i]].completed)
						++comp_num2;
						else if (comp_num2 > 0)
						curHeap2[i - comp_num2] = curHeap2[i];
					}

					for (auto i = heap_size1; i < (int)curHeap1.size() - comp_num1; i++) {
						push_heap(curHeap1.begin(), curHeap1.begin() + i + 1, std::bind(&OvlpRSJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));
					}

					for (auto i = heap_size2; i < (int)curHeap2.size() - comp_num2; i++) {
						push_heap(curHeap2.begin(), curHeap2.begin() + i + 1, std::bind(&OvlpRSJoinParallel::comp_comb2, this, std::placeholders::_1, std::placeholders::_2, tid));
					}

					while (comp_num1-- > 0)
						curHeap1.pop_back();
					heap_size1 = curHeap1.size();

					while (comp_num2-- > 0)
						curHeap2.pop_back();
					heap_size2 = curHeap2.size();
					break;
				}
			}
		} while ((heap_size1 > 0 || (heap_size1 >= 0 && !pop1)) && (heap_size2 > 0 || (heap_size2 >= 0 && !pop2)));

		// fprintf(fp, "%d\n", totalZero);
	}
	// fclose(fp);

	// cout << "Res lists num: " << res_lists1.size() << " " << res_lists2.size() << endl;
	// fflush(stdout);

	gettimeofday(&mid1, NULL);

#pragma omp parallel for
	for(ui tid = 0; tid < MAXTHREADNUM; tid++) {
		std::vector<std::vector<int>> id_lists(n1);
		for (auto i = 0; i < (int)res_lists1[tid].size(); i++) {
			for (auto j = 0; j < (int)res_lists1[tid][i].size(); j++)
				id_lists[res_lists1[tid][i][j]].push_back(i);
		}
		// cout << "ID list num: " << id_lists.size() << endl;

		auto &cur_result_pairs = result_pairs[tid];
		auto &cur_result_pairs_ = result_pairs_[tid];

		std::vector<int> results(n2, -1);
		for (auto i = n1 - 1; i >= 0; i--) {
			if (id_lists[i].empty()) 
				continue;
#if MAINTAIN_VALUE_OVLP == 0
			if(earlyTerminated[tid] == 1)
				break;
#endif

			for (auto j = 0; j < (int)id_lists[i].size(); j++) {
				for (auto k = 0; k < (int)res_lists2[tid][id_lists[i][j]].size(); k++) {
					if (results[res_lists2[tid][id_lists[i][j]][k]] != i) {
						int idd1 = idmap_records1[i].first;
						int idd2 = idmap_records2[res_lists2[tid][id_lists[i][j]][k]].first;
#if MAINTAIN_VALUE_OVLP == 0
						cur_result_pairs.emplace_back(idd1, idd2);
#elif MAINTAIN_VALUE_OVLP == 1
						double val = isWeightedComp ? weightedOverlapCoeff(idd1, idd2) 
													: overlapCoeff(idd1, idd2);
						if(cur_result_pairs_.size() < maxHeapSize) {
							cur_result_pairs_.emplace_back(idd1, idd2, val);
						}
						else {
							if(isHeap[tid] == 0) {
								std::make_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
								isHeap[tid] = 1;
							}
							
							if(cur_result_pairs_[0].val < val) {
								std::pop_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
								cur_result_pairs_.pop_back();
								cur_result_pairs_.emplace_back(idd1, idd2, val);
								std::push_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
							}
						}
#endif
						// result_pairs[tid].emplace_back(idmap_records1[i].first, idmap_records2[res_lists2[tid][id_lists[i][j]][k]].first);
						results[res_lists2[tid][id_lists[i][j]][k]] = i;
						// ++result_num;
					}
				}
#if MAINTAIN_VALUE_OVLP == 0
				if(result_pairs[tid].size() >= maxHeapSize) {
					earlyTerminated[tid] = 1;
					break;
				}
#endif
			}
		}
	} 

#if MAINTAIN_VALUE_OVLP == 0
	for(ui tid = 0; tid < MAXTHREADNUM; tid++)
		finalPairs.insert(finalPairs.end(), result_pairs[tid].begin(), result_pairs[tid].end());
#elif MAINTAIN_VALUE_OVLP == 1
	for(int i = 0; i < MAXTHREADNUM; i++)
		for(const auto &wp : result_pairs_[i])
			finalPairs.emplace_back(wp.id1, wp.id2);
#endif

	// deduplicate
#if DEDUPLICATE == 1
	sort(finalPairs.begin(), finalPairs.end());
	auto iter = unique(finalPairs.begin(), finalPairs.end());
	if(iter != finalPairs.end()) {
		std::cerr << "Duplicate pairs detected" << std::endl;
		finalPairs.erase(iter, finalPairs.end());
		// exit(1);
	}
#endif

	++c;
	gettimeofday(&end, NULL);
	std::cout << " small p1 : " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << std::endl;
	std::cout << " small p2 : " << mid1.tv_sec - mid.tv_sec + (mid1.tv_usec - mid.tv_usec) / 1e6 << std::endl;
	std::cout << " small p3 : " << end.tv_sec - mid1.tv_sec + (end.tv_usec - mid1.tv_usec) / 1e6 << std::endl;
	fflush(stdout);
}



void OvlpRSJoinParallel::overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs)
{
	srand(time(NULL));
	
	timeval starting, ending, s1, s2, t2;
	timeval time1, time3, time4;
#if PREPROCESS_TIMER_ON == 1
	timeval removeStart, widowStart, transformStart;
	timeval removeEnd, widowEnd, transformEnd;
#endif

	gettimeofday(&starting, NULL);

	c = overlap_threshold;           // get threshold
	n1 = records1.size();
	n2 = records2.size();
	
	std::unordered_map<ui, std::vector<int>> ele1;
	std::unordered_map<ui, std::vector<int>> ele2;

	gettimeofday(&time1, NULL);
#if PREPROCESS_TIMER_ON == 1
	gettimeofday(&removeStart, NULL);
#endif
	OvlpUtilParallel::removeShort(records1, ele1, *this);
	OvlpUtilParallel::removeShort(records2, ele2, *this);
#if PREPROCESS_TIMER_ON == 1
	gettimeofday(&removeEnd, NULL);
#endif

#if BRUTEFORCE_COMB == 0
	// removeWidow(ele1, ele2); // we dont need to remove widow in ele1 cause they all from joiner.records2
#if PREPROCESS_TIMER_ON == 1
	gettimeofday(&widowStart, NULL);
#endif
	OvlpUtilParallel::removeWidow(ele2, ele1);
#if PREPROCESS_TIMER_ON == 1
	gettimeofday(&widowEnd, NULL);
#endif
#endif

	std::vector<std::pair<int, int>> eles;
	for (auto it = ele1.begin(); it != ele1.end(); it++)
		eles.emplace_back(it->first, it->second.size() + ele2[it->first].size());

	// frequency increasing order
	sort(eles.begin(), eles.end(), [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
		return p1.second < p2.second;
	});

	// the number of elements (tokens) cannot exceed the maximum of ui
	assert(eles.size() <= 4294967295);
	total_eles = eles.size();

#if PREPROCESS_TIMER_ON == 1
	gettimeofday(&transformStart, NULL);
#endif
	OvlpUtilParallel::transform(ele1, eles, idmap_records1, ele_lists1, datasets1, total_eles, n1, *this);
	OvlpUtilParallel::transform(ele2, eles, idmap_records2, ele_lists2, datasets2, total_eles, n2, *this);
#if PREPROCESS_TIMER_ON == 1
	gettimeofday(&transformEnd, NULL);
#endif

	gettimeofday(&time3, NULL);
	std::cout << "Transform Time: " << time3.tv_sec - time1.tv_sec + (time3.tv_usec - time1.tv_usec) / 1e6 << std::endl;
#if PREPROCESS_TIMER_ON == 1
	double removeTime = removeEnd.tv_sec - removeStart.tv_sec + (removeEnd.tv_usec - removeStart.tv_usec) / 1e6;
	double widowTime = widowEnd.tv_sec - widowStart.tv_sec + (widowEnd.tv_usec - widowStart.tv_usec) / 1e6;
	double transformTime = transformEnd.tv_sec - transformStart.tv_sec + (transformEnd.tv_usec - transformStart.tv_usec) / 1e6;
	printf("%.2lf\t%.2lf\t%.2lf\n", removeTime, widowTime, transformTime);
#endif

#if REPORT_INDEX == 1
	printf("###          eles's size: %u\n", total_eles);
	printf("###     ele_list1's size: %u\n", ele_lists1.size());
	printf("###     ele_list2's size: %u\n", ele_lists2.size());
	printf("###     datasets1's size: %u\n", datasets1.size());
	printf("###     datasets2's size: %u\n", datasets2.size());
#endif

	gettimeofday(&time4, NULL);

	// ****** conduct joining ******  
	result_num = 0;
	candidate_num= 0;

	gettimeofday(&s1, NULL);

	gettimeofday(&s2, NULL);
	small_case(0, n1, 0, n2, finalPairs);
	gettimeofday(&t2, NULL);

	gettimeofday(&ending, NULL);
	std::cout << "Join Time: " << ending.tv_sec - time4.tv_sec + (ending.tv_usec - time4.tv_usec) / 1e6 << std::endl;
	std::cout << "  small Time: " << t2.tv_sec - s2.tv_sec + (t2.tv_usec - s2.tv_usec) / 1e6 << std::endl;
	std::cout << "All Time: " << ending.tv_sec - starting.tv_sec + (ending.tv_usec - starting.tv_usec) / 1e6 << std::endl;
	std::cout << "Result Num: " << result_num << std::endl;
}


/*
 * Self
 */
bool OvlpSelfJoinParallel::build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				 			 	  int L, std::vector<int> &heap, std::vector<combination_p1> &combs, int &heap_size, int tid) 
{

	// get the number of large sets in the inverted list
	int size = std::distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, OvlpUtilParallel::comp_pair));
	// cause they are all treat as small sets
	// int size = 0;

	// return false if there is no small set
	if ((int)vec.size() < size + 1) return false;  

	// initialize heap and combination_tests
	heap.clear();
	combs.clear();
	for (auto i = size; i < (int)vec.size(); i++) {
		// remove if there are not enough (>= c) elements left
		if ((int)(dataset[vec[i].first].size()) - 1 - vec[i].second < c) continue;
		heap.push_back(heap_size++);
		combs.emplace_back(vec[i].first, vec[i].second, *this);
	}

	if (heap_size == 0) return false;

	make_heap(heap.begin(), heap.end(), std::bind(&OvlpSelfJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));

	return true;
}


void OvlpSelfJoinParallel::small_case(int L, int R, std::vector<std::pair<int, int>> &finalPairs) 
{
    --c;

    timeval mid, mid1, end;
    std::vector<std::vector<int>> res_lists[MAXTHREADNUM];
	std::vector<bool> resterminated(MAXTHREADNUM, false);
    
    // cout << "Running threads" << omp_get_num_threads() << endl;
    
    gettimeofday(&mid, NULL);
#pragma omp parallel for schedule(dynamic)
    for (int idx = total_eles - 1; idx >= 0; idx--) {
        if (ele_lists[idx].size() < 2) {
			// cout << ele_lists[idx].size() << endl;
            continue;
		}

        int tid = omp_get_thread_num();
		int heap_size = 0;

		auto &curCombs = combs[tid];
		auto &curHeap = heap[tid];
		auto &curEleList = ele_lists[idx]; 
		auto &curResList = res_lists[tid];

		if (!build_heap(curEleList, datasets, L, curHeap, curCombs, heap_size, tid) || resterminated[tid]) {
			// cout << heap_size << endl;
			continue;
		}

        if (heap_size < 2)
            continue;

        std::vector<int> inv_list;

		// pop heaps
		// cout << heap_size << endl;
        while (heap_size > 1) {
            inv_list.clear();

            do {
                pop_heap(curHeap.begin(), curHeap.begin() + heap_size, std::bind(&OvlpSelfJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));
                --heap_size;
                inv_list.emplace_back(curCombs[curHeap[heap_size]].id);
            } while (heap_size > 0 && OvlpUtilParallel::is_equal(curCombs[curHeap[heap_size]], curCombs[curHeap.front()], *this));

#if LIMIT_INV_SIZE == 1
			if (inv_list.size() > MAX_INV_SIZE)
				inv_list.resize(MAX_INV_SIZE);
#endif

            if (inv_list.size() > 1) 
                curResList.push_back(std::move(inv_list));

            if (heap_size == 0)
                break;

            for (auto i = heap_size; i < (int)curHeap.size(); ++i) 
                curCombs[curHeap[i]].binary(curCombs[curHeap.front()], *this);

            int comp_num = 0;
            for (auto i = heap_size; i < (int)curHeap.size(); ++i) {
                if (curCombs[curHeap[i]].completed)
                    ++comp_num;
                else if (comp_num > 0)
                    curHeap[i - comp_num] = curHeap[i];
            }

            for (auto i = heap_size; i < (int)curHeap.size() - comp_num; i++)
                push_heap(curHeap.begin(), curHeap.begin() + i + 1, std::bind(&OvlpSelfJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, tid));

            while (comp_num-- > 0)
               	curHeap.pop_back();
            heap_size = curHeap.size();
        }
    }

    gettimeofday(&mid1, NULL);

	// get results
#if REPORT_LIST == 1
	FILE *listFp = fopen("buffer/ovlp_inv_list.txt", "w");
#endif
	int candidateNum[MAXTHREADNUM] = {0};
	int resultNum[MAXTHREADNUM] = {0};
	int listNum = 0;
	uint64_t totalNum = 0;
	for(int tid = 0; tid < MAXTHREADNUM; tid++) {
		listNum += res_lists[tid].size();
		for(const auto & invList : res_lists[tid]) {
			totalNum += invList.size();
#if REPORT_LIST == 1
			for(const auto & entity : invList)
				fprintf(listFp, "%d ", entity);
			fprintf(listFp, "\n");
#endif
		}
	}
	std::cout << "List num: " << listNum << " " << totalNum << std::endl;
#if REPORT_LIST == 1
	fclose(listFp);
#endif

	int **quickRef = new int*[MAXTHREADNUM];
	for(int tid = 0; tid < MAXTHREADNUM; tid++) {
		quickRef[tid] = new int[n1];
		std::fill(quickRef[tid], quickRef[tid] + n1, -1);
	}

#pragma omp parallel for
	for(int tid = 0; tid < MAXTHREADNUM; tid++) {
		std::vector<std::vector<int>> id_lists(n1);
		auto &curResList = res_lists[tid];
		ui curResListSize = curResList.size();

		// printf("Working on: %d %d\n", tid, (int)curResListSize);

		for (ui i = 0; i < curResListSize; i++) 
			for (auto item : curResList[i])
				id_lists[item].emplace_back(i);

		auto &curRef = quickRef[tid];
#if MAINTAIN_VALUE_OVLP == 0
		auto &cur_result_pairs = result_pairs[tid];
#elif MAINTAIN_VALUE_OVLP == 1
		auto &cur_result_pairs_ = result_pairs_[tid];
#endif

		for (int i = n1 - 1; i >= 0; i--) {
			if (id_lists[i].empty())
				continue;
#if MAINTAIN_VALUE_OVLP == 0
			if(earlyTerminated[tid] == 1)
				break;
#endif

			int idd1 = idmap_records[i].first;
			
			for (auto curListIdx : id_lists[i]) {
				curResList[curListIdx].pop_back();

				for (auto item : curResList[curListIdx]) {
					if (curRef[item] != i) {
						curRef[item] = i;

						int idd2 = idmap_records[item].first;
						assert(idd1 != idd2);
#if APPROXIMATE_OVLP == 1
						bool filter = false;
						for(int sharing = 0; sharing < SHARING_PREFIX; sharing++) {
							if(records[idd1][0] != records[idd2][0]) {
								filter = true;
								break;
							}
						}
						if(filter)
							continue;
#endif

#if MAINTAIN_VALUE_OVLP == 0
						if(idd1 < idd2)
							cur_result_pairs.emplace_back(idd1, idd2);
						else
							cur_result_pairs.emplace_back(idd2, idd1);
#elif MAINTAIN_VALUE_OVLP == 1
						double val = isWeightedComp ? weightedOverlapCoeff(idd1, idd2) 
													: overlapCoeff(idd1, idd2);
						if(cur_result_pairs_.size() < maxHeapSize) {
							if(idd1 < idd2) cur_result_pairs_.emplace_back(idd1, idd2, val);
							else cur_result_pairs_.emplace_back(idd2, idd1, val);
						}
						else {
							if(isHeap[tid] == 0) {
								make_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
								isHeap[tid] = 1;
							}
							
							if(cur_result_pairs_[0].val < val) {
								pop_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
								cur_result_pairs_.pop_back();
								if(idd1 < idd2) cur_result_pairs_.emplace_back(idd1, idd2, val);
								else cur_result_pairs_.emplace_back(idd2, idd1, val);
								push_heap(cur_result_pairs_.begin(), cur_result_pairs_.end());
							}
						}
#endif
						// ++ resultNum[tid];
						// ++ candidateNum[tid];
					}
				}

#if MAINTAIN_VALUE_OVLP == 0
				if(result_pairs[tid].size() >= maxHeapSize) {
					earlyTerminated[tid] = 1;
					break;
				}
#endif
			}
		}
	}
	// cout << "finished" << endl << flush;

	for(int tid = 0; tid < MAXTHREADNUM; tid++)
		delete[] quickRef[tid];
	delete[] quickRef;

	// merge
	for(int i = 0; i < MAXTHREADNUM; i++) {
#if MAINTAIN_VALUE_OVLP == 0
	 	finalPairs.insert(finalPairs.end(), result_pairs[i].begin(), result_pairs[i].end());
#elif MAINTAIN_VALUE_OVLP == 1
		for(const auto &wp : result_pairs_[i])
			finalPairs.emplace_back(wp.id1, wp.id2);
#endif
		candidate_num += candidateNum[i];
		result_num += resultNum[i];
	}

	// deduplicate
#if DEDUPLICATE == 1
	sort(finalPairs.begin(), finalPairs.end());
	auto iter = unique(finalPairs.begin(), finalPairs.end());
	if(iter != finalPairs.end()) {
		std::cerr << "Duplicate pairs detected" << std::endl;
		finalPairs.erase(iter, finalPairs.end());
		// exit(1);
	}
#endif

    ++c;

    std::cout << "candidate number: " << candidate_num << std::endl;
	std::cout << "result number: " << result_num << std::endl;
    gettimeofday(&end, NULL);
    std::cout << " small p2 : " << mid1.tv_sec - mid.tv_sec + (mid1.tv_usec - mid.tv_usec) / 1e6 << std::endl;
    std::cout << " small p3 : " << end.tv_sec - mid1.tv_sec + (end.tv_usec - mid1.tv_usec) / 1e6 << std::endl << std::flush;
}


int64_t OvlpSelfJoinParallel::small_estimate(int L, int R)
{
	if (L >= R) 
		return 0;
  
	timeval beg, mid, end;
	gettimeofday(&beg, NULL);

	int total_num = R - L;
	int sample_time = (R - L);
	double ratio =  (total_num - 1) * 1.0 / sample_time * total_num / 2;
	// cout << "sample ratio: " << ratio << endl;
	int r1, r2;
	int64_t pair_num = 0;
	for (auto i = 0; i < sample_time; i++) {
		do {
			r1 = rand() % (R - L) + L;
			r2 = rand() % (R - L) + L;
		} while(r1 == r2);
		int start1 = 0;
		int start2 = 0;
		int overlap = 0;
		while (start1 < (int)datasets[r1].size() && start2 < (int)datasets[r2].size()) {
			if (datasets[r1][start1] == datasets[r2][start2]) {
				++start1, ++start2;
				overlap++;
			} else {
				if (datasets[r1][start1] > datasets[r2][start2]) ++start1;
				else ++start2;
			}
		}
		if (overlap >= c){
			// cout << overlap << " " << c << endl;
			pair_num += OvlpUtilParallel::nchoosek(overlap, c);
			// cout << pair_num << endl;
		}
	}
	list_cost = pair_num * ratio;

	--c;

	heap_cost = 0;
	binary_cost = 0;

	gettimeofday(&mid, NULL);

	auto &sample_heap = heap[0];
	auto &sample_combs = combs[0];

	for(auto sit = random_ids.begin(); sit != random_ids.end(); ++sit) {
		auto idx = *sit;

		std::vector<std::pair<int,int>> & vec = ele_lists[idx];
		int size = std::distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, OvlpUtilParallel::comp_pair));
		if ((int)vec.size() <= size + 1) continue;

		sample_heap.clear();
		sample_combs.clear();
		int heap_size = 0;
		for (auto i = size; i < (int)vec.size(); i++) {
			if ((int)(datasets[vec[i].first].size()) - 1 - vec[i].second < c) continue;
			sample_heap.push_back(heap_size++);
			sample_combs.emplace_back(vec[i].first, vec[i].second, *this);
		}

		if (heap_size < 2) 
			continue;

		std::make_heap(sample_heap.begin(), sample_heap.end(), std::bind(&OvlpSelfJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, 0));
		heap_cost += (3 * c * heap_size);

		while (heap_size > 1) {
			do {
				++heap_op;
				heap_cost += (c * log2(heap_size) + c);
				std::pop_heap(sample_heap.begin(), sample_heap.begin() + heap_size, std::bind(&OvlpSelfJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, 0));
				--heap_size;
			} while (heap_size > 0 && OvlpUtilParallel::is_equal(sample_combs[sample_heap[heap_size]], sample_combs[sample_heap.front()], *this));

			if (heap_size == 0) break;

			for (auto i = heap_size; i < (int)sample_heap.size(); ++i) {
				sample_combs[sample_heap[i]].binary(sample_combs[sample_heap.front()], *this);
				binary_cost += (c * log2(datasets[sample_combs[sample_heap[i]].id].size()));
			}

			int comp_num = 0;
			for (auto i = heap_size; i < (int)sample_heap.size(); ++i) {
				if (sample_combs[sample_heap[i]].completed)
				++comp_num;
				else if (comp_num > 0)
				sample_heap[i - comp_num] = sample_heap[i];
			}

			for (auto i = heap_size; i < (int)sample_heap.size() - comp_num; i++) {
				std::push_heap(sample_heap.begin(), sample_heap.begin() + i + 1, std::bind(&OvlpSelfJoinParallel::comp_comb1, this, std::placeholders::_1, std::placeholders::_2, 0));
				heap_cost += (c * log2(i));
			}
			while (comp_num-- > 0)
				sample_heap.pop_back();
			heap_size = sample_heap.size();
		}
	}

	++c;

	sample_heap.clear();
	sample_combs.clear();

	gettimeofday(&end, NULL);
	std::cout << " small est time p1 : " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << std::endl;
	std::cout << " small est time p2 : " << end.tv_sec - mid.tv_sec + (end.tv_usec - mid.tv_usec) / 1e6 << std::endl;
	return binary_cost * TIMES + heap_cost * TIMES + list_cost;
}


int64_t OvlpSelfJoinParallel::large_estimate(int L, int R) 
{
	timeval beg, end;
	gettimeofday(&beg, NULL);
	std::vector<int> count(total_eles);
	for (int i = n1 - 1; i >= R; --i)
	for (auto x : datasets[i])
		++count[x];

	int64_t ret = 0;
	for (int i = R - 1; i >= L; --i) {
	for (auto x : datasets[i]) {
		++count[x];
		ret += count[x];
	}
	}
	large_est_cost = ret;
	gettimeofday(&end, NULL);
	std::cout << " large est time : " << end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1e6 << std::endl;
	return ret;
}


int OvlpSelfJoinParallel::divide(int nL) 
{
	int l = 0, r = n1;
	while (l < r) {
		int m = (l + r) >> 1;
		if ((int)datasets[m].size() > nL) l = m + 1;
		else r = m;
	}
	return r;
}


int OvlpSelfJoinParallel::estimate()
{
	// get random elements for sampling
	while (random_ids.size() < total_eles * RATIO)
		random_ids.insert(rand() % total_eles);

	int64_t small, large;
	int min_size = datasets.back().size();
	int max_size = datasets.front().size();
	auto bound = (min_size <= c ? c : min_size);
	int pos = divide(bound);
	int prev_pos = pos;
	int64_t prev_large = large_estimate(0, pos);
	int64_t prev_small = small_estimate(pos, n1);
	++bound;

	for (; bound <= max_size; bound++) {

		pos = divide(bound);
		if (pos == prev_pos) 
			continue;
		// std::cout << std::endl << "size boud: " << bound << std::endl;
		// std::cout << "larg numb: " << pos << std::endl;
		// std::cout << "smal numb: " << n1 - pos << std::endl;

		large = large_estimate(0, pos);
		small = small_estimate(pos, n1);

		// std::cout << "heap cost: " << heap_cost * TIMES << std::endl;
		// std::cout << "biny cost: " << binary_cost * TIMES <<std::endl;
		// std::cout << "list cost: " << list_cost << std::endl; 
		// std::cout << "smal cost: " << small << std::endl;
		// std::cout << "larg cost: " << large << std::endl;

		if (small - prev_small > 1.2 * (prev_large - large)) return prev_pos;

		prev_pos = pos;
		prev_large = large;
		prev_small = small;
	}

	std::cout << "size bound: " << bound << std::endl;
	return prev_pos;
}


void OvlpSelfJoinParallel::large_case(int L, int R, std::vector<std::pair<int, int>> &finalPairs)
{
	timeval beg, mid, end;
	gettimeofday(&beg, NULL);

	std::vector<std::vector<int>> ele(total_eles);

	for (int i = n1 - 1; i >= R; --i)
		for (auto x : datasets[i])
			ele[x].push_back(i);

	gettimeofday(&mid, NULL);
	std::vector<int> bucket;

	for (int i = R - 1; i >= L; --i) {
		int count = 0;
		for (auto x : datasets[i]) 
			count += ele[x].size();
		large_cost += count;

		if (count > 0.2 * n1) {
			// n + count * value + count * if
			bucket.assign(n1, 0);
			for (auto x : datasets[i]) {
				for (auto id : ele[x]) {
					if (++bucket[id] == c) {
						int idd1 = idmap_records[i].first;
						int idd2 = idmap_records[id].first;
						int minId = std::min(idd1, idd2);
						int maxId = std::max(idd1, idd2);
						finalPairs.emplace_back(minId, maxId);
						// finalPairs.emplace_back(idd1, idd2);
						++result_num;
					}
				}
			}
		} 
		else {
			// count * if + count * value + count * if or value
			alive_id++;
			for (auto x : datasets[i]) {
				for (auto id : ele[x]) {
					if (buck[id].first != alive_id) {
						buck[id].first = alive_id;
						buck[id].second = 1;
					} 
					else {
						if (++buck[id].second == c) {
							int idd1 = idmap_records[i].first;
							int idd2 = idmap_records[id].first;
							int minId = std::min(idd1, idd2);
							int maxId = std::max(idd1, idd2);
							finalPairs.emplace_back(minId, maxId);
							// finalPairs.emplace_back(idd1, idd2);
							++result_num;
						}
					}
				}
			}
		}
		for (auto x : datasets[i])
			ele[x].push_back(i);
	}

	gettimeofday(&end, NULL);
	std::cout << " large p1 : " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << std::endl;
	std::cout << " large p2 : " << end.tv_sec - mid.tv_sec + (end.tv_usec - mid.tv_usec) / 1e6 << std::endl;
}


void OvlpSelfJoinParallel::overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs) 
{
    srand(time(NULL));

    timeval starting, ending, s1, t1, s2, t2;
    timeval time1, time3, time4;

    gettimeofday(&starting, NULL);

    c = overlap_threshold;           // get threshold
    n1 = records.size();              // get number of records
	buck.assign(n1, std::make_pair(0, 0));

    std::vector<std::pair<int, int>> eles;
    std::unordered_map<int, std::vector<int>> ele;

    for (int i = 0; i < (int)records.size(); i++) {
        if ((int)records[i].size() < c)
            continue;                               // remove records with size smaller than c
        for (int j = 0; j < (int)records[i].size(); j++) // build inverted index
            ele[records[i][j]].push_back(i);
    }

    for (auto it = ele.begin(); it != ele.end(); it++)
        eles.emplace_back(it->first, it->second.size()); // build element frequency table

    // get global order: frequency increasing order
    // sort the elements
    sort(eles.begin(), eles.end(), [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) { return p1.second < p2.second; });
    
    // container initialize
    datasets.resize(n1);

    // sort elements by its global order: frequence increasing order
    // remove widow word
    // encode elements in decreasing order
    // so the dataset is the same as the record, only the element is encoded to 0~ total_eles-1.
    // The encoding way is the less frequency of the element is, the large number it gets
    total_eles = eles.size();
    for (auto i = 0; i < int(eles.size()); ++i) {
        if (eles[i].second < 2)
            continue;
        for (auto j = ele[eles[i].first].begin(); j != ele[eles[i].first].end(); j++)
            datasets[*j].push_back(total_eles - i - 1);
    }

    gettimeofday(&time1, NULL);
    std::cout << "Initial Time: " << time1.tv_sec - starting.tv_sec + (time1.tv_usec - starting.tv_usec) / 1e6 << std::endl;

    // ****** cost model for prefix length selection ******
    // remove short records
    for (auto i = 0; i < n1; i++)
        if ((int)datasets[i].size() < c)
            datasets[i].clear();

    // create id mappings: from sorted to origin
    for (auto i = 0; i < n1; i++)
        idmap_records.emplace_back(i, datasets[i].size());

    // sort records by length in decreasing order
    sort(idmap_records.begin(), idmap_records.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.second > b.second; });
    // todo Writing idmap_records[idmap_records]
    sort(datasets.begin(), datasets.end(), [](const std::vector<ui> &a, const std::vector<ui> &b) { return a.size() > b.size(); });
    std::cout << " largest set: " << datasets.front().size() << " smallest set: " << datasets.back().size() << " It might be 0 cause some row in dataset, its length is smaller than c" << std::endl;

    // build real inverted index
    ele_lists.resize(total_eles);
    for (int i = 0; i < n1; i++)
        for (int j = 0; j < (int)datasets[i].size(); j++)
            ele_lists[datasets[i][j]].emplace_back(i, j);

    gettimeofday(&time3, NULL);
    std::cout << "Transform Time: " << time3.tv_sec - time1.tv_sec + (time3.tv_usec - time1.tv_usec) / 1e6 << std::endl;

    // ****** cost model for boundary selection ******
	int nL = estimate();
	// nL = 70000;
  	int nP = nL;
  	std::cout << " large sets: " << nP << " small sets: " << n1 - nP << std::endl;
    // std::cout << " All are treated as small sets: " << n1 << std::endl;

    gettimeofday(&time4, NULL);
    // ****** conduct joining ******
    result_num = 0;

    gettimeofday(&s1, NULL);
	large_case(0, nP, finalPairs);
    gettimeofday(&t1, NULL);

    gettimeofday(&s2, NULL);
    small_case(nP, n1, finalPairs);
    gettimeofday(&t2, NULL);

    gettimeofday(&ending, NULL);
    std::cout << "Join Time: " << ending.tv_sec - time4.tv_sec + (ending.tv_usec - time4.tv_usec) / 1e6 << std::endl;
    std::cout << "  small Time: " << t2.tv_sec - s2.tv_sec + (t2.tv_usec - s2.tv_usec) / 1e6 << std::endl;
    std::cout << "All Time: " << ending.tv_sec - starting.tv_sec + (ending.tv_usec - starting.tv_usec) / 1e6 << std::endl;
    std::cout << "Result Num: " << result_num << std::endl;
}




/*
 * Combination
 */
combination_p1::combination_p1(int d, int beg, const OvlpRSJoinParallel& joiner) :
        N(joiner.datasets1[d].size()), id(d), completed(false)
{
	if (N < 1 || joiner.c > N)
		completed = true;
	for (auto i = 0; i < joiner.c; ++i)
		curr.push_back(beg + 1 + i);
}

combination_p1::combination_p1(int d, int beg, const OvlpSelfJoinParallel& joiner) :
        N(joiner.datasets[d].size()), id(d), completed(false)
{
	if (N < 1 || joiner.c > N)
		completed = true;
	for (auto i = 0; i < joiner.c; ++i)
		curr.push_back(beg + 1 + i);
}

inline int combination_p1::getlastcurr(const OvlpRSJoinParallel& joiner) 
{
	assert(curr[joiner.c - 1] < N);
	return curr[joiner.c - 1];
}

inline int combination_p1::getlastcurr(const OvlpSelfJoinParallel& joiner) 
{
	assert(curr[joiner.c - 1] < N);
	return curr[joiner.c - 1];
}

void combination_p1::next(const OvlpRSJoinParallel &joiner) 
{
    int i = joiner.c - 1;
    while (i >= 0 && curr[i] >= N - joiner.c + i)
        --i;
    if (i < 0)
        completed = true;
    else {
        int temp = curr[i];
        for (int j = i; j <= joiner.c - 1; j++)
            curr[j] = temp + 1 + j - i;
    }
}

void combination_p1::print(const OvlpRSJoinParallel &joiner) const 
{
    std::cout << "combination1 from " << id << " " << joiner.datasets1[id].size() << " " << completed << " : ";
	// cout << "dataset's size" << joiner.datasets1[id].size() << " ";
    for (auto j = 0; j < joiner.c; j++)
        std::cout << joiner.datasets1[id][curr[j]] << " ";
    std::cout << " ----> ";
    for (auto j = 0; j < joiner.c; j++)
        std::cout << curr[j] << " ";
    std::cout << std::endl;
}

bool combination_p1::stepback(const int i, const OvlpRSJoinParallel &joiner) 
{
    if (i == 0)
        return true;
	// if(curr[i-1] < N-1)
    //	curr[i - 1]++;
    if (curr[i - 1] + 1 + joiner.c - 1 - i + 1 >= N) {
		// curr[i - 1] --;
        return stepback(i - 1, joiner);
	}
	curr[i-1] ++;
    for (int j = i; j < joiner.c; j++)
        curr[j] = curr[i - 1] + j - i + 1;
    return false;
}

bool combination_p1::stepback(const int i, const OvlpSelfJoinParallel &joiner)
{
	if (i == 0)
        return true;
	// if(curr[i-1] < N-1)
    //	curr[i - 1]++;
    if (curr[i - 1] + 1 + joiner.c - 1 - i + 1 >= N) {
		// curr[i - 1] --;
        return stepback(i - 1, joiner);
	}
	curr[i-1] ++;
    for (int j = i; j < joiner.c; j++)
        curr[j] = curr[i - 1] + j - i + 1;
    return false;
}

void combination_p1::binary(const combination_p1 &value, const OvlpSelfJoinParallel &joiner) 
{
    auto it = joiner.datasets[id].begin() + curr[0];
    for (int i = 0; i < joiner.c; i++) {
        // find the first one not larger than the value
        it = lower_bound(it, joiner.datasets[id].end(), joiner.datasets[value.id][value.curr[i]], OvlpUtilParallel::comp_int);
        // if get the end, we will increase the last one by 1 and set the rest as max
        if (it == joiner.datasets[id].end()) {
            completed = stepback(i, joiner);
            return;
            // if we get the same value, we fill in it
        } else if (*it == joiner.datasets[value.id][value.curr[i]]) {
			// int temp = curr[i];
            curr[i] = distance(joiner.datasets[id].begin(), it);
            // if we get the smaller value, we set the rest as max
        } else {
            curr[i] = distance(joiner.datasets[id].begin(), it);
            if (curr[i] + joiner.c - 1 - i >= N) {
                completed = stepback(i, joiner);
                return;
            }
            for (int j = i + 1; j < joiner.c; j++)
                curr[j] = curr[i] + j - i;
            return;
        }
    }
    return;
}

void combination_p1::binary(const combination_p2 &value, const OvlpRSJoinParallel &joiner) 
{
   auto it = joiner.datasets1[id].begin() + curr[0];
    for (int i = 0; i < joiner.c; i++) {
        // find the first one not larger than the value
        it = lower_bound(it, joiner.datasets1[id].end(), joiner.datasets2[value.id][value.curr[i]], OvlpUtilParallel::comp_int);
        // if get the end, we will increase the last one by 1 and set the rest as max
        if (it == joiner.datasets1[id].end()) {
            completed = stepback(i, joiner);
            return;
            // if we get the same value, we fill in it
        } else if (*it == joiner.datasets2[value.id][value.curr[i]]) {
			// int temp  = curr[i];
            curr[i] = distance(joiner.datasets1[id].begin(), it);
			// if(id == 187) cout << "id 187" << temp << " to " << curr[i] << endl;
            // if we get the smaller value, we set the rest as max
        } else {
			// int temp  = curr[i];
            curr[i] = distance(joiner.datasets1[id].begin(), it);
			// if(id == 187) cout << "id 187" << temp << " to " << curr[i] << endl;
            if (curr[i] + joiner.c - 1 - i >= N) {
                completed = stepback(i, joiner);
                return;
            }
            for (int j = i + 1; j < joiner.c; j++)
                curr[j] = curr[i] + j - i;
            return;
        }
    }
    return;
}

bool combination_p1::ifsame(const std::vector<ui> &data, const OvlpRSJoinParallel &joiner)
{
	for(int i = 0; i < joiner.c; i++)
		if(joiner.datasets1[id][curr[i]] != data[i])
			return false;
	return true;
}

combination_p2::combination_p2(int d, int beg, const OvlpRSJoinParallel &joiner):
        N(joiner.datasets2[d].size()), id(d), completed(false)
{
	if (N < 1 || joiner.c > N)
		completed = true;
	for (auto i = 0; i < joiner.c; ++i)
		curr.push_back(beg + 1 + i);
}

inline int combination_p2::getlastcurr(const OvlpRSJoinParallel &joiner) 
{
	assert(curr[joiner.c - 1] < N);
	return curr[joiner.c - 1];
}

void combination_p2::next(const OvlpRSJoinParallel &joiner) 
{
    int i = joiner.c - 1;
    while (i >= 0 && curr[i] >= N - joiner.c + i)
        --i;
    if (i < 0)
        completed = true;
    else {
        int temp = curr[i];
        for (int j = i; j <= joiner.c - 1; j++)
            curr[j] = temp + 1 + j - i;
    }
}

void combination_p2::print(const OvlpRSJoinParallel &joiner) const 
{
    std::cout << "combination2 from " << id << " " << joiner.datasets2[id].size() << " " << completed << " : ";
	// cout << "dataset's size" << joiner.datasets2[id].size() << " ";
    for (auto j = 0; j < joiner.c; j++)
        std::cout << joiner.datasets2[id][curr[j]] << " ";
    std::cout << " ----> ";
    for (auto j = 0; j < joiner.c; j++)
        std::cout << curr[j] << " ";
    std::cout << std::endl;
}

bool combination_p2::stepback(const int i, const OvlpRSJoinParallel &joiner) 
{
    if (i == 0)
        return true;
	// if(curr[i-1] < N-1)
    //	curr[i - 1]++;
    if (curr[i - 1] + 1 + joiner.c - 1 - i + 1 >= N) {
		// curr[i - 1] --;
        return stepback(i - 1, joiner);
	}
	curr[i-1] ++;
    for (int j = i; j < joiner.c; j++)
        curr[j] = curr[i - 1] + j - i + 1;
    return false;
}

void combination_p2::binary(const combination_p2 &value, const OvlpRSJoinParallel &joiner) 
{
    auto it = joiner.datasets2[id].begin() + curr[0];
    for (int i = 0; i < joiner.c; i++) {
        // find the first one not larger than the value
        it = lower_bound(it, joiner.datasets2[id].end(), joiner.datasets2[value.id][value.curr[i]], OvlpUtilParallel::comp_int);
        // if get the end, we will increase the last one by 1 and set the rest as max
        if (it == joiner.datasets2[id].end()) {
            completed = stepback(i, joiner);
            return;
            // if we get the same value, we fill in it
        } else if (*it == joiner.datasets2[value.id][value.curr[i]]) {
            curr[i] = distance(joiner.datasets2[id].begin(), it);
            // if we get the smaller value, we set the rest as max
        } else {
            curr[i] = distance(joiner.datasets2[id].begin(), it);
            if (curr[i] + joiner.c - 1 - i >= N) {
                completed = stepback(i, joiner);
                return;
            }
            for (int j = i + 1; j < joiner.c; j++)
                curr[j] = curr[i] + j - i;
            return;
        }
    }
    return;
}

void combination_p2::binary(const combination_p1 &value, const OvlpRSJoinParallel &joiner) 
{
    // cout << c << ' ';
    auto it = joiner.datasets2[id].begin() + curr[0];

    for (int i = 0; i < joiner.c; i++) {
		auto val = joiner.datasets1[value.id][value.curr[i]];
		// if(value.curr[i] >= joiner.datasets1[value.id].size())
		// 	val = 0;

        // find the first one not larger than the value
		// cout << *it << endl;
        it = lower_bound(it, joiner.datasets2[id].end(), val, OvlpUtilParallel::comp_int);
		// cout << *it << endl;
        // if get the end, we will increase the last one by 1 and set the rest as max
        if (it == joiner.datasets2[id].end()) {
            completed = stepback(i, joiner);
            return;
            // if we get the same value, we fill in it
        } else if (*it == val) {
            curr[i] = distance(joiner.datasets2[id].begin(), it);
            // if we get the smaller value, we set the rest as max
        } else {
            curr[i] = distance(joiner.datasets2[id].begin(), it);
            if (curr[i] + joiner.c - 1 - i >= N) {
                completed = stepback(i, joiner);
                return;
            }
            for (int j = i + 1; j < joiner.c; j++)
                curr[j] = curr[i] + j - i;
            return;
        }
    }
    return;
}

bool combination_p2::ifsame(const std::vector<ui> &data, const OvlpRSJoinParallel &joiner)
{
	for(int i = 0; i < joiner.c; i++)
		if(joiner.datasets2[id][curr[i]] != data[i])
			return false;
	return true;
}