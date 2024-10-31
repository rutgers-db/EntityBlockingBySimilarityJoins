/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/ovlpjoin.h"


// build heap for combination1
bool OvlpRSJoin::build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				 		int L, std::vector<int> &heap, std::vector<combination1> &combs, int &heap_size) 
{

	// get the number of large sets in the inverted list
	// int size = distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, comp_pair));
	// cause they are all treat as small sets
	int size = 0;

	// return false if there is no small set
	if (vec.size() < size + 1) return false;  

	// initialize heap and combination_tests
	heap.clear();
	combs.clear();
	for (auto i = size; i < vec.size(); i++) {
		// remove if there are not enough (>= c) elements left
		if ((int)(dataset[vec[i].first].size()) - 1 - vec[i].second < c) continue;
		heap.push_back(heap_size++);
		combs.emplace_back(vec[i].first, vec[i].second, *this);
	}

	if (heap_size == 0) return false;

	make_heap(heap.begin(), heap.end(), std::bind(&OvlpRSJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));

	return true;
}


// build heap for combination2
bool OvlpRSJoin::build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				 		int L, std::vector<int> &heap, std::vector<combination2> &combs, int &heap_size) 
{
	// get the number of large sets in the inverted list
	// int size = distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, comp_pair));
	// cause they are all treat as small sets
	int size = 0;

	// return false if there is no small set
	if (vec.size() < size + 1) return false;  

	// initialize heap and combination_tests
	heap.clear();
	combs.clear();
	for (auto i = size; i < vec.size(); i++) {
		// remove if there are not enough (>= c) elements left
		if ((int)(dataset[vec[i].first].size()) - 1 - vec[i].second < c) continue;
		heap.push_back(heap_size++);
		combs.emplace_back(vec[i].first, vec[i].second, *this);
	}

	if (heap_size == 0) return false;

	make_heap(heap.begin(), heap.end(), std::bind(&OvlpRSJoin::comp_comb2, this, std::placeholders::_1, std::placeholders::_2));
	
	return true;
}

bool OvlpSelfJoin::build_heap(const std::vector<std::pair<int,int>> &vec, const std::vector<std::vector<ui>> &dataset,
    				 		  int L, std::vector<int> &heap, std::vector<combination1> &combs, int &heap_size) 
{

	// get the number of large sets in the inverted list
	int size = std::distance(vec.begin(), lower_bound(vec.begin(), vec.end(), L, OvlpUtil::comp_pair));
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

	make_heap(heap.begin(), heap.end(), std::bind(&OvlpSelfJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));

	return true;
}


/*
 * Join
 */
void OvlpRSJoin::small_case(int L1, int R1, int L2, int R2, std::vector<std::pair<int, int>> &finalPairs) 
{
	if (L1 >= R1) return;
	if (L2 >= R2) return;
	--c;

	timeval beg, mid, mid1, end;
	gettimeofday(&beg, NULL);

	std::cout << " number of small sets: " << R1 - L1 << " and " << R2 - L2 << std::endl;

	std::vector<std::vector<int>> res_lists1;
	std::vector<std::vector<int>> res_lists2;

	gettimeofday(&mid, NULL);

	// FILE *fp = fopen("buffer/heap_size.txt", "w");

	int id1 = 0; 
	int id2 = 0;
	int64_t turn = 0;
	int64_t buildTurn = 0;

	for (int idx = total_eles - 1; idx >= 0; idx--) {

		int heap_size1 = 0;
		int heap_size2 = 0;
		ui totalZero = 0;

		if (!build_heap(ele_lists1[idx], datasets1, L1, heap1, combs1, heap_size1)) continue;
		if (!build_heap(ele_lists2[idx], datasets2, L2, heap2, combs2, heap_size2)) continue;
		++ buildTurn;

		bool pop1 = true;
		bool pop2 = true;

		// pop heaps
		// heap1: S, heap2: R
		std::vector<int> inv_list1;
		std::vector<int> inv_list2;

		do {
			bool found1 = false;
			bool found2 = false;
			turn ++;

			if (pop1) pop_heap(heap1.begin(), heap1.begin() + heap_size1--, std::bind(&OvlpRSJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));
			else  pop1 = true;
			if (pop2) pop_heap(heap2.begin(), heap2.begin() + heap_size2--, std::bind(&OvlpRSJoin::comp_comb2, this, std::placeholders::_1, std::placeholders::_2));
			else  pop2 = true;

			switch (OvlpUtil::compare(combs1[heap1[heap_size1]], combs2[heap2[heap_size2]], *this)) {
				case 1 : {	
#if BRUTEFORCE == 0
					combs1[heap1[heap_size1]].binary(combs2[heap2[heap_size2]], *this);
#elif BRUTEFORCE == 1
					combs1[heap1[heap_size1]].next(*this);
#endif
					if (combs1[heap1[heap_size1]].completed) {
						heap1.pop_back();
					} else {
						push_heap(heap1.begin(), heap1.end(), std::bind(&OvlpRSJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));
						++heap_size1;
					}
					pop2 = false;
					break;
				}

				case -1 : {
#if BRUTEFORCE == 0
					combs2[heap2[heap_size2]].binary(combs1[heap1[heap_size1]], *this);
#elif BRUTEFORCE == 1
					combs2[heap2[heap_size2]].next(*this);
#endif
					if (combs2[heap2[heap_size2]].completed) {
						heap2.pop_back();
						// if(turn == 34) cout << "pop here" << endl;
					} else {
						push_heap(heap2.begin(), heap2.end(), std::bind(&OvlpRSJoin::comp_comb2, this, std::placeholders::_1, std::placeholders::_2));
						++heap_size2;
					}
					pop1 = false;
					break;
				}

				case 0 : {
					inv_list1.clear();
					inv_list2.clear();

					totalZero ++;

					inv_list1.push_back(combs1[heap1[heap_size1]].id);
		
					while (heap_size1 > 0 && OvlpUtil::is_equal(combs1[heap1[heap_size1]], combs1[heap1.front()], *this)) {
						inv_list1.push_back(combs1[heap1.front()].id);
						pop_heap(heap1.begin(), heap1.begin() + heap_size1, std::bind(&OvlpRSJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));
						--heap_size1;
					} 

					inv_list2.push_back(combs2[heap2[heap_size2]].id);
	
					while (heap_size2 > 0 && OvlpUtil::is_equal(combs2[heap2[heap_size2]], combs2[heap2.front()], *this)) {
						inv_list2.push_back(combs2[heap2.front()].id);
						pop_heap(heap2.begin(), heap2.begin() + heap_size2, std::bind(&OvlpRSJoin::comp_comb2, this, std::placeholders::_1, std::placeholders::_2));
						--heap_size2;
					} 

#if LIMIT_INV_SIZE == 1
					if (inv_list1.size() > MAX_INV_SIZE)
						inv_list1.resize(MAX_INV_SIZE);
					if (inv_list2.size() > MAX_INV_SIZE)
						inv_list2.resize(MAX_INV_SIZE);
#endif

					res_lists1.push_back(std::move(inv_list1));
					res_lists2.push_back(std::move(inv_list2));

					if (heap_size1 == 0 && heap_size2 == 0) break;

					for (auto i = heap_size1; i < heap1.size(); ++i) {
#if REPORT_BINARY == 1
						printf("Before binary\n");
						combs1[heap1[i]].print(*this);
#endif
#if BRUTEFORCE == 0
						combs1[heap1[i]].binary(combs2[heap2.front()], *this);
#elif BRUTEFORCE == 1
						combs1[heap1[i]].next(*this);
#endif
#if REPORT_BINARY == 1
						printf("After binary\n");
						combs1[heap1[i]].print(*this);
#endif
					}

					for (auto i = heap_size2; i < heap2.size(); ++i) {
#if REPORT_BINARY == 1
						printf("Before binary\n");
						combs2[heap2[i]].print(*this);
#endif
#if BRUTEFORCE == 0
						combs2[heap2[i]].binary(combs1[heap1.front()], *this);
#elif BRUTEFORCE == 1
						combs2[heap2[i]].next(*this);
#endif
#if REPORT_BINARY == 1
						printf("After binary\n");
						combs2[heap2[i]].print(*this);
#endif
					}

					int comp_num1 = 0;
					for (auto i = heap_size1; i < heap1.size(); ++i) {
						if (combs1[heap1[i]].completed)
						++comp_num1;
						else if (comp_num1 > 0)
						heap1[i - comp_num1] = heap1[i];
					}

					int comp_num2 = 0;
					for (auto i = heap_size2; i < heap2.size(); ++i) {
						if (combs2[heap2[i]].completed)
						++comp_num2;
						else if (comp_num2 > 0)
						heap2[i - comp_num2] = heap2[i];
					}

					for (auto i = heap_size1; i < (int)heap1.size() - comp_num1; i++) {
						push_heap(heap1.begin(), heap1.begin() + i + 1, std::bind(&OvlpRSJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));
					}

					for (auto i = heap_size2; i < (int)heap2.size() - comp_num2; i++) {
						push_heap(heap2.begin(), heap2.begin() + i + 1, std::bind(&OvlpRSJoin::comp_comb2, this, std::placeholders::_1, std::placeholders::_2));
					}

					while (comp_num1-- > 0)
						heap1.pop_back();
					heap_size1 = heap1.size();

					while (comp_num2-- > 0)
						heap2.pop_back();
					heap_size2 = heap2.size();
					break;
				}
			}
		} while ((heap_size1 > 0 || (heap_size1 >= 0 && !pop1)) && (heap_size2 > 0 || (heap_size2 >= 0 && !pop2)));

		// fprintf(fp, "%d\n", totalZero);
	}
	// fclose(fp);

	std::cout << "Res lists num: " << res_lists1.size() << " " << res_lists2.size() << std::endl;
	fflush(stdout);

	gettimeofday(&mid1, NULL);

	std::vector<std::vector<int>> id_lists(n1);
	for (auto i = 0; i < res_lists1.size(); i++) {
		for (auto j = 0; j < res_lists1[i].size(); j++)
			id_lists[res_lists1[i][j]].push_back(i);
	}
	std::cout << "ID list num: " << id_lists.size() << std::endl;

	std::vector<int> results(n2, -1);
	int isHeap = 0;

	for (auto i = n1 - 1; i >= 0; i--) {
		if (id_lists[i].empty()) 
			continue;

		for (auto j = 0; j < id_lists[i].size(); j++) {
			for (auto k = 0; k < res_lists2[id_lists[i][j]].size(); k++) {
				if (results[res_lists2[id_lists[i][j]][k]] != i) {
					int idd1 = idmap_records1[i].first;
					int idd2 = idmap_records2[res_lists2[id_lists[i][j]][k]].first;
#if MAINTAIN_VALUE_OVLP == 0
						result_pairs.emplace_back(idd1, idd2);
						if(result_pairs.size() >= maxHeapSize) {
							finalPairs = result_pairs;
							return;
						}
#elif MAINTAIN_VALUE_OVLP == 1
						double val = isWeightedComp ? weightedOverlapCoeff(idd1, idd2)
													: overlapCoeff(idd1, idd2);
						if(result_pairs_.size() < maxHeapSize) {
							result_pairs_.emplace_back(idd1, idd2, val);
						}
						else {
							if(isHeap == 0) {
								std::make_heap(result_pairs_.begin(), result_pairs_.end());
								isHeap = 1;
							}
							
							if(result_pairs_[0].val < val) {
								std::pop_heap(result_pairs_.begin(), result_pairs_.end());
								result_pairs_.pop_back();
								result_pairs_.emplace_back(idd1, idd2, val);
								std::push_heap(result_pairs_.begin(), result_pairs_.end());
							}
						}
#endif
					results[res_lists2[id_lists[i][j]][k]] = i;
					++result_num;
				}
			}
		}
	}

	// Deduplicate
	sort(result_pairs.begin(), result_pairs.end());
	auto iter = unique(result_pairs.begin(), result_pairs.end());
	if(iter != result_pairs.end()) {
		std::cerr << "Duplicate pairs detected" << std::endl;
		exit(1);
	}

#if MAINTAIN_VALUE_OVLP == 0
	finalPairs = result_pairs;
#elif MAINTAIN_VALUE_OVLP == 1
	for(const auto &p : result_pairs_) {
		finalPairs.emplace_back(p.id1, p.id2);
	}
#endif

	++c;
	gettimeofday(&end, NULL);
	std::cout << " small p1 : " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << std::endl;
	std::cout << " small p2 : " << mid1.tv_sec - mid.tv_sec + (mid1.tv_usec - mid.tv_usec) / 1e6 << std::endl;
	std::cout << " small p3 : " << end.tv_sec - mid1.tv_sec + (end.tv_usec - mid1.tv_usec) / 1e6 << std::endl;
	fflush(stdout);
}


void OvlpRSJoin::overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs)
{
	srand(time(NULL));
	
	timeval starting, ending, s1, t1, s2, t2;
	timeval time1, time2, time3, time4;

	gettimeofday(&starting, NULL);

	c = overlap_threshold;           // get threshold
	n1 = records1.size();
	n2 = records2.size();
	
	std::unordered_map<ui, std::vector<int>> ele1;
	std::unordered_map<ui, std::vector<int>> ele2;

	gettimeofday(&time1, NULL);
	OvlpUtil::removeShort(records1, ele1, *this);
	OvlpUtil::removeShort(records2, ele2, *this);

#if BRUTEFORCE == 0
	// removeWidow(ele1, ele2); // we dont need to remove widow in ele1 cause they all from joiner.records2
	OvlpUtil::removeWidow(ele2, ele1);
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

	OvlpUtil::transform(ele1, eles, idmap_records1, ele_lists1, datasets1, total_eles, n1, *this);
	OvlpUtil::transform(ele2, eles, idmap_records2, ele_lists2, datasets2, total_eles, n2, *this);

	gettimeofday(&time3, NULL);
	std::cout << "Transform Time: " << time3.tv_sec - time1.tv_sec + (time3.tv_usec - time1.tv_usec) / 1e6 << std::endl;

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


void OvlpSelfJoin::small_case(int L, int R, std::vector<std::pair<int, int>> &finalPairs) 
{
    --c;

    timeval mid, mid1, end;
    std::vector<std::vector<int>> res_lists;
    
    gettimeofday(&mid, NULL);
    for (int idx = total_eles - 1; idx >= 0; idx--) {
        if (ele_lists[idx].size() < 2) {
			// cout << ele_lists[idx].size() << endl;
            continue;
		}

		int heap_size = 0;
		auto &curEleList = ele_lists[idx]; 

		if (!build_heap(curEleList, datasets, L, heap, combs, heap_size)) {
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
                std::pop_heap(heap.begin(), heap.begin() + heap_size, std::bind(&OvlpSelfJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));
                --heap_size;
                inv_list.emplace_back(combs[heap[heap_size]].id);
            } while (heap_size > 0 && OvlpUtil::is_equal(combs[heap[heap_size]], combs[heap.front()], *this));

#if LIMIT_INV_SIZE == 1
			if (inv_list.size() > MAX_INV_SIZE)
				inv_list.resize(MAX_INV_SIZE);
#endif

            if (inv_list.size() > 1) 
                res_lists.push_back(std::move(inv_list));

            if (heap_size == 0)
                break;

            for (auto i = heap_size; i < (int)heap.size(); ++i) 
                combs[heap[i]].binary(combs[heap.front()], *this);

            int comp_num = 0;
            for (auto i = heap_size; i < (int)heap.size(); ++i) {
                if (combs[heap[i]].completed)
                    ++comp_num;
                else if (comp_num > 0)
                    heap[i - comp_num] = heap[i];
            }

            for (auto i = heap_size; i < (int)heap.size() - comp_num; i++)
                std::push_heap(heap.begin(), heap.begin() + i + 1, std::bind(&OvlpSelfJoin::comp_comb1, this, std::placeholders::_1, std::placeholders::_2));

            while (comp_num-- > 0)
               	heap.pop_back();
            heap_size = heap.size();
        }
    }

    gettimeofday(&mid1, NULL);

	// get results
	std::vector<std::vector<int>> id_lists(n);
	for (auto i = 0; i < res_lists.size(); i++) {
		for (auto j = 0; j < res_lists[i].size(); j++)
		id_lists[res_lists[i][j]].push_back(i);
	}

	std::vector<int> results(n, -1);
	int isHeap = 0;

	for (auto i = n - 1; i >= 0; i--) {
		if (id_lists[i].empty())
			 continue;
		for (auto j = 0; j < id_lists[i].size(); j++) {
			res_lists[id_lists[i][j]].pop_back();
			for (auto k = 0; k < res_lists[id_lists[i][j]].size(); k++) {
				if (results[res_lists[id_lists[i][j]][k]] != i) {
					int idd1 = idmap_records[i].first;
					int idd2 = idmap_records[res_lists[id_lists[i][j]][k]].first;
#if MAINTAIN_VALUE_OVLP == 0
					if(idd1 < idd2) result_pairs.emplace_back(idd1, idd2);
					else result_pairs.emplace_back(idd2, idd1);
					if(result_pairs.size() >= maxHeapSize) {
						finalPairs = result_pairs;
						return;
					}
#elif MAINTAIN_VALUE_OVLP == 1
					double val = isWeightedComp ? weightedOverlapCoeff(idd1, idd2)
												: overlapCoeff(idd1, idd2);
					if(result_pairs_.size() < maxHeapSize) {
						if(idd1 < idd2) result_pairs_.emplace_back(idd1, idd2, val);
						else result_pairs_.emplace_back(idd2, idd1, val);
					}
					else {
						if(isHeap == 0) {
							std::make_heap(result_pairs_.begin(), result_pairs_.end());
							isHeap = 1;
						}
						
						if(result_pairs_[0].val < val) {
							std::pop_heap(result_pairs_.begin(), result_pairs_.end());
							result_pairs_.pop_back();
							if(idd1 < idd2) result_pairs_.emplace_back(idd1, idd2, val);
							else result_pairs_.emplace_back(idd2, idd1, val);
							std::push_heap(result_pairs_.begin(), result_pairs_.end());
						}
					}
#endif
					results[res_lists[id_lists[i][j]][k]] = i;
					++result_num;
				}
			}
		}
	}

#if MAINTAIN_VALUE_OVLP == 0
	finalPairs = result_pairs;
#elif MAINTAIN_VALUE_OVLP == 1
	for(const auto &p : result_pairs_) {
		finalPairs.emplace_back(p.id1, p.id2);
	}
#endif

    ++c;

    std::cout << "candidate number: " << candidate_num << std::endl;
	std::cout << "result number: " << result_num << std::endl;
    gettimeofday(&end, NULL);
    std::cout << " small p2 : " << mid1.tv_sec - mid.tv_sec + (mid1.tv_usec - mid.tv_usec) / 1e6 << std::endl;
    std::cout << " small p3 : " << end.tv_sec - mid1.tv_sec + (end.tv_usec - mid1.tv_usec) / 1e6 << std::endl << std::flush;
}


void OvlpSelfJoin::overlapjoin(int overlap_threshold, std::vector<std::pair<int, int>> &finalPairs) 
{
    srand(time(NULL));

    timeval starting, ending, s1, t1, s2, t2;
    timeval time1, time3, time4;

    gettimeofday(&starting, NULL);

    c = overlap_threshold;           // get threshold
    n = records.size();              // get number of records

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
    datasets.resize(n);

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
    for (auto i = 0; i < n; i++)
        if ((int)datasets[i].size() < c)
            datasets[i].clear();

    // create id mappings: from sorted to origin
    for (auto i = 0; i < n; i++)
        idmap_records.emplace_back(i, datasets[i].size());

    // sort records by length in decreasing order
    sort(idmap_records.begin(), idmap_records.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.second > b.second; });
    // todo Writing idmap_records[idmap_records]
    sort(datasets.begin(), datasets.end(), [](const std::vector<ui> &a, const std::vector<ui> &b) { return a.size() > b.size(); });
    std::cout << " largest set: " << datasets.front().size() << " smallest set: " << datasets.back().size() << " It might be 0 cause some row in dataset, its length is smaller than c" << std::endl;

    // build real inverted index
    ele_lists.resize(total_eles);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)datasets[i].size(); j++)
            ele_lists[datasets[i][j]].emplace_back(i, j);

    gettimeofday(&time3, NULL);
    std::cout << "Transform Time: " << time3.tv_sec - time1.tv_sec + (time3.tv_usec - time1.tv_usec) / 1e6 << std::endl;

    std::cout << " All are treated as small sets: " << n << std::endl;

    gettimeofday(&time4, NULL);
    // ****** conduct joining ******
    result_num = 0;

    gettimeofday(&s2, NULL);
    small_case(0, n, finalPairs);
    gettimeofday(&t2, NULL);

    gettimeofday(&ending, NULL);
    std::cout << "Join Time: " << ending.tv_sec - time4.tv_sec + (ending.tv_usec - time4.tv_usec) / 1e6 << std::endl;
    std::cout << "  small Time: " << t2.tv_sec - s2.tv_sec + (t2.tv_usec - s2.tv_usec) / 1e6 << std::endl;
    std::cout << "All Time: " << ending.tv_sec - starting.tv_sec + (ending.tv_usec - starting.tv_usec) / 1e6 << std::endl;
    std::cout << "Result Num: " << result_num << std::endl;
}


/*
 * utils
 */
void OvlpUtil::removeShort(const std::vector<std::vector<ui>> &records, std::unordered_map<ui, std::vector<int>> &ele, 
							 const OvlpRSJoin &joiner) 
{
	for (int i = 0; i < records.size(); i++) {
		if (records[i].size() < joiner.c) 
			continue;
		for (ui j = 0; j < records[i].size(); j++)
			ele[records[i][j]].push_back(i);
	}
}


// Remove "widows" from a hash map based on another hash map.
// This function removes key-value pairs from the unordered_map 'ele' 
// if the key doesn't exist in another unordered_map 'ele_other'.
void OvlpUtil::removeWidow(std::unordered_map<ui, std::vector<int>> &ele, const std::unordered_map<ui, std::vector<int>> &ele_other) 
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


void OvlpUtil::transform(std::unordered_map<ui, std::vector<int>> &ele, const std::vector<std::pair<int, int>> &eles, 
                         std::vector<std::pair<int, int>> &idmap, std::vector<std::vector<std::pair<int, int>>> &ele_lists,
                         std::vector<std::vector<ui>> &dataset, const ui total_eles, const int n, const OvlpRSJoin &joiner) 
{
	dataset.resize(n);

	// the numbers in dataset is from large to small
	// the frequency in dataset is from small to large
	for (ui i = 0; i < eles.size(); ++i) {
		for (auto j = ele[eles[i].first].begin(); j != ele[eles[i].first].end(); j++)
			dataset[*j].push_back(total_eles - i - 1);
	}

	for (auto i = 0; i < n; i++)
		if (dataset[i].size() < joiner.c) 
			dataset[i].clear();

	for (auto i = 0; i < n; i++)
		idmap.emplace_back(i, (int)dataset[i].size());

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
		for (int j = 0; j < dataset[i].size(); j++)
			ele_lists[dataset[i][j]].emplace_back(i,j);

	std::cout << "Build ele_list" << std::endl;
}


combination1::combination1(int d, int beg, const OvlpRSJoin &joiner)
: N(joiner.datasets1[d].size()), id(d), completed(false) 
{
	if (N < 1 || joiner.c > N)
		completed = true;
	for (auto i = 0; i < joiner.c; ++i)
		curr.push_back(beg + 1 + i);
}

combination1::combination1(int d, int beg, const OvlpSelfJoin &joiner) 
: N(joiner.datasets[d].size()), id(d), completed(false) 
{
	if (N < 1 || joiner.c > N)
		completed = true;
	for (auto i = 0; i < joiner.c; ++i)
		curr.push_back(beg + 1 + i);
}

int combination1::getlastcurr(const OvlpRSJoin &joiner) 
{
	assert(curr[joiner.c - 1] < N);
	return curr[joiner.c - 1];
}

int combination1::getlastcurr(const OvlpSelfJoin &joiner) 
{
	assert(curr[joiner.c - 1] < N);
	return curr[joiner.c - 1];
}

    // compute next combination_test1
void combination1::next(const OvlpRSJoin &joiner) 
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

void combination1::next(const OvlpSelfJoin &joiner) 
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

void combination1::print(const OvlpRSJoin &joiner) const 
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

void combination1::print(const OvlpSelfJoin &joiner) const 
{
	std::cout << "combination1 from " << id << " " << joiner.datasets[id].size() << " " << completed << " : ";
	// cout << "dataset's size" << joiner.datasets1[id].size() << " ";
	for (auto j = 0; j < joiner.c; j++)
		std::cout << joiner.datasets[id][curr[j]] << " ";
	std::cout << " ----> ";
	for (auto j = 0; j < joiner.c; j++)
		std::cout << curr[j] << " ";
	std::cout << std::endl;
}

bool combination1::stepback(const int i, const OvlpRSJoin &joiner) 
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

bool combination1::stepback(const int i, const OvlpSelfJoin &joiner) 
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

void combination1::binary(const combination1 &value, const OvlpSelfJoin &joiner) 
{
	auto it = joiner.datasets[id].begin() + curr[0];
	for (int i = 0; i < joiner.c; i++) {
		// find the first one not larger than the value
		it = std::lower_bound(it, joiner.datasets[id].end(), joiner.datasets[value.id][value.curr[i]], OvlpUtil::comp_int);
		// if get the end, we will increase the last one by 1 and set the rest as max
		if (it == joiner.datasets[id].end()) {
			completed = stepback(i, joiner);
			return;
			// if we get the same value, we fill in it
		} else if (*it == joiner.datasets[value.id][value.curr[i]]) {
			int temp = curr[i];
			curr[i] = std::distance(joiner.datasets[id].begin(), it);
			// if we get the smaller value, we set the rest as max
		} else {
			curr[i] = std::distance(joiner.datasets[id].begin(), it);
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

void combination1::binary(const combination2 &value, const OvlpRSJoin &joiner) 
{
	auto it = joiner.datasets1[id].begin() + curr[0];
	for (int i = 0; i < joiner.c; i++) {
		// find the first one not larger than the value
		it = std::lower_bound(it, joiner.datasets1[id].end(), joiner.datasets2[value.id][value.curr[i]], OvlpUtil::comp_int);
		// if get the end, we will increase the last one by 1 and set the rest as max
		if (it == joiner.datasets1[id].end()) {
			completed = stepback(i, joiner);
			return;
			// if we get the same value, we fill in it
		} else if (*it == joiner.datasets2[value.id][value.curr[i]]) {
			// int temp  = curr[i];
			curr[i] = std::distance(joiner.datasets1[id].begin(), it);
			// if(id == 187) cout << "id 187" << temp << " to " << curr[i] << endl;
			// if we get the smaller value, we set the rest as max
		} else {
			// int temp  = curr[i];
			curr[i] = std::distance(joiner.datasets1[id].begin(), it);
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

// test unit
bool combination1::ifsame(const std::vector<ui> &data, const OvlpRSJoin &joiner) 
{
	for(ui i = 0; i < joiner.c; i++)
		if(joiner.datasets1[id][curr[i]] != data[i])
			return false;
	return true;
}

combination2::combination2(int d, int beg, const OvlpRSJoin &joiner)
: N(joiner.datasets2[d].size()), id(d), completed(false) 
{
	if (N < 1 || joiner.c > N)
		completed = true;
	for (auto i = 0; i < joiner.c; ++i)
		curr.push_back(beg + 1 + i);
}

int combination2::getlastcurr(const OvlpRSJoin &joiner) 
{
	assert(curr[joiner.c - 1] < N);
	return curr[joiner.c - 1];
}

    // compute next combination2
void combination2::next(const OvlpRSJoin &joiner) 
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

void combination2::print(const OvlpRSJoin &joiner) const 
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

bool combination2::stepback(const int i, const OvlpRSJoin &joiner) 
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

void combination2::binary(const combination2 &value, const OvlpRSJoin &joiner) 
{
	auto it = joiner.datasets2[id].begin() + curr[0];
	for (int i = 0; i < joiner.c; i++) {
		// find the first one not larger than the value
		it = std::lower_bound(it, joiner.datasets2[id].end(), joiner.datasets2[value.id][value.curr[i]], OvlpUtil::comp_int);
		// if get the end, we will increase the last one by 1 and set the rest as max
		if (it == joiner.datasets2[id].end()) {
			completed = stepback(i, joiner);
			return;
			// if we get the same value, we fill in it
		} else if (*it == joiner.datasets2[value.id][value.curr[i]]) {
			curr[i] = std::distance(joiner.datasets2[id].begin(), it);
			// if we get the smaller value, we set the rest as max
		} else {
			curr[i] = std::distance(joiner.datasets2[id].begin(), it);
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

void combination2::binary(const combination1 &value, const OvlpRSJoin &joiner) 
{
	// cout << c << ' ';
	auto it = joiner.datasets2[id].begin() + curr[0];

	for (int i = 0; i < joiner.c; i++) {
		auto val = joiner.datasets1[value.id][value.curr[i]];
		// if(value.curr[i] >= joiner.datasets1[value.id].size())
		// 	val = 0;

		// find the first one not larger than the value
		// cout << *it << endl;
		it = std::lower_bound(it, joiner.datasets2[id].end(), val, OvlpUtil::comp_int);
		// cout << *it << endl;
		// if get the end, we will increase the last one by 1 and set the rest as max
		if (it == joiner.datasets2[id].end()) {
			completed = stepback(i, joiner);
			return;
			// if we get the same value, we fill in it
		} else if (*it == val) {
			curr[i] = std::distance(joiner.datasets2[id].begin(), it);
			// if we get the smaller value, we set the rest as max
		} else {
			curr[i] = std::distance(joiner.datasets2[id].begin(), it);
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

// test unit
bool combination2::ifsame(const std::vector<ui> &data, const OvlpRSJoin &joiner) 
{
	for(ui i = 0; i < joiner.c; i++)
		if(joiner.datasets2[id][curr[i]] != data[i])
			return false;
	return true;
}