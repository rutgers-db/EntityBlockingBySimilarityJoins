/*
 * author: Dong Deng 
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/stringjoin_parallel.h"


void StringJoinParallel::init()
{
	for(ui i = 0; i < MAXTHREADNUM; i++) {
		quickRef[i] = new bool[N + 1];
		hv[i] = new hashValue[PN];
	}
	partPos = new int *[PN + 1];
	partLen = new int *[PN];
	invLists = new InvListsParallel *[PN];
	invListsPre = new InvListPrefix *[PN];
	partIndex = new std::vector<PIndex> *[PN];
	power = new uint64_t[maxDictLen + 1];

	for (int lp = 0; lp < PN; lp++) {
		partPos[lp] = new int[maxDictLen + 1];
		partLen[lp] = new int[maxDictLen + 1];
		invLists[lp] = new InvListsParallel[maxDictLen + 1];
		invListsPre[lp] = new InvListPrefix[maxDictLen + 1];
		partIndex[lp] = new std::vector<PIndex>[maxDictLen + 1];
	}
	partPos[PN] = new int[maxDictLen + 1];

	dist = new int[maxDictLen + 2];
	for(int lp = 0; lp <= maxDictLen + 1; lp++)
		dist[lp] = N;

	for(ui i = 0; i < MAXTHREADNUM; i++) {
		matrix[i] = new int *[maxDictLen + 1];
		_matrix[i] = new int *[maxDictLen + 1];
		for (int lp = 0; lp <= maxDictLen; lp++) {
			matrix[i][lp] = new int[2 * D + 1];
			_matrix[i][lp] = new int[2 * D + 1];
		}
	}

	// init
	for(ui i = 0; i < MAXTHREADNUM; i++) {
		for (int lp = 0; lp <= D; lp++)
			matrix[i][0][D + lp] = _matrix[i][0][D + lp] = lp;
		for (int lp = 0; lp <= N; lp++)
			quickRef[i][lp] = false;
	}

	power[0] = 1;
	for (int lp = 1; lp <= maxDictLen / PN + 2; lp++)
		power[lp] = (hashNumber * power[lp - 1]);

	// fill in partPos and partLen
	// init the information of the first and last segment
	for (int len = minDictLen; len <= maxDictLen; len++) {
		partPos[0][len] = 0;
		partLen[0][len] = len / PN;
		partPos[PN][len] = len;
	}

	// fill int the information of middle segments
	for (int pid = 1; pid < PN; pid++) {
		for (int len = minDictLen; len <= maxDictLen; len++) {
			partPos[pid][len] = partPos[pid - 1][len] + partLen[pid - 1][len];
			if (pid == (PN - len % PN))
				partLen[pid][len] = partLen[pid - 1][len] + 1;
			else
				partLen[pid][len] = partLen[pid - 1][len];
		}
	}
}


void StringJoinParallel::prepareSelf()
{
	int cLen = -1;

	// fill in dist
	for (int id = 0; id < workN; id++) {
		if (cLen == (int)work_dataset[id].length()) {
			worklengthMap[worklengthMap.size()-1].emplace_back(id);
			continue;
		}
		for (int lp = cLen + 1; lp <= (int)work_dataset[id].length(); lp++) 
			dist[lp] = id;
		cLen = work_dataset[id].length();

		workLengthArray.emplace_back(cLen);
		worklengthMap.emplace_back();
		worklengthMap[worklengthMap.size()-1].emplace_back(id);
	}

	// generate the set of pairs of segment and substring to be check
	int diffLength = (int)worklengthMap.size();
#pragma omp parallel for
	for (int wid = 0; wid < diffLength; wid++) {
		int currLen = (int)work_dataset[worklengthMap[wid][0]].length();

		for (int pid = 0; pid < PN; pid++) {
			// enumerate all lengths of string which potentially
			// similar to current string
			int lowerBound = std::max(currLen - D, workMinDictLen);
			for (int len = lowerBound; len <= currLen; len++) {
				if (dist[len] == dist[len + 1]) 
					continue;

				// enumerate the valid substrings with respect to clen, len and pid
				int delta = currLen - len;
				for (int stPos = std::max(partPos[pid][len] - pid,
					partPos[pid][len] + delta - (D - pid));
					stPos <= std::min(partPos[pid][len] + pid,
					partPos[pid][len] + delta + (D - pid)); stPos++)
					partIndex[pid][currLen].push_back(PIndex(
						stPos, partPos[pid][len], partLen[pid][len], len));
			}

			// sort the pairs of segment and substring to be check,
			// for conveniently hash substring and segment
			sort(partIndex[pid][currLen].begin(), partIndex[pid][currLen].end(), StringJoinUtil::PIndexLess);
		}
	}
}


void StringJoinParallel::prepareRS()
{
	int cLen = -1;

	// fill in dist
	for (int id = 0; id < workN; id++) {
		if (cLen == (int)work_dataset[id].length()) {
			worklengthMap[worklengthMap.size()-1].emplace_back(id);
			continue;
		}
		for (int lp = cLen + 1; lp <= (int)work_dataset[id].length(); lp++) 
			dist[lp] = id;
		cLen = work_dataset[id].length();

		workLengthArray.emplace_back(cLen);
		worklengthMap.emplace_back();
		worklengthMap[worklengthMap.size()-1].emplace_back(id);
	}

	// generate the set of pairs of segment and substring to be check
	std::vector<int> queryLengthId;
	cLen = -1;
	for(int qid = 0; qid < queryN; qid++) {
		int currLen = (int)query_dataset[qid].length();
		if(currLen != cLen) {
			queryLengthId.emplace_back(qid);
			cLen = currLen;
		}
	}

	int diffLength = (int)queryLengthId.size();
#pragma omp parallel for
	for (int qid = 0; qid < diffLength; qid++) {
		int currLen = (int)query_dataset[queryLengthId[qid]].length();

		for (int pid = 0; pid < PN; pid++) {
			// enumerate all lengths of string which potentially
			// similar to current string
			for (int len = std::max(currLen - D, workMinDictLen); len <= std::min(currLen + D, workMaxDictLen); len++) {
				if (dist[len] == dist[len + 1]) 
					continue;

				// enumerate the valid substrings with respect to clen, len and pid
				int delta = currLen - len;
				for (int stPos = std::max(partPos[pid][len] - pid,
					partPos[pid][len] + delta - (D - pid));
					stPos <= std::min(partPos[pid][len] + pid,
					partPos[pid][len] + delta + (D - pid)); stPos++)
					partIndex[pid][currLen].push_back(PIndex(
						stPos, partPos[pid][len], partLen[pid][len], len));
			}

			// sort the pairs of segment and substring to be check,
			// for conveniently hash substring and segment
			sort(partIndex[pid][currLen].begin(), partIndex[pid][currLen].end(), StringJoinUtil::PIndexLess);
		}
	}

#if APPROXIMATE == 1
	workPrefixHash.reserve(workN);
	queryPrefixHash.reserve(queryN);

	for(int i = 0; i < workN; i++) {
		if(work_dataset[i].length() < sharePrefix)
			workPrefixHash.emplace_back(0);
		else
			workPrefixHash.emplace_back(StringJoinUtil::strHash(work_dataset[i], 0, sharePrefix));
	}
	for(int i = 0; i < queryN; i++) {
		if(query_dataset[i].length() < sharePrefix)
			queryPrefixHash.emplace_back(0);
		else
			queryPrefixHash.emplace_back(StringJoinUtil::strHash(query_dataset[i], 0, sharePrefix));
	}
#endif
}


void StringJoinParallel::selfJoin(std::vector<std::pair<int, int>> &finalPairs)
{
	timeval joinBeign, joinEnd;
	gettimeofday(&joinBeign, NULL);

	init();

	prepareSelf();

	// inverted index
	workInvSC = new int[workN];
	for(int i = 0; i < workN; i++) {
		// if(work_dataset[i].empty())
		// 	std::cerr << i << endl;
		workInvSC[i] = std::distance(strCount.begin(), strCount.find(work_dataset[i]));
	}
#if REPORT_STR_COUNT == 1
	for(const auto &str : strCount) {
		// if(str.empty())
		// 	std::cerr << "Empty" << endl;
		printf("%s\n", str.c_str());
	}
#endif

	// index
	// may index in iteration for an optimization
	ui diffLengthSize = worklengthMap.size();
#pragma omp parallel for
	for(ui i = 0; i < diffLengthSize; i++) {
		ui sameLengthSize = worklengthMap[i].size();
		int clen = workLengthArray[i];

		for(ui j = 0; j < sameLengthSize; j++) {
			int wid = worklengthMap[i][j];

			for (int partId = 0; partId < PN; partId++) {
				int pLen = partLen[partId][clen];
				int stPos = partPos[partId][clen];
				uint64_t hashIndex = StringJoinUtil::strHash(work_dataset[wid], stPos, pLen);

				invLists[partId][clen][hashIndex].push_back(wid);

				if(invLists[partId][clen][hashIndex].size() == 1) 
					invListsPre[partId][clen][hashIndex].emplace_back(0, 0);
				else {
					int lidx = 0, leftSharing = 0;
					int ridx = 0, rightSharing = 0;
					std::string prevString = work_dataset[invLists[partId][clen][hashIndex][invLists[partId][clen][hashIndex].size() - 2]];

					while(prevString[lidx] == work_dataset[wid][lidx] && lidx < stPos) {
						++ leftSharing;
						++ lidx;
					}
					while(prevString[stPos + pLen + ridx] == work_dataset[wid][stPos + pLen + ridx] && ridx < clen - stPos - pLen) {
						++ rightSharing;
						++ ridx;
					}
					invListsPre[partId][clen][hashIndex].emplace_back(leftSharing, rightSharing);
				}
			}
		}
	}
	printf("Finish indexing\n");

	// current string, to generate substrings
#pragma omp parallel for schedule(dynamic)
	for (int id = 0; id < workN; id++) {
		int tid = omp_get_thread_num();
		int clen = work_dataset[id].length();
#if MAINTAIN_VALUE_EDIT == 0
		if(earlyTerminated[tid] == 1)
			continue;
#endif
		
		auto &currQuickRef = quickRef[tid];
		auto &currHv = hv[tid];
		auto &currPair = pairs[tid];
#if MAINTAIN_VALUE_EDIT == 1
		auto &currPair_ = result_pairs_[tid];
#endif
		std::vector<int> currRes;

		// current part
		for (int partId = 0; partId < PN; partId++) {

			// threshold of left part M and right part _M
			int M = partId;
			int _M = D - partId;

			for (const auto &index : partIndex[partId][clen]) {
				int stPos = index.stPos;
				int Lo = index.Lo;
				int pLen = index.partLen;
				int len = index.len;

				currHv[partId].update(id, stPos, stPos + pLen - 1, work_dataset[id], power);

				auto inv_it = invLists[partId][len].find(currHv[partId].value);
				auto invPre_it = invListsPre[partId][len].find(currHv[partId].value);

				if (inv_it == invLists[partId][len].end()) 
					continue;

				if(len <= D && clen <= D) {
					for(const auto &vid : inv_it->second) {
						if (currQuickRef[vid] || id <= vid) 
							continue;
						currQuickRef[vid] = true;
						currRes.emplace_back(vid);
						// assert(id < vid);
						// currPair.emplace_back(vid, id);
#if MAINTAIN_VALUE_EDIT == 0
						currPair.emplace_back(vid, id);
#elif MAINTAIN_VALUE_EDIT == 1
						int val = SimFuncs::levDist(work_dataset[vid], work_dataset[id]);
						if(currPair_.size() < maxHeapSize)
							currPair_.emplace_back(vid, id, val);
						else {
							if(isHeap[tid] == 0) {
								std::make_heap(currPair_.begin(), currPair_.end());
								isHeap[tid] = 1;
							}
							if(currPair_[0].val < val) {
								std::pop_heap(currPair_.begin(), currPair_.end());
								currPair_.pop_back();
								currPair_.emplace_back(vid, id, val);
								std::push_heap(currPair_.begin(), currPair_.end());
							}
						}
#endif
					}
					continue;
				}

				// verify
				// length aware verification
				right[tid] = (M + (stPos - Lo)) / 2;
				left[tid] = (M - (stPos - Lo)) / 2;
				_right[tid] = (_M + (clen - stPos - (len - Lo))) / 2;
				_left[tid] = (_M - (clen - stPos - (len - Lo))) / 2;

				// enumerate all element in inverted list
				auto vitP = invPre_it->second.begin();
				bool ifPrevRefLeft = false, ifPrevRefRight = false;

				for(const auto &vid : inv_it->second) {
					if (!currQuickRef[vid] && vid < id) {
#if APPROXIMATE == 2
						// approximate, 4-length prefix comparsion
						__m128i x_chars = _mm_loadu_si128((__m128i*)&work_dataset[vid][0]);
						__m128i y_chars = _mm_loadu_si128((__m128i*)&work_dataset[id][0]);
						__m128i cmp = _mm_cmpeq_epi8(x_chars, y_chars);
						int cmp_mask = _mm_movemask_epi8(cmp);
						if ((cmp_mask & 0xF) == 0xF) {
							++ vitP;
							ifPrevRefLeft = false;
							ifPrevRefRight = false;
							continue;
						}
#endif
#if LEAVE_EXACT_MATCH == 1
						if(workInvSC[vid] == workInvSC[id])
							continue;
#endif
						int leftSharing = (!ifPrevRefLeft) ? 0 : vitP->first;
						// int leftSharing = 0;
						if (M == 0 || verifyLeftPartSelf(vid, id, Lo, stPos, M, tid, leftSharing)) {
							ifPrevRefLeft = M == 0 ? false : true;
							int rightSharing = (!ifPrevRefRight) ? 0 : vitP->second;
							// int rightSharing = 0;

							if (_M == 0 || verifyRightPartSelf(vid, id, len - Lo - pLen,
								clen - stPos - pLen, Lo + pLen, stPos + pLen, _M, tid, rightSharing)) {
								ifPrevRefRight = _M == 0 ? false : true;
								currQuickRef[vid] = true;
								currRes.push_back(vid);
								// assert(id < vid);
								// currPair.emplace_back(vid, id);
#if MAINTAIN_VALUE_EDIT == 0
								currPair.emplace_back(vid, id);
#elif MAINTAIN_VALUE_EDIT == 1
								int val = SimFuncs::levDist(work_dataset[vid], work_dataset[id]);
								if(currPair_.size() < maxHeapSize)
									currPair_.emplace_back(vid, id, val);
								else {
									if(isHeap[tid] == 0) {
										std::make_heap(currPair_.begin(), currPair_.end());
										isHeap[tid] = 1;
									}
									if(currPair_[0].val < val) {
										std::pop_heap(currPair_.begin(), currPair_.end());
										currPair_.pop_back();
										currPair_.emplace_back(vid, id, val);
										std::push_heap(currPair_.begin(), currPair_.end());
									}
								}
#endif
							}
							else	
								ifPrevRefRight = false;
						}
						else {
							ifPrevRefLeft = false;
							ifPrevRefRight = false;
						}
					}
					else {
						ifPrevRefLeft = false;
						ifPrevRefRight = false;
					}

					++ vitP;
				}
			}

#if MAINTAIN_VALUE_EDIT == 0
			if(currPair.size() >= maxHeapSize)
				earlyTerminated[tid] = 1;
#endif
		}

		for(const auto &vid : currRes) {
			currQuickRef[vid] = false;
		}
		currRes.clear();
	}

	gettimeofday(&joinEnd, NULL);
	double joinTime = joinEnd.tv_sec - joinBeign.tv_sec + (joinEnd.tv_usec - joinBeign.tv_usec) / 1e6;
	printf("Edit join time: %.4lf\n", joinTime);

#if MAINTAIN_VALUE_EDIT == 0
	for(ui i = 0; i < MAXTHREADNUM; i++)
		finalPairs.insert(finalPairs.end(), pairs[i].begin(), pairs[i].end());
#elif MAINTAIN_VALUE_EDIT == 1
	for(int tid = 0; tid < MAXTHREADNUM; tid++)
		for(const auto &p : result_pairs_[tid])
			finalPairs.emplace_back(p.id1, p.id2);
#endif

	std::sort(finalPairs.begin(), finalPairs.end());
	auto fiter = std::unique(finalPairs.begin(), finalPairs.end());
	if(fiter != finalPairs.end()) {
		std::cerr << "Duplicate" << std::endl;
		exit(1);
	}
}


void StringJoinParallel::RSJoin(std::vector<std::pair<int, int>> &finalPairs)
{
#if TIMER_ON == 1
	timeval begin[MAXTHREADNUM], end[MAXTHREADNUM];
#endif

	init();
	printf("Finish init\n");
	fflush(stdout);

	prepareRS();
	printf("Finish preparing\n");
	fflush(stdout);

	// inverted index
	workInvSC = new int[workN];
	queryInvSC = new int[queryN];
	for(int i = 0; i < workN; i++)
		workInvSC[i] = std::distance(strCount.begin(), strCount.find(work_dataset[i]));
	for(int i = 0; i < queryN; i++)
		queryInvSC[i] = std::distance(strCount.begin(), strCount.find(query_dataset[i]));

	// index
	ui diffLengthSize = worklengthMap.size();
#pragma omp parallel for
	for(ui i = 0; i < diffLengthSize; i++) {
		ui sameLengthSize = worklengthMap[i].size();
		int clen = workLengthArray[i];

		for(ui j = 0; j < sameLengthSize; j++) {
			int wid = worklengthMap[i][j];

			for (int partId = 0; partId < PN; partId++) {
				int pLen = partLen[partId][clen];
				int stPos = partPos[partId][clen];
				uint64_t hashIndex = StringJoinUtil::strHash(work_dataset[wid], stPos, pLen);

				invLists[partId][clen][hashIndex].push_back(wid);

				if(invLists[partId][clen][hashIndex].size() == 1) 
					invListsPre[partId][clen][hashIndex].emplace_back(0, 0);
				else {
					int lidx = 0, leftSharing = 0;
					int ridx = 0, rightSharing = 0;
					std::string prevString = work_dataset[invLists[partId][clen][hashIndex][invLists[partId][clen][hashIndex].size() - 2]];

					while(prevString[lidx] == work_dataset[wid][lidx] && lidx < stPos) {
						++ leftSharing;
						++ lidx;
					}
					while(prevString[stPos + pLen + ridx] == work_dataset[wid][stPos + pLen + ridx] && ridx < clen - stPos - pLen) {
						++ rightSharing;
						++ ridx;
					}
					invListsPre[partId][clen][hashIndex].emplace_back(leftSharing, rightSharing);
				}
			}
		}
	}
	printf("Finish indexing\n");

	std::vector<int> results[MAXTHREADNUM];

	// current string, to generate substrings
	// omp_set_num_threads(1);
#pragma omp parallel for schedule(dynamic)
	for (int id = 0; id < queryN; id++) {

		int tid = omp_get_thread_num();
		int clen = query_dataset[id].length();

		auto &currQuickRef = quickRef[tid];
		auto &currHv = hv[tid];
		auto &currPair = pairs[tid];
#if MAINTAIN_VALUE_EDIT == 1
		auto &currPair_ = result_pairs_[tid];
#endif
		std::vector<int> currRes;

		// current part
		for (int partId = 0; partId < PN; partId++) {

			// threshold of left part M and right part _M
			int M = partId;
			int _M = D - partId;

			for (const auto &index : partIndex[partId][clen]) {
				int stPos = index.stPos;
				int Lo = index.Lo;
				int pLen = index.partLen;
				int len = index.len;

				currHv[partId].update(id, stPos, stPos + pLen - 1, query_dataset[id], power);
#if TIMER_ON == 1
				gettimeofday(&begin[tid], NULL);
#endif
				auto inv_it = invLists[partId][len].find(currHv[partId].value);
				auto invPre_it = invListsPre[partId][len].find(currHv[partId].value);
#if TIMER_ON == 1
				gettimeofday(&end[tid], NULL);
				double elapsedTime = end[tid].tv_sec - begin[tid].tv_sec + (end[tid].tv_usec - begin[tid].tv_usec) / 1e6;
				indexProbingCost[tid] += elapsedTime;
#endif

				if (inv_it == invLists[partId][len].end()) 
					continue;

				if(len <= D && clen <= D) {
					for(const auto &vid : inv_it->second) {
						if (currQuickRef[vid]) 
							continue;
						currQuickRef[vid] = true;
						currRes.emplace_back(vid);
						// currPair.emplace_back(id, vid);
#if MAINTAIN_VALUE_EDIT == 0
						currPair.emplace_back(id, vid);
#elif MAINTAIN_VALUE_EDIT == 1
						int val = SimFuncs::levDist(query_dataset[id], work_dataset[vid]);
						if(currPair_.size() < maxHeapSize)
							currPair_.emplace_back(id, vid, val);
						else {
							if(isHeap[tid] == 0) {
								std::make_heap(currPair_.begin(), currPair_.end());
								isHeap[tid] = 1;
							}
							if(currPair_[0].val < val) {
								std::pop_heap(currPair_.begin(), currPair_.end());
								currPair_.pop_back();
								currPair_.emplace_back(id, vid, val);
								std::push_heap(currPair_.begin(), currPair_.end());
							}
						}
#endif
					}
					continue;
				}

				// verify
				// length aware verification
				right[tid] = (M + (stPos - Lo)) / 2;
				left[tid] = (M - (stPos - Lo)) / 2;
				_right[tid] = (_M + (clen - stPos - (len - Lo))) / 2;
				_left[tid] = (_M - (clen - stPos - (len - Lo))) / 2;

				// enumerate all element in inverted list
				auto vitP = invPre_it->second.begin();
				bool ifPrevRefLeft = false, ifPrevRefRight = false;
#if VERIFY_PREFIX == 1
				string prevStr = "";
#endif 
				for(const auto &vid : inv_it->second) {
					if (!currQuickRef[vid]) {
						// iterative
						// if(lp == 0 && partId >= 1 && iterativeVerifyLeftPartRS(*vit, id, partPos[partId-1][len], Lo, stPos, len, M, partId) == false) {
						// 	cout << "Skip" << endl;
						// 	continue;
						// }

#if APPROXIMATE == 2
						// approximate, 4-length prefix comparsion
						__m128i x_chars = _mm_loadu_si128((__m128i*)&work_dataset[vid][0]);
						__m128i y_chars = _mm_loadu_si128((__m128i*)&query_dataset[id][0]);
						__m128i cmp = _mm_cmpeq_epi8(x_chars, y_chars);
						int cmp_mask = _mm_movemask_epi8(cmp);
						if ((cmp_mask & 0xF) == 0xF) {
							++ vitP;
							ifPrevRefLeft = false;
							ifPrevRefRight = false;
							continue;
						}
#endif
#if LEAVE_EXACT_MATCH == 1
						if(workInvSC[vid] == queryInvSC[id])
							continue;
#endif
						int leftSharing = (!ifPrevRefLeft) ? 0 : vitP->first;
						// int leftSharing = 0;

						if (M == 0 || verifyLeftPartRS(vid, id, Lo, stPos, M, tid, leftSharing)) {
							ifPrevRefLeft = M == 0 ? false : true;
							int rightSharing = (!ifPrevRefRight) ? 0 : vitP->second;
							// int rightSharing = 0;
#if VERIFY_PREFIX == 1
							if(leftSharing != 0 || rightSharing != 0) {
								int vlidx = 0, vleftSharing = 0;
								int vrdix = 0, vrightSharing = 0;
								while(prevStr[vlidx] == work_dataset[*vit][vlidx] && vlidx < Lo) {
									++ vleftSharing;
									++ vlidx;
								}
								while(prevStr[Lo + pLen + vrdix] == work_dataset[*vit][Lo + pLen + vrdix] && vrdix < len - Lo - pLen) {
									++ vrightSharing;
									++ vrdix;
								}
								if(vleftSharing != leftSharing) {
									printf("Different left sharing: %d %d\n", leftSharing, vleftSharing);
									exit(1);
								}
								if(vrightSharing != rightSharing) {
									printf("Different right sharing: %d %d\n", rightSharing, vrightSharing);
									exit(1);
								}
							}		
#endif
							if (_M == 0 || verifyRightPartRS(vid, id, len - Lo - pLen,
									clen - stPos - pLen, Lo + pLen, stPos + pLen, _M, tid, rightSharing)) {
								ifPrevRefRight = _M == 0 ? false : true;
								currQuickRef[vid] = true;
								currRes.emplace_back(vid);
								// currPair.emplace_back(id, vid);		
#if MAINTAIN_VALUE_EDIT == 0
								currPair.emplace_back(id, vid);
#elif MAINTAIN_VALUE_EDIT == 1
								int val = SimFuncs::levDist(query_dataset[id], work_dataset[vid]);
								if(currPair_.size() < maxHeapSize)
									currPair_.emplace_back(id, vid, val);
								else {
									if(isHeap[tid] == 0) {
										std::make_heap(currPair_.begin(), currPair_.end());
										isHeap[tid] = 1;
									}
									if(currPair_[0].val < val) {
										std::pop_heap(currPair_.begin(), currPair_.end());
										currPair_.pop_back();
										currPair_.emplace_back(id, vid, val);
										std::push_heap(currPair_.begin(), currPair_.end());
									}
								}
#endif
							}
							else	
								ifPrevRefRight = false;

						}
						else {
							ifPrevRefLeft = false;
							ifPrevRefRight = false;
						}
					}
					else {
						ifPrevRefLeft = false;
						ifPrevRefRight = false;
					}

#if VERIFY_PREFIX == 1
					prevStr = work_dataset[*vit];
#endif 
					++ vitP;
				}
			}
		}


		for(const auto &vit : currRes) {
			currQuickRef[vit] = false;
		}
		currRes.clear();

#if TIMER_ON == 1
		printf("Verifying cost: %.4lf\tProbing cost: %.4lf\n", verifyingCost[tid], indexProbingCost[tid]);
		fflush(stdout);
#endif
		// verifyingCost[tid] = 0.0;
		// indexProbingCost[tid] = 0.0;
	}

#if MAINTAIN_VALUE_EDIT == 0
	for(ui i = 0; i < MAXTHREADNUM; i++)
		finalPairs.insert(finalPairs.end(), pairs[i].begin(), pairs[i].end());
#elif MAINTAIN_VALUE_EDIT == 1
	for(int tid = 0; tid < MAXTHREADNUM; tid++)
		for(const auto &p : result_pairs_[tid])
			finalPairs.emplace_back(p.id1, p.id2);
#endif
}


void StringJoinParallel::checkSelfResults() const
{
	std::vector<std::pair<int, int>> truth;
	FILE *fp = fopen("../pass-join/pairs.txt", "r");

	ui size;
	fscanf(fp, "%d\n", &size);

	for(ui i = 0; i < size; i++) {
		ui id1, id2;
		fscanf(fp, "%d %d\n", &id1, &id2);
		truth.emplace_back(id1, id2);
	}

	fclose(fp);

	printf("Finish loading truth: %zu\n", truth.size());
	fflush(stdout);
}


void StringJoinParallel::printDebugInfo(int currLen) const 
{
    fprintf(stderr, "~~~~~~~~~~~ Current Length : %d ~~~~~~~~~~~\n", currLen);
    for (int pid = 0; pid < PN; pid++) {
    fprintf(stderr, "    #### Part Id : %d ####\n", pid);
    for (size_t lp = 0; lp < partIndex[pid][currLen].size(); lp++)
        fprintf(stderr, "stPos: %d\tendPos: %d\tpartLen: %d\tLo: %d\tlen: %d\n",
            partIndex[pid][currLen][lp].stPos,
            partIndex[pid][currLen][lp].stPos + partIndex[pid][currLen][lp].partLen,
            partIndex[pid][currLen][lp].partLen,
            partIndex[pid][currLen][lp].Lo,
            partIndex[pid][currLen][lp].len);
    }
}