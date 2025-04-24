/*
 * author: Dong Deng 
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/stringjoin.h"


void StringJoin::init()
{
	quickRef = new bool[N + 1];
	partPos = new int *[PN + 1];
	partLen = new int *[PN];
	invLists = new InvLists *[PN];
	partIndex = new std::vector<PIndex> *[PN];
	power = new uint64_t[maxDictLen + 1];

	for (int lp = 0; lp < PN; lp++) {
		partPos[lp] = new int[maxDictLen + 1];
		partLen[lp] = new int[maxDictLen + 1];
		invLists[lp] = new InvLists[maxDictLen + 1];
		partIndex[lp] = new std::vector<PIndex>[maxDictLen + 1];
	}
	partPos[PN] = new int[maxDictLen + 1];

	dist = new int[maxDictLen + 2];
	matrix = new int *[maxDictLen + 1];
	_matrix = new int *[maxDictLen + 1];

	for (int lp = 0; lp <= maxDictLen; lp++) {
		dist[lp] = N;
		matrix[lp] = new int[2 * D + 1];
		_matrix[lp] = new int[2 * D + 1];
	}
	dist[maxDictLen + 1] = N;

	// init
	for (int lp = 0; lp <= D; lp++)
		matrix[0][D + lp] = _matrix[0][D + lp] = lp;
	for (int lp = 0; lp <= N; lp++)
		quickRef[lp] = false;

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


void StringJoin::prepareSelf()
{
	int clen = 0;

	// fill in dist
	for (int id = 0; id < workN; id++) {
		if (clen == (int)work_dataset[id].length()) 
			continue;
		for (int lp = clen + 1; lp <= (int)work_dataset[id].length(); lp++) 
			dist[lp] = id;
		clen = (int)work_dataset[id].length();
	}

	// generate the set of pairs of segment and substring to be check
	clen = 0;
	for (int id = 0; id < workN; id++) {
		if (clen == (int)work_dataset[id].length()) 
			continue;
		// length of current string
		clen = work_dataset[id].length();

		for (int pid = 0; pid < PN; pid++) {
			// enumerate all lengths of string which potentially
			// similar to current string
			for (int len = std::max(clen - D, workMinDictLen); len <= clen; len++) {
				if (dist[len] == dist[len + 1]) 
					continue;

				// enumerate the valid substrings with respect to clen, len and pid
				for (int stPos = std::max(partPos[pid][len] - pid,
					partPos[pid][len] + (clen - len) - (D - pid));
					stPos <= std::min(partPos[pid][len] + pid,
					partPos[pid][len] + (clen - len) + (D - pid)); stPos++)
					partIndex[pid][clen].push_back(PIndex(
						stPos, partPos[pid][len], partLen[pid][len], len));
			}
			// sort the pairs of segment and substring to be check,
			// for conveniently hash substring and segment
			sort(partIndex[pid][clen].begin(), partIndex[pid][clen].end(), StringJoinUtil::PIndexLess);
		}
	}
}


void StringJoin::prepareRS()
{
	int clen = 0;

	// fill in dist
	for (int id = 0; id < workN; id++) {
		if (clen == (int)work_dataset[id].length()) 
			continue;
		for (int lp = clen + 1; lp <= (int)work_dataset[id].length(); lp++) 
			dist[lp] = id;
		clen = (int)work_dataset[id].length();
	}

	// generate the set of pairs of segment and substring to be check
	clen = 0;
	for (int id = 0; id < queryN; id++) {
		if (clen == (int)query_dataset[id].length()) 
			continue;
		// length of current string
		clen = (int)query_dataset[id].length();

		for (int pid = 0; pid < PN; pid++) {
			// enumerate all lengths of string which potentially
			// similar to current string
			for (int len = std::max(clen - D, workMinDictLen); len <= std::min(clen + D, workMaxDictLen); len++) {
				if (dist[len] == dist[len + 1]) 
					continue;

				// enumerate the valid substrings with respect to clen, len and pid
				int delta = clen - len;
				for (int stPos = std::max(partPos[pid][len] - pid,
					partPos[pid][len] + delta - (D - pid));
					stPos <= std::min(partPos[pid][len] + pid,
					partPos[pid][len] + delta + (D - pid)); stPos++)
					partIndex[pid][clen].push_back(PIndex(
						stPos, partPos[pid][len], partLen[pid][len], len));
			}
			// sort the pairs of segment and substring to be check,
			// for conveniently hash substring and segment
			sort(partIndex[pid][clen].begin(), partIndex[pid][clen].end(), StringJoinUtil::PIndexLess);
		}
	}
}


bool StringJoin::verifyLeftPartSelf(int xid, int yid, int xlen, int ylen, int Tau)
{
	const auto &workString = work_dataset[xid];
	const auto &queryString = work_dataset[yid];

	for (int i = 1; i <= xlen; i++) {
		valid = 0;
		if (i <= left) {
			matrix[i][D - i] = i;
			valid = 1;
		}

		int val1 = i - left;
		int val2 = i + right;
		int lowerBound = std::max(val1, 1);
		int upperBound = std::min(val2, ylen);

		for (int j = lowerBound; j <= upperBound; j++) {
			int val = j - i + D;
			if (workString[i - 1] == queryString[j - 1])
				matrix[i][val] = matrix[i - 1][val];
			else
				matrix[i][val] = StringJoinUtil::min(matrix[i - 1][val],
					j - 1 >= val1 ? matrix[i][val - 1] : D,
					j + 1 <= val2 ? matrix[i - 1][val + 1] : D) + 1;
			if (abs(xlen - ylen - i + j) + matrix[i][val] <= Tau) 
				valid = 1;
		}

		if (!valid) 
			return false;
  	}	

  	return matrix[xlen][ylen - xlen + D] <= Tau;
}


bool StringJoin::verifyRightPartSelf(int xid, int yid, int xlen, int ylen, int xpos, int ypos, int Tau)
{	
	const auto &workString = work_dataset[xid];
	const auto &queryString = work_dataset[yid];

	for (int i = 1; i <= xlen; i++) {
		valid = 0;
		if (i <= _left) {
			_matrix[i][D - i] = i;
			valid = 1;
		}

		int val1 = i - _left;
		int val2 = i + _right;
		int lowerBound = std::max(val1, 1);
		int upperBound = std::min(val2, ylen);

		for (int j = lowerBound; j <= upperBound; j++) {
			int val = j - i + D;
			if (workString[xpos + i - 1] == queryString[ypos + j - 1])
				_matrix[i][val] = _matrix[i - 1][val];
			else
				_matrix[i][val] = StringJoinUtil::min(_matrix[i - 1][val],
					j - 1 >= val1 ? _matrix[i][val - 1] : D,
					j + 1 <= val2 ? _matrix[i - 1][val + 1] : D) + 1;
			if (abs(xlen - ylen - i + j) + _matrix[i][val] <= Tau) 
				valid = 1;
		}

		if (!valid) 
			return false;
  	}

  	return _matrix[xlen][ylen - xlen + D] <= Tau;
}


bool StringJoin::verifyLeftPartRS(int xid, int yid, int xlen, int ylen, int Tau)
{
	const auto &workString = work_dataset[xid];
	const auto &queryString = query_dataset[yid];

	for (int i = 1; i <= xlen; i++) {
		valid = 0;
		if (i <= left) {
			matrix[i][D - i] = i;
			valid = 1;
		}

		int val1 = i - left;
		int val2 = i + right;
		int lowerBound = std::max(val1, 1);
		int upperBound = std::min(val2, ylen);

		for (int j = lowerBound; j <= upperBound; j++) {
			int val = j - i + D;
			if (workString[i - 1] == queryString[j - 1])
				matrix[i][val] = matrix[i - 1][val];
			else
				matrix[i][val] = StringJoinUtil::min(matrix[i - 1][val],
					j - 1 >= val1 ? matrix[i][val - 1] : D,
					j + 1 <= val2 ? matrix[i - 1][val + 1] : D) + 1;
			if (abs(xlen - ylen - i + j) + matrix[i][val] <= Tau) 
				valid = 1;
		}

		if (!valid) 
			return false;
	}

	return matrix[xlen][ylen - xlen + D] <= Tau;
}


bool StringJoin::verifyRightPartRS(int xid, int yid, int xlen, int ylen, int xpos, int ypos, int Tau)
{	
	const auto &workString = work_dataset[xid];
	const auto &queryString = query_dataset[yid];

	for (int i = 1; i <= xlen; i++) {
		valid = 0;
		if (i <= _left) {
			_matrix[i][D - i] = i;
			valid = 1;
		}

		int val1 = i - _left;
		int val2 = i + _right;
		int lowerBound = std::max(val1, 1);
		int upperBound = std::min(val2, ylen);

		for (int j = lowerBound; j <= upperBound; j++) {
			int val = j - i + D;
			if (workString[xpos + i - 1] == queryString[ypos + j - 1])
				_matrix[i][val] = _matrix[i - 1][val];
			else
				_matrix[i][val] = StringJoinUtil::min(_matrix[i - 1][val],
					j - 1 >= val1 ? _matrix[i][val - 1] : D,
					j + 1 <= val2 ? _matrix[i - 1][val + 1] : D) + 1;
			if (abs(xlen - ylen - i + j) + _matrix[i][val] <= Tau) 
				valid = 1;
		}

		if (!valid) 
			return false;
	}

	return _matrix[xlen][ylen - xlen + D] <= Tau;
}


void StringJoin::selfJoin(std::vector<std::pair<int, int>> &finalPairs)
{
	hashValue hv[PN];
	int len, stPos, pLen, Lo;

	printf("%d %d %d\n", workN, workMinDictLen, workMaxDictLen);

	init();

	prepareSelf();

	// index current string
	for(int id = 0; id < workN; id++) {
		int clen = (int)work_dataset[id].length();
		for (int partId = 0; partId < PN; partId++) {
			pLen = partLen[partId][clen];
			stPos = partPos[partId][clen];
			invLists[partId][clen][StringJoinUtil::strHash(work_dataset[id], stPos, pLen)].push_back(id);
		}
	}

	// current string, to generate substrings
	for (int id = 0; id < workN; id++) {
		int clen = work_dataset[id].length();
		// current part
		for (int partId = 0; partId < PN; partId++) {
			// threshold of left part M and right part _M
			int M = partId;
			int _M = D - partId;

			for (const auto &index : partIndex[partId][clen]) {
				stPos = index.stPos;
				Lo = index.Lo;
				pLen = index.partLen;
				len = index.len;

				hv[partId].update(id, stPos, stPos + pLen - 1, work_dataset[id], power);
				auto inv_it = invLists[partId][len].find(hv[partId].value);

				if (inv_it == invLists[partId][len].end()) 
					continue;

				// verify
				// length aware verification
				right = (M + (stPos - Lo)) / 2;
				left = (M - (stPos - Lo)) / 2;
				_right = (_M + (clen - stPos - (len - Lo))) / 2;
				_left = (_M - (clen - stPos - (len - Lo))) / 2;

				// enumerate all element in inverted list
				for(const auto &vid : inv_it->second) {
					if (!quickRef[vid] && id > vid) {
						++candNum;
						if (M == 0 || verifyLeftPartSelf(vid, id, Lo, stPos, M)) {
							++veriNum;
							if (_M == 0 || verifyRightPartSelf(vid, id, len - Lo - pLen,
								clen - stPos - pLen, Lo + pLen, stPos + pLen, _M)) {
								quickRef[vid] = true;
#if MAINTAIN_VALUE_EDIT == 0
								finalPairs.emplace_back(vid, id);
								if(finalPairs.size() >= maxHeapSize)
									return;
#elif MAINTAIN_VALUE_EDIT == 1
								int val = SimFuncs::levDist(work_dataset[id], work_dataset[vid]);
								if(result_pairs_.size() < maxHeapSize)
									result_pairs_.emplace_back(vid, id, val);
								else {
									if(isHeap == 0) {
										std::make_heap(result_pairs_.begin(), result_pairs_.end());
										isHeap = 1;
									}
									if(result_pairs_[0].val < val) {
										std::pop_heap(result_pairs_.begin(), result_pairs_.end());
										result_pairs_.pop_back();
										result_pairs_.emplace_back(vid, id, val);
										std::push_heap(result_pairs_.begin(), result_pairs_.end());
									}
								}
#endif
								results.emplace_back(vid);
							}
						}
					}
				}
			}
		}

		for(const auto &vid : results)
			quickRef[vid] = false;
		results.clear();
	}

#if MAINTAIN_VALUE_EDIT == 1
	for(const auto &p : result_pairs_)
		finalPairs.emplace_back(p.id1, p.id2);
#endif
	
	// printf("%d %d %d\n", listNum, candNum, veriNum);
	std::cout << listNum << " " << candNum << " " << veriNum << std::endl;
}


void StringJoin::RSJoin(std::vector<std::pair<int, int>> &finalPairs)
{
	hashValue hv[PN];
	int len, stPos, pLen, Lo;

	init();
	prepareRS();

	// index
	for (int id = 0; id < workN; id++) {
		int clen = work_dataset[id].length();
		// index current string
		for (int partId = 0; partId < PN; partId++) {
			pLen = partLen[partId][clen];
			stPos = partPos[partId][clen];
			invLists[partId][clen][StringJoinUtil::strHash(work_dataset[id], stPos, pLen)].emplace_back(id);
		}
	}

	// current string, to generate substrings
	for (int id = 0; id < queryN; id++) {
		int clen = (int)query_dataset[id].length();
		// current part
		for (int partId = 0; partId < PN; partId++) {
			// threshold of left part M and right part _M
			int M = partId;
			int _M = D - partId;
			for (const auto &index : partIndex[partId][clen]) {
				stPos = index.stPos;
				Lo = index.Lo;
				pLen = index.partLen;
				len = index.len;

				hv[partId].update(id, stPos, stPos + pLen - 1, query_dataset[id], power);
				auto inv_it = invLists[partId][len].find(hv[partId].value);

				if (inv_it == invLists[partId][len].end()) 
					continue;

				// verify
				// length aware verification
				right = (M + (stPos - Lo)) / 2;
				left = (M - (stPos - Lo)) / 2;
				_right = (_M + (clen - stPos - (len - Lo))) / 2;
				_left = (_M - (clen - stPos - (len - Lo))) / 2;

				// enumerate all element in inverted list
				for (const auto &vid : inv_it->second) {
					if (!quickRef[vid]) {
						if (M == 0 || verifyLeftPartRS(vid, id, Lo, stPos, M)) {
							if (_M == 0 || verifyRightPartRS(vid, id, len - Lo - pLen,
								clen - stPos - pLen, Lo + pLen, stPos + pLen, _M)) {
								quickRef[vid] = true;
#if MAINTAIN_VALUE_EDIT == 0
								finalPairs.emplace_back(id, vid);
								if(finalPairs.size() >= maxHeapSize)
									return;
#elif MAINTAIN_VALUE_EDIT == 1
								int val = SimFuncs::levDist(query_dataset[id], work_dataset[vid]);
								if(result_pairs_.size() < maxHeapSize)
									result_pairs_.emplace_back(id, vid, val);
								else {
									if(isHeap == 0) {
										std::make_heap(result_pairs_.begin(), result_pairs_.end());
										isHeap = 1;
									}
									if(result_pairs_[0].val < val) {
										std::pop_heap(result_pairs_.begin(), result_pairs_.end());
										result_pairs_.pop_back();
										result_pairs_.emplace_back(id, vid, val);
										std::push_heap(result_pairs_.begin(), result_pairs_.end());
									}
								}
#endif
								results.push_back(vid);
							}
						}
					}
				}
			}
		}

		for(const auto &vid : results)
			quickRef[vid] = false;
		results.clear();
	}

#if MAINTAIN_VALUE_EDIT == 1
	for(const auto &p : result_pairs_)
		finalPairs.emplace_back(p.id1, p.id2);
#endif
	
	// printf("%d %d %d\n", listNum, candNum, veriNum);
	std::cout << listNum << " " << candNum << " " << veriNum << std::endl;
}


void StringJoin::checkSelfResults() const
{
	std::vector<std::pair<int, int>> truth;
	FILE *fp = fopen("../pass-join/pairs.txt", "r");

	int size;
	fscanf(fp, "%d\n", &size);

	for(int i = 0; i < size; i++) {
		int id1, id2;
		fscanf(fp, "%d %d\n", &id1, &id2);
		truth.emplace_back(id1, id2);
	}

	fclose(fp);

	printf("Finish loading truth: %zu\n", truth.size());
	fflush(stdout);

	for(ui i = 0; i < truth.size(); i++) {
		if(std::find(pairs.begin(), pairs.end(), truth[i]) == pairs.end()) 
			printf("%d %d %s %s\n", truth[i].first, truth[i].second, 
									work_dataset[truth[i].first].c_str(), work_dataset[truth[i].second].c_str());
	}
}


void StringJoin::printDebugInfo(int currLen) const 
{
	fprintf(stderr, "~~~~~~~~~~~ Current Length : %d ~~~~~~~~~~~\n", currLen);
	for (int pid = 0; pid < PN; pid++) {
	fprintf(stderr, "    #### Part Id : %d ####\n", pid);
	for (int lp = 0; lp < partIndex[pid][currLen].size(); lp++)
		fprintf(stderr, "stPos: %d\tendPos: %d\tpartLen: %d\tLo: %d\tlen: %d\n",
			partIndex[pid][currLen][lp].stPos,
			partIndex[pid][currLen][lp].stPos + partIndex[pid][currLen][lp].partLen,
			partIndex[pid][currLen][lp].partLen,
			partIndex[pid][currLen][lp].Lo,
			partIndex[pid][currLen][lp].len);
	}
}	