/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "blocker/blocker_util.h"


void BlockerUtil::mergePairs(const std::vector<std::vector<int>> &bucket)
{
#pragma omp parallel for
	for(int i = 0; i < table_A.row_no; i++) {
		ui size = bucket[i].size();
		for(ui j = 0; j < size; j++) {
			int val = bucket[i][j];
			auto it2 = std::lower_bound(final_pairs[i].begin(), final_pairs[i].end(), val);
			
			if(it2 == final_pairs[i].end()) {
				final_pairs[i].emplace_back(val);
				passedRules[i].emplace_back(val, 1);
			}
			else if(*it2 != val) {
				ui idx = std::distance(final_pairs[i].begin(), it2);
				final_pairs[i].insert(it2, val);	
				passedRules[i].insert(passedRules[i].begin() + idx, std::make_pair(val, 1));
			}
			else {
				ui idx = std::distance(final_pairs[i].begin(), it2);
				++ passedRules[i][idx].second;
			}
		}
	}
}


// Optimize: first sort then unique
// mapid: 0 for id_map; 1 for idStringMap
void BlockerUtil::synthesizePairsRS(ui pos, std::vector<std::pair<int, int>> &pairs, int mapid)
{
	size_t size = pairs.size();

	std::vector<std::vector<int>> bucket;
	bucket.resize(table_A.row_no);

	if(mapid == 0) {
		for(size_t i = 0; i < size; i++) {
			auto id1 = (int)id_mapA[pos][pairs[i].first];
			auto id2 = (int)id_mapB[pos][pairs[i].second];
			bucket[id1].emplace_back(id2);
		}
	}
	else if(mapid == 1) {
		for(size_t i = 0; i < size; i++) {
			auto id1 = (int)idStringMapA[pos][pairs[i].first];
			auto id2 = (int)idStringMapB[pos][pairs[i].second];
			bucket[id1].emplace_back(id2);
		}
	}

#pragma omp parallel for
	for(int i = 0; i < table_A.row_no; i++)
		std::sort(bucket[i].begin(), bucket[i].end());

	BlockerUtil::mergePairs(bucket);
	printf("~~~Merging results~~~\n");
}


void BlockerUtil::synthesizePairsSelf(ui pos, std::vector<std::pair<int, int>> &pairs, int mapid)
{
    size_t size = pairs.size();

	std::vector<std::vector<int>> bucket;
	bucket.resize(table_A.row_no);

    if(mapid == 0) {
        for(size_t i = 0; i < size; i++) {
            auto id1 = (int)id_mapA[pos][pairs[i].first];
            auto id2 = (int)id_mapA[pos][pairs[i].second];
			int minId = std::min(id1, id2);
			int maxId = std::max(id1, id2);
            bucket[minId].emplace_back(maxId);
            assert(id1 != id2);
        }
    }
    else if(mapid == 1) {
        for(size_t i = 0; i < size; i++) {
            auto id1 = (int)idStringMapA[pos][pairs[i].first];
            auto id2 = (int)idStringMapA[pos][pairs[i].second];
            int minId = std::min(id1, id2);
			int maxId = std::max(id1, id2);
            bucket[minId].emplace_back(maxId);
            assert(id1 != id2);
        }
    }

#pragma omp parallel for
	for(int i = 0; i < table_A.row_no; i++) {
		std::sort(bucket[i].begin(), bucket[i].end());
		auto iter = std::unique(bucket[i].begin(), bucket[i].end());
		if(iter != bucket[i].end()) {
			std::cerr << "duplicate in bucket" << std::endl;
			exit(1);
		}
	}

	BlockerUtil::mergePairs(bucket);
	printf("~~~Merging results~~~\n");
}


void BlockerUtil::pretopKviaTASelf(uint64_t K, const std::string &topKattr, const std::string &attrType, 
						           bool isWeighted)
{
	uint64_t numEntity = 0;

	for(const auto &fv : final_pairs)
		numEntity += (uint64_t)fv.size();

	std::cout << "total entities: " << numEntity << std::endl;

	if(numEntity <= MAX_TOTAL_SIZE)
		return;

	printf("~~~Triggering Top K~~~\n");
	std::ofstream tmpStream;
	TopK::topKviaTASelf(table_A, topKattr, attrType, recordsA, 
						weightsA, wordwt, datasets_map, id_mapA, 
						final_pairs, Table (), K, tmpStream,
						isWeighted, "replace");
}


void BlockerUtil::pretopKviaTARS(uint64_t K, const std::string &topKattr, const std::string &attrType, 
						         bool isWeighted)
{
	uint64_t numEntity = 0;

	for(const auto &fv : final_pairs)
		numEntity += (uint64_t)fv.size();

	if(numEntity <= MAX_TOTAL_SIZE)
		return;

	printf("~~~Triggering Top K~~~\n");
	std::ofstream tmpStream;
	TopK::topKviaTARS(table_A, table_B, topKattr, attrType, recordsA, recordsB, 
					  weightsA, weightsB, wordwt, datasets_map, id_mapA, id_mapB, 
					  final_pairs, Table (), K, tmpStream, isWeighted, "replace");
} 