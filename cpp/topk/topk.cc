/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "topk/topk.h"


// implementation
void TopK::allocateBuffers(uint64_t numEntity, ui numDimension, TableEntry **&valueTable, double **&backup)
{
	valueTable = new TableEntry*[numDimension];
	for(ui i = 0; i < numDimension; i++)
		valueTable[i] = new TableEntry[numEntity];
	backup = new double*[numDimension];
	for(ui i = 0; i < numDimension; i++)
		backup[i] = new double[numEntity];
}


void TopK::releaseBuffers(ui numDimension, TableEntry **&valueTable, double **&backup)
{
	for(ui i = 0; i < numDimension; i++)
		delete[] valueTable[i];
	delete[] valueTable;
	for(ui i = 0; i < numDimension; i++)
		delete[] backup[i];
	delete[] backup;
}


void TopK::prepareSelf(const std::vector<std::vector<ui>> &records, const std::vector<ui> &id_map, 
					   const std::vector<std::vector<int>> &final_pairs, 
					   std::vector<std::pair<int, int>> &id2Pair, 
					   TableEntry **&valueTable, double **&backup,
					   ui numRow, uint64_t &numEntity)
{
	// get reverse id map
	std::vector<ui> reverseIdMap(numRow, 0);
	for(ui i = 0; i < numRow; i++)
		reverseIdMap[id_map[i]] = i;

	// get pair to new id
	for(ui i = 0; i < numRow; i++) {
		int id1 = (int)reverseIdMap[i];
		if(records[id1].empty())
			continue;
		for(const auto &e : final_pairs[i]) {
			int id2 = (int)reverseIdMap[e];
			if(!records[id2].empty()) {
				++ numEntity;
				id2Pair.emplace_back(id1, id2);
			}
		}
	}

	// fill in table
	allocateBuffers(numEntity, 4, valueTable, backup);
	std::vector<int> ovlpval(numEntity, 0);

#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++)
		ovlpval[i] = SimFuncs::overlap(records[id2Pair[i].first], records[id2Pair[i].second]);
#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++) {
		int id1 = id2Pair[i].first;
		int id2 = id2Pair[i].second;
		auto &e1 = valueTable[0][i];
		auto &e2 = valueTable[1][i];
		auto &e3 = valueTable[2][i];
		auto &e4 = valueTable[3][i];

		e1.id = i;
		e1.val = SimFuncs::jaccard(records[id1], records[id2], ovlpval[i]);
		backup[0][i] = e1.val;

		e2.id = i;
		e2.val = SimFuncs::cosine(records[id1], records[id2], ovlpval[i]);
		backup[1][i] = e2.val;

		e3.id = i;
		e3.val = SimFuncs::dice(records[id1], records[id2], ovlpval[i]);
		backup[2][i] = e3.val;

		e4.id = i;
		e4.val = SimFuncs::overlapCoeff(records[id1], records[id2], ovlpval[i]);
		backup[3][i] = e4.val;
	}
	
	// sort
	for(ui i = 0; i < 4; i++) 
		__gnu_parallel::sort(valueTable[i], valueTable[i] + numEntity, 
		[](const TableEntry &e1, const TableEntry &e2) {
			return e1.val > e2.val;
		});
}


void TopK::prepareSelfWeighted(const std::vector<std::vector<ui>> &records, const std::vector<ui> &id_map, 
							   const std::vector<double> &recWeights, const std::vector<double> &wordwt, 
							   const std::vector<std::vector<int>> &final_pairs, 
							   std::vector<std::pair<int, int>> &id2Pair, 	
							   TableEntry **&valueTable, double **&backup,
							   ui numRow, uint64_t &numEntity)
{
	// get reverse id map
	std::vector<ui> reverseIdMap(numRow, 0);
	for(ui i = 0; i < numRow; i++)
		reverseIdMap[id_map[i]] = i;

	// get pair to new id
	for(ui i = 0; i < numRow; i++) {
		int id1 = (int)reverseIdMap[i];
		if(records[id1].empty())
			continue;
		for(const auto &e : final_pairs[i]) {
			int id2 = (int)reverseIdMap[e];
			if(!records[id2].empty()) {
				++ numEntity;
				id2Pair.emplace_back(id1, id2);
			}
		}
	}

	// fill in table
	allocateBuffers(numEntity, 4, valueTable, backup);
	std::vector<double> ovlpval(numEntity, 0);

#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++)
		ovlpval[i] = SimFuncs::weightedOverlap(records[id2Pair[i].first], records[id2Pair[i].second], wordwt);
#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++) {
		int id1 = id2Pair[i].first;
		int id2 = id2Pair[i].second;
		auto &e1 = valueTable[0][i];
		auto &e2 = valueTable[1][i];
		auto &e3 = valueTable[2][i];
		auto &e4 = valueTable[3][i];
		double metric1 = 0.0;
		double metric2 = 0.0;

		e1.id = i;
		metric1 = SimFuncs::weightedJaccard(recWeights[id1], recWeights[id2], ovlpval[i]);
		metric2 = SimFuncs::jaccard(records[id1], records[id2]);
		e1.val = std::max(metric1, metric2);
		backup[0][i] = e1.val;

		e2.id = i;
		metric1 = SimFuncs::weightedCosine(recWeights[id1], recWeights[id2], ovlpval[i]);
		metric2 = SimFuncs::cosine(records[id1], records[id2]);
		e2.val = std::max(metric1, metric2);
		backup[1][i] = e2.val;

		e3.id = i;
		metric1 = SimFuncs::weightedDice(recWeights[id1], recWeights[id2], ovlpval[i]);
		metric2 = SimFuncs::dice(records[id1], records[id2]);
		e3.val = std::max(metric1, metric2);
		backup[2][i] = e3.val;

		e4.id = i;
		metric1 = SimFuncs::weightedOverlapCoeff(recWeights[id1], recWeights[id2], ovlpval[i]);
		metric2 = SimFuncs::dice(records[id1], records[id2]);
		e4.val = std::max(metric1, metric2);
		backup[3][i] = e4.val;
	}
	
	// sort
	for(ui i = 0; i < 4; i++) 
		__gnu_parallel::sort(valueTable[i], valueTable[i] + numEntity, 
		[](const TableEntry &e1, const TableEntry &e2) {
			return e1.val > e2.val;
		});
}


void TopK::prepareRS(const std::vector<std::vector<ui>> &recordsA, const std::vector<std::vector<ui>> &recordsB, 
					 const std::vector<ui> &id_mapA, const std::vector<ui> &id_mapB, 
					 const std::vector<std::vector<int>> &final_pairs, 
					 std::vector<std::pair<int, int>> &id2Pair, 
					 TableEntry **&valueTable, double **&backup,
					 ui numRowA, ui numRowB, uint64_t &numEntity)
{
    // get reverse id map
	std::vector<ui> reverseIdMapA(numRowA, 0);
	std::vector<ui> reverseIdMapB(numRowB, 0);
	for(ui i = 0; i < numRowA; i++)
		reverseIdMapA[id_mapA[i]] = i;
	for(ui i = 0; i < numRowB; i++)
		reverseIdMapB[id_mapB[i]] = i;

	// get pair to new id
	for(ui i = 0; i < numRowA; i++) {
		int id1 = (int)reverseIdMapA[i];
		if(recordsA[id1].empty())
			continue;
		for(const auto &e : final_pairs[i]) {
			int id2 = (int)reverseIdMapB[e];
			if(!recordsB[id2].empty()) {
				++ numEntity;
				id2Pair.emplace_back(id1, id2);
			}
		}
	}

	// fill in table
	allocateBuffers(numEntity, 4, valueTable, backup);
	std::vector<int> ovlpval(numEntity, 0);

#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++)
		ovlpval[i] = SimFuncs::overlap(recordsA[id2Pair[i].first], recordsB[id2Pair[i].second]);
#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++) {
		int id1 = id2Pair[i].first;
		int id2 = id2Pair[i].second;
		auto &e1 = valueTable[0][i];
		auto &e2 = valueTable[1][i];
		auto &e3 = valueTable[2][i];
		auto &e4 = valueTable[3][i];

		e1.id = i;
		e1.val = SimFuncs::jaccard(recordsA[id1], recordsB[id2], ovlpval[i]);
		backup[0][i] = e1.val;

		e2.id = i;
		e2.val = SimFuncs::cosine(recordsA[id1], recordsB[id2], ovlpval[i]);
		backup[1][i] = e2.val;

		e3.id = i;
		e3.val = SimFuncs::dice(recordsA[id1], recordsB[id2], ovlpval[i]);
		backup[2][i] = e3.val;

		e4.id = i;
		e4.val = SimFuncs::overlapCoeff(recordsA[id1], recordsB[id2], ovlpval[i]);
		backup[3][i] = e4.val;
	}
	
	// sort
	for(ui i = 0; i < 4; i++) 
		__gnu_parallel::sort(valueTable[i], valueTable[i] + numEntity, 
		[](const TableEntry &e1, const TableEntry &e2) {
			return e1.val > e2.val;
		});
}


void TopK::prepareRSWeighted(const std::vector<std::vector<ui>> &recordsA, const std::vector<std::vector<ui>> &recordsB, 
							 const std::vector<ui> &id_mapA, const std::vector<ui> &id_mapB, 
							 const std::vector<double> &recWeightsA, const std::vector<double> &recWeightsB,
							 const std::vector<double> &wordwt, 
							 const std::vector<std::vector<int>> &final_pairs, 
							 std::vector<std::pair<int, int>> &id2Pair, 
							 TableEntry **&valueTable, double **&backup, 
							 ui numRowA, ui numRowB, uint64_t &numEntity)
{
	// get reverse id map
	std::vector<ui> reverseIdMapA(numRowA, 0);
	std::vector<ui> reverseIdMapB(numRowB, 0);
	for(ui i = 0; i < numRowA; i++)
		reverseIdMapA[id_mapA[i]] = i;
	for(ui i = 0; i < numRowB; i++)
		reverseIdMapB[id_mapB[i]] = i;

	// get pair to new id
	for(ui i = 0; i < numRowA; i++) {
		int id1 = (int)reverseIdMapA[i];
		if(recordsA[id1].empty())
			continue;
		for(const auto &e : final_pairs[i]) {
			int id2 = (int)reverseIdMapB[e];
			if(!recordsB[id2].empty()) {
				++ numEntity;
				id2Pair.emplace_back(id1, id2);
			}
		}
	}

	// fill in table
	allocateBuffers(numEntity, 4, valueTable, backup);
	std::vector<int> ovlpval(numEntity, 0);
	std::vector<double> ovlpvalw(numEntity, 0.0);

#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++) {
		ovlpval[i] = SimFuncs::overlap(recordsA[id2Pair[i].first], recordsB[id2Pair[i].second]);
		ovlpvalw[i] = SimFuncs::weightedOverlap(recordsA[id2Pair[i].first], recordsB[id2Pair[i].second], wordwt);
	}
#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++) {
		int id1 = id2Pair[i].first;
		int id2 = id2Pair[i].second;
		auto &e1 = valueTable[0][i];
		auto &e2 = valueTable[1][i];
		auto &e3 = valueTable[2][i];
		auto &e4 = valueTable[3][i];
		double metric1 = 0.0;
		double metric2 = 0.0;

		e1.id = i;
		metric1 = SimFuncs::weightedJaccard(recWeightsA[id1], recWeightsB[id2], ovlpvalw[i]);
		// metric2 = SimFuncs::jaccard(recordsA[id1], recordsB[id2], ovlpval[i]);
		metric2 = 0.0;
		e1.val = std::max(metric1, metric2);
		backup[0][i] = e1.val;

		e2.id = i;
		metric1 = SimFuncs::weightedCosine(recWeightsA[id1], recWeightsB[id2], ovlpvalw[i]);
		// metric2 = SimFuncs::cosine(recordsA[id1], recordsB[id2], ovlpval[i]);
		metric2 = 0.0;
		e2.val = std::max(metric1, metric2);
		backup[1][i] = e2.val;

		e3.id = i;
		metric1 = SimFuncs::weightedDice(recWeightsA[id1], recWeightsB[id2], ovlpvalw[i]);
		// metric2 = SimFuncs::dice(recordsA[id1], recordsB[id2], ovlpval[i]);
		metric2 = 0.0;
		e3.val = std::max(metric1, metric2);
		backup[2][i] = e3.val;

		e4.id = i;
		metric1 = SimFuncs::weightedOverlapCoeff(recWeightsA[id1], recWeightsB[id2], ovlpvalw[i]);
		// metric2 = SimFuncs::dice(recordsA[id1], recordsB[id2], ovlpval[i]);
		metric2 = 0.0;
		e4.val = std::max(metric1, metric2);
		backup[3][i] = e4.val;
	}
	
	// sort
	for(ui i = 0; i < 4; i++) 
		__gnu_parallel::sort(valueTable[i], valueTable[i] + numEntity, 
		[](const TableEntry &e1, const TableEntry &e2) {
			return e1.val > e2.val;
		});
}


/*
 * TA
 */
void TopK::updateFinalPairs(const std::vector<ui> &topKidMapA, const std::vector<ui> &topKidMapB, const std::vector<std::pair<int, int>> &id2Pair,
							std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> &q,
							std::vector<std::vector<int>> &final_pairs, uint64_t K)
{
	for(auto &fv : final_pairs)
		fv.clear();
	for(uint64_t i = 0; i < K; i++) {
		int id1 = topKidMapA[id2Pair[q.top().id].first];
		int id2 = topKidMapB[id2Pair[q.top().id].second];
		final_pairs[id1].emplace_back(id2);
		q.pop();
	}
	// sort
	for(auto &fv : final_pairs)
		std::sort(fv.begin(), fv.end());
}


void TopK::getCurrentRecall(const std::vector<ui> &topKidMapA, const std::vector<ui> &topKidMapB, const std::vector<std::pair<int, int>> &id2Pair,
							std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> &q,
							const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian)
{
	std::vector<std::pair<int, int>> tmp;
	for(uint64_t i = 0; i < K; i++) {
		int id1 = topKidMapA[id2Pair[q.top().id].first];
		int id2 = topKidMapB[id2Pair[q.top().id].second];
		tmp.emplace_back(id1, id2);
		q.pop();
	}

	auto tmpGolds = golds;

	__gnu_parallel::sort(tmpGolds.begin(), tmpGolds.end());
	__gnu_parallel::sort(tmp.begin(), tmp.end());
	std::vector<std::pair<int, int>> internal;
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp.begin(), tmp.end(), 
									 std::back_inserter(internal));

	std::cout << "-------------- recall on representative attribute threshold accepting --------------" << std::endl;
	std::cout << "     K: " << K << std::endl;
	std::cout << std::setprecision(4) << "recall: " << internal.size() * 1.0 / golds.size() << std::endl;
	std::cout << "   |C|: " << tmp.size() << std::endl;
	std::cout << std::setprecision(4) << "  CSSR: " << tmp.size() * 1.0 / cartesian * 1.0 << std::endl;
	std::cout << "-------------- recall on representative attribute threshold accepting --------------" << std::endl << std::flush;
}


void TopK::getCurrentRecallExp(const std::vector<ui> &topKidMapA, const std::vector<ui> &topKidMapB, const std::vector<std::pair<int, int>> &id2Pair,
							   std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> &q,
							   const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian, 
							   std::ofstream &statStream)
{
	std::vector<std::pair<int, int>> tmp5, tmp2, tmp;
	for(uint64_t i = 0; i < K; i++) {
		int lid = topKidMapA[id2Pair[q.top().id].first];
		int rid = topKidMapB[id2Pair[q.top().id].second];
		if(i >= K * 4 / 5) {
			tmp5.emplace_back(lid, rid);
			tmp2.emplace_back(lid, rid);
			tmp.emplace_back(lid, rid);
		}
		else if(i >= K * 3 / 5) {
			tmp2.emplace_back(lid, rid);
			tmp.emplace_back(lid, rid);
		}
		else {
			tmp.emplace_back(lid, rid);
		}
		q.pop();
	}

	auto tmpGolds = golds;

	__gnu_parallel::sort(tmpGolds.begin(), tmpGolds.end());
	__gnu_parallel::sort(tmp5.begin(), tmp5.end());
	__gnu_parallel::sort(tmp2.begin(), tmp2.end());
	__gnu_parallel::sort(tmp.begin(), tmp.end());

	std::vector<std::pair<int, int>> internal5, internal2, internal;
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp5.begin(), tmp5.end(), 
									 std::back_inserter(internal5));
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp2.begin(), tmp2.end(), 
									 std::back_inserter(internal2));
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp.begin(), tmp.end(), 
									 std::back_inserter(internal));

	statStream << "*********************" << std::endl;
	statStream << "-------------- recall on representative attribute threshold accepting --------------" << std::endl;
	statStream << "     K: " << K / 5 << std::endl;
	statStream << std::setprecision(4) << "recall: " << internal5.size() * 1.0 / golds.size() << std::endl;
	statStream << "   |C|: " << tmp5.size() << std::endl;
	statStream << std::setprecision(4) << "  CSSR: " << tmp5.size() * 1.0 / cartesian << std::endl;
	statStream << std::endl;
	statStream << "     K: " << K * 2 / 5 << std::endl;
	statStream << std::setprecision(4) << "recall: " << internal2.size() * 1.0 / golds.size() << std::endl;
	statStream << "   |C|: " << tmp2.size() << std::endl;
	statStream << std::setprecision(4) << "  CSSR: " << tmp2.size() * 1.0 / cartesian << std::endl;
	statStream << std::endl;
	statStream << "     K: " << K << std::endl;
	statStream << std::setprecision(4) << "recall: " << internal.size() * 1.0 / golds.size() << std::endl;
	statStream << "   |C|: " << tmp.size() << std::endl;
	statStream << std::setprecision(4) << "  CSSR: " << tmp.size() * 1.0 / cartesian << std::endl;
	statStream << "-------------- recall on representative attribute threshold accepting --------------" << std::endl << std::flush;
	statStream << "*********************" << std::endl;
}


void TopK::topKviaTASelf(const Table &table_A, const std::string &topKattr, const std::string &attrType,
						 const std::vector<std::vector<std::vector<ui>>> &recordsA, 
						 const std::vector<std::vector<double>> &recWeights, 
						 const std::vector<std::vector<double>> &wordwt, 
						 const std::unordered_map<std::string, ui> &datasets_map, 
						 const std::vector<std::vector<ui>> &id_mapA, 
						 std::vector<std::vector<int>> &final_pairs,
						 const Table &groundTruth, uint64_t K, 
						 std::ofstream &statStream, bool ifWeighted, 
						 const std::string &mode)
{
	std::vector<std::vector<ui>> tempRecords;
	std::vector<ui> tempId_map;
	std::vector<double> tempRecWeights, tempWordwt;
	ui tempNumWord = 0;

	std::string topKey = "dlm_dc0_" + topKattr;
	ui pos = 0;
    auto diter = datasets_map.find(topKey);
	if(diter != datasets_map.end())
		pos = diter->second;
	else {
		std::cerr << "Error in topK key: " << topKey << std::endl;
		// exit(1);
		ui posAttr = table_A.inverted_schema.at(topKattr);
		pos = recordsA.size();
		Tokenizer::SelftableAttr2IntVector(table_A, tempRecords, tempRecWeights, 
										   tempWordwt, tempId_map, posAttr, 
										   TokenizerType::Dlm, tempNumWord, 0);
	}

	const auto &topKRecords = pos == recordsA.size() ? tempRecords : recordsA[pos];
	// const auto &topKString = table_A.cols[table_A.inverted_schema.at(topKattr)];
	const auto &topKidMap = pos == recordsA.size() ? tempId_map : id_mapA[pos];
	const auto &topKWeights = pos == recordsA.size() ? tempRecWeights : recWeights[pos];
	const auto &topKwordwt = pos == recordsA.size() ? tempWordwt : wordwt[pos];

	// prepare
	std::vector<std::pair<int, int>> id2Pair;
	TableEntry **valueTable;
	double **backup;
	ui numRow = (ui)table_A.row_no;
	uint64_t numEntity = 0;

	if(!ifWeighted)
		prepareSelf(topKRecords, topKidMap, final_pairs, id2Pair, valueTable, backup, numRow, numEntity);
	else 
		prepareSelfWeighted(topKRecords, topKidMap, topKWeights, topKwordwt, final_pairs, id2Pair, valueTable, backup, numRow, numEntity);

	if(numEntity <= K) {
		releaseBuffers(4, valueTable, backup);
		return;
	}

	// TA
	std::unordered_set<uint64_t> quickRef;
	std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> q;

	for(uint64_t i = 0; i < numEntity; i++) {
		double threshold = 0.0;
		for(ui j = 0; j < 4; j++) {
			const auto &e = valueTable[j][i];
			threshold += e.val;

			if(quickRef.find(e.id) == quickRef.end()) {
				quickRef.insert(e.id);

				uint64_t qid = e.id;
				double qval = 0.0;
				for(ui l = 0; l < 4; l++)
					qval += backup[l][qid];

				if((uint64_t)q.size() < K)
					q.emplace(qid, qval);
				else {
					q.pop();
					q.emplace(qid, qval);
				}
			}
		}

		if((uint64_t)q.size() >= K && q.top().val >= threshold)
			break;
	}

	if(mode == "replace") {
		updateFinalPairs(topKidMap, topKidMap, id2Pair, q, final_pairs, K);
		// return;
	}
	else {
		// ground truth
		std::vector<std::pair<int, int>> golds;
		for(int i = 0; i < groundTruth.row_no; i++) {
			int lid = std::stoi(groundTruth.rows[i][0]);
			int rid = std::stoi(groundTruth.rows[i][1]);
			golds.emplace_back(lid, rid);
		}
		
		if(mode == "report") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)(table_A.row_no - 1) / 2;
			getCurrentRecall(topKidMap, topKidMap, id2Pair, q, golds, K, cartesian);
			// return;
		}
		else if(mode == "hybrid") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)(table_A.row_no - 1) / 2;
			auto qCopy = q;
			getCurrentRecall(topKidMap, topKidMap, id2Pair, qCopy, golds, K, cartesian);
			updateFinalPairs(topKidMap, topKidMap, id2Pair, q, final_pairs, K);
			// return;
		}
		else if(mode == "exp") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)(table_A.row_no - 1) / 2;
			getCurrentRecallExp(topKidMap, topKidMap, id2Pair, q, golds, K, cartesian, statStream);
			// return;
		}
		else {
			std::cerr << "no such top k mode: " << mode << std::endl;
			exit(1);
		}
	}

	// release
	releaseBuffers(4, valueTable, backup);
}


void TopK::topKviaTARS(const Table &table_A, const Table &table_B, const std::string &topKattr, const std::string &attrType,
					   const std::vector<std::vector<std::vector<ui>>> &recordsA,
					   const std::vector<std::vector<std::vector<ui>>> &recordsB,
					   const std::vector<std::vector<double>> &recWeightsA, 
					   const std::vector<std::vector<double>> &recWeightsB, 
					   const std::vector<std::vector<double>> &wordwt,
					   const std::unordered_map<std::string, ui> &datasets_map, 
					   const std::vector<std::vector<ui>> &id_mapA,
					   const std::vector<std::vector<ui>> &id_mapB, 
					   std::vector<std::vector<int>> &final_pairs,
					   const Table &groundTruth, uint64_t K, 
					   std::ofstream &statStream, bool ifWeighted, 
					   const std::string &mode)
{
	std::vector<std::vector<ui>> tempRecordsA, tempRecordsB;
	std::vector<ui> tempId_mapA, tempId_mapB;
	std::vector<double> tempRecWeightsA, tempRecWeightsB, tempWordwt;
	ui tempNumWord = 0;

	std::string topKey = "dlm_dc0_" + topKattr;
	ui pos = 0;
    auto diter = datasets_map.find(topKey);
	if(diter != datasets_map.end())
		pos = diter->second;
	else {
		std::cerr << "Error in topK key: " << topKey << std::endl;
		// exit(1);
		ui posAttr = table_A.inverted_schema.at(topKattr);
		pos = recordsA.size();
		Tokenizer::RStableAttr2IntVector(table_A, table_B, tempRecordsA, tempRecordsB, 
										 tempRecWeightsA, tempRecWeightsB, tempWordwt, 
										 tempId_mapA, tempId_mapB, posAttr, posAttr, 
										 TokenizerType::Dlm, tempNumWord, 0);
	}

	const auto &topKRecordsA = pos == recordsA.size() ? tempRecordsA : recordsA[pos];
	const auto &topKRecordsB = pos == recordsA.size() ? tempRecordsB : recordsB[pos];
	const auto &topKidMapA = pos == recordsA.size() ? tempId_mapA : id_mapA[pos];
	const auto &topKidMapB = pos == recordsA.size() ? tempId_mapB : id_mapB[pos];
	const auto &topKWeightsA = pos == recordsA.size() ? tempRecWeightsA : recWeightsA[pos];
	const auto &topKWeightsB = pos == recordsA.size() ? tempRecWeightsB : recWeightsB[pos];
	// const auto &topKString = table_A.cols[table_A.inverted_schema.at(topKattr)];
	const auto &topKwordwt = pos == recordsA.size() ? tempWordwt : wordwt[pos];

	// prepare
	std::vector<std::pair<int, int>> id2Pair;
	TableEntry **valueTable;
	double **backup;
	ui numRowA = (ui)table_A.row_no;
	ui numRowB = (ui)table_B.row_no;
	uint64_t numEntity = 0;

	if(!ifWeighted)
		prepareRS(topKRecordsA, topKRecordsB, topKidMapA, topKidMapB, final_pairs, id2Pair, valueTable, backup, numRowA, numRowB, numEntity);
	else 
		prepareRSWeighted(topKRecordsA, topKRecordsB, topKidMapA, topKidMapB, topKWeightsA, topKWeightsB, topKwordwt, final_pairs, id2Pair, 
						  valueTable, backup, numRowA, numRowB, numEntity);

	if(numEntity <= K) {
		releaseBuffers(4, valueTable, backup);
		return;
	}
	std::cout << "done TA preparation" << std::endl << std::flush;

	// TA
	std::unordered_set<uint64_t> quickRef;
	std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> q;

	for(uint64_t i = 0; i < numEntity; i++) {
		double threshold = 0.0;
		for(ui j = 0; j < 4; j++) {
			const auto &e = valueTable[j][i];
			threshold += e.val;

			if(quickRef.find(e.id) == quickRef.end()) {
				quickRef.insert(e.id);

				uint64_t qid = e.id;
				double qval = 0.0;
				for(ui l = 0; l < 4; l++)
					qval += backup[l][qid];

				if((uint64_t)q.size() < K)
					q.emplace(qid, qval);
				else {
					q.pop();
					q.emplace(qid, qval);
				}
			}
		}

		if((uint64_t)q.size() >= K && q.top().val >= threshold)
			break;
	}

	if(mode == "replace") {
		updateFinalPairs(topKidMapA, topKidMapB, id2Pair, q, final_pairs, K);
		// return;
	}
	else {
		// ground truth
		std::vector<std::pair<int, int>> golds;
		for(int i = 0; i < groundTruth.row_no; i++) {
			int lid = std::stoi(groundTruth.rows[i][0]);
			int rid = std::stoi(groundTruth.rows[i][1]);
			golds.emplace_back(lid, rid);
		}
		
		if(mode == "report") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
			getCurrentRecall(topKidMapA, topKidMapB, id2Pair, q, golds, K, cartesian);
			// return;
		}
		else if(mode == "hybrid") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
			auto qCopy = q;
			getCurrentRecall(topKidMapA, topKidMapB, id2Pair, qCopy, golds, K, cartesian);
			updateFinalPairs(topKidMapA, topKidMapB, id2Pair, q, final_pairs, K);
			// return;
		}
		else if(mode == "exp") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
			getCurrentRecallExp(topKidMapA, topKidMapB, id2Pair, q, golds, K, cartesian, statStream);
			// return;
		}
		else {
			std::cerr << "no such top k mode: " << mode << std::endl;
			exit(1);
		}
	}

	// release
	releaseBuffers(4, valueTable, backup);
}


/*
 * TODO:
 * optimized version of TA
 * vectorization (SIMD + parallel)
 */
void TopK::topKviaTASelfOpt(const Table &table_A, const std::string &topKattr, const std::string &attrType,
							const std::vector<std::vector<std::vector<ui>>> &recordsA, 
							const std::unordered_map<std::string, ui> &datasets_map, 
							const std::vector<std::vector<ui>> &id_mapA, 
							std::vector<std::vector<int>> &final_pairs,
							uint64_t K)
{
	std::string topKey = "dlm_dc0_" + topKattr;
	ui pos = 0;
    auto diter = datasets_map.find(topKey);
	if(diter != datasets_map.end())
		pos = diter->second;
	else {
		std::cerr << "Error in topK key: " << topKey << std::endl;
		exit(1);
	}

	const auto &topKRecords = recordsA[pos];
	// const auto &topKString = table_A.cols[table_A.inverted_schema.at(topKattr)];
	const auto &topKidMap = id_mapA[pos];

	// prepare
	std::vector<std::pair<int, int>> id2Pair;
	TableEntry **valueTable;
	double **backup;
	ui numRow = (ui)table_A.row_no;
	uint64_t numEntity = 0;

	prepareSelf(topKRecords, topKidMap, final_pairs, id2Pair, valueTable, backup, numRow, numEntity);

	if(numEntity <= K) {
		releaseBuffers(4, valueTable, backup);
		return;
	}

	// TA
	std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> q[MAXTHREADNUM];

#pragma omp parallel
{
	std::unordered_set<uint64_t> quickRef;
	uint64_t tid = (uint64_t)omp_get_thread_num();
	uint64_t lb = tid * numEntity / (uint64_t)MAXTHREADNUM;
	uint64_t ub = (tid + 1) * numEntity / (uint64_t)MAXTHREADNUM;

	for(uint64_t i = lb; i < ub; i++) {
		double threshold = 0.0;
		for(ui j = 0; j < 4; j++) {
			const auto &e = valueTable[j][i];
			threshold += e.val;

			
		}
	}
}

	// save & flush

	// sort
	for(auto &fv : final_pairs)
		std::sort(fv.begin(), fv.end());

	// release
	releaseBuffers(4, valueTable, backup);
}


void TopK::updateFinalPairs(const std::vector<ui> &sortedIdMap, const std::vector<std::pair<int, int>> &allPairs,
							std::vector<std::vector<int>> &final_pairs, uint64_t K)
{
	uint64_t sizeK = K <= (uint64_t)sortedIdMap.size() ? K : (uint64_t)sortedIdMap.size();
	
	for(auto &fv : final_pairs)
		fv.clear();
	for(uint64_t i = 0; i < sizeK; i++) {
		int lid = allPairs[sortedIdMap[i]].first;
		int rid = allPairs[sortedIdMap[i]].second;
		final_pairs[lid].emplace_back(rid);
	}
	for(auto &fv : final_pairs)
		std::sort(fv.begin(), fv.end());
}


void TopK::getCurrentRecall(const std::vector<ui> &sortedIdMap, const std::vector<std::pair<int, int>> &allPairs,
							const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian)
{
	uint64_t sizeK = K <= (uint64_t)sortedIdMap.size() ? K : (uint64_t)sortedIdMap.size();
	
	std::vector<std::pair<int, int>> tmp;
	for(uint64_t i = 0; i < sizeK; i++) {
		int lid = allPairs[sortedIdMap[i]].first;
		int rid = allPairs[sortedIdMap[i]].second;
		tmp.emplace_back(lid, rid);
	}

	auto tmpGolds = golds;

	__gnu_parallel::sort(tmpGolds.begin(), tmpGolds.end());
	__gnu_parallel::sort(tmp.begin(), tmp.end());
	std::vector<std::pair<int, int>> internal;
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp.begin(), tmp.end(), 
									 std::back_inserter(internal));

	std::cout << "-------------- recall on all similarity scores --------------" << std::endl;
	std::cout << "     K: " << K << std::endl;
	std::cout << std::setprecision(4) << "recall: " << internal.size() * 1.0 / golds.size() << std::endl;
	std::cout << "   |C|: " << tmp.size() << std::endl;
	std::cout << std::setprecision(4) << "  CSSR: " << tmp.size() * 1.0 / cartesian << std::endl;
	std::cout << "-------------- recall on all similarity scores --------------" << std::endl << std::flush;
}


void TopK::getCurrentRecallExp(const std::vector<ui> &sortedIdMap, const std::vector<std::pair<int, int>> &allPairs,
							   const std::vector<std::pair<int, int>> &golds, uint64_t K, uint64_t cartesian, 
							   std::ofstream &statStream)
{
	uint64_t sizeK = K <= (uint64_t)sortedIdMap.size() ? K : (uint64_t)sortedIdMap.size();
	
	std::vector<std::pair<int, int>> tmp5, tmp2, tmp;
	for(uint64_t i = 0; i < sizeK; i++) {
		int lid = allPairs[sortedIdMap[i]].first;
		int rid = allPairs[sortedIdMap[i]].second;
		if(i < sizeK / 5) {
			tmp5.emplace_back(lid, rid);
			tmp2.emplace_back(lid, rid);
			tmp.emplace_back(lid, rid);
		}
		else if(i < sizeK * 2 / 5) {
			tmp2.emplace_back(lid, rid);
			tmp.emplace_back(lid, rid);
		}
		else {
			tmp.emplace_back(lid, rid);
		}
	}

	auto tmpGolds = golds;

	__gnu_parallel::sort(tmpGolds.begin(), tmpGolds.end());
	__gnu_parallel::sort(tmp5.begin(), tmp5.end());
	__gnu_parallel::sort(tmp2.begin(), tmp2.end());
	__gnu_parallel::sort(tmp.begin(), tmp.end());

	std::vector<std::pair<int, int>> internal5, internal2, internal;
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp5.begin(), tmp5.end(), 
									 std::back_inserter(internal5));
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp2.begin(), tmp2.end(), 
									 std::back_inserter(internal2));
	__gnu_parallel::set_intersection(tmpGolds.begin(), tmpGolds.end(), tmp.begin(), tmp.end(), 
									 std::back_inserter(internal));

	statStream << "*********************" << std::endl;
	statStream << "-------------- recall on all similarity scores --------------" << std::endl;
	statStream << "     K: " << K / 5 << std::endl;
	statStream << std::setprecision(4) << "recall: " << internal5.size() * 1.0 / golds.size() << std::endl;
	statStream << "   |C|: " << tmp5.size() << std::endl;
	statStream << std::setprecision(4) << "  CSSR: " << tmp5.size() * 1.0 / cartesian << std::endl;
	statStream << std::endl;
	statStream << "     K: " << K * 2 / 5 << std::endl;
	statStream << std::setprecision(4) << "recall: " << internal2.size() * 1.0 / golds.size() << std::endl;
	statStream << "   |C|: " << tmp2.size() << std::endl;
	statStream << std::setprecision(4) << "  CSSR: " << tmp2.size() * 1.0 / cartesian << std::endl;
	statStream << std::endl;
	statStream << "     K: " << K << std::endl;
	statStream << std::setprecision(4) << "recall: " << internal.size() * 1.0 / golds.size() << std::endl;
	statStream << "   |C|: " << tmp.size() << std::endl;
	statStream << std::setprecision(4) << "  CSSR: " << tmp.size() * 1.0 / cartesian << std::endl;
	statStream << "-------------- recall on all similarity scores --------------" << std::endl << std::flush;
	statStream << "*********************" << std::endl;
}


void TopK::topKviaAllSimilarityScoresRS(const Table &table_A, const Table &table_B, const Rule *rules,
										const std::vector<double> &simWeights,
										const std::unordered_map<std::string, double> &attrAverage,
										const std::vector<std::vector<std::vector<ui>>> &recordsA,
										const std::vector<std::vector<std::vector<ui>>> &recordsB,
										const std::vector<std::vector<double>> &recWeightsA, 
										const std::vector<std::vector<double>> &recWeightsB, 
										const std::vector<std::vector<double>> &wordwt,
										const std::unordered_map<std::string, ui> &datasets_map, 
										const std::vector<std::vector<ui>> &id_mapA,
										const std::vector<std::vector<ui>> &id_mapB, 
										const std::vector<std::vector<ui>> &id_mapAString,
										const std::vector<std::vector<ui>> &id_mapBString,
										std::vector<std::vector<int>> &final_pairs,
										const Table &groundTruth, uint64_t K, ui numRule, 
										std::ofstream &statStream, bool isWeighted, 
										const std::string &mode)
{
	std::vector<double> newWeights;
	double tot = 0.0;

	// normalize
	for(const auto &val : simWeights)
		tot += val;
	for(const auto &val : simWeights)
		newWeights.emplace_back(val / tot);

	// sigmoid
	// double m = tot / simWeights.size();
	// int k = 10;
	// for(const auto &val : simWeights) {
	// 	double power = -1.0 * k * (val - m);
	// 	newWeights.emplace_back(1.0 / (1.0 + exp(power)));
	// }

	std::vector<std::pair<std::string, double>> attrAverageVec;
	for(const auto &iter : attrAverage)
		attrAverageVec.emplace_back(iter.first, iter.second);
	std::sort(attrAverageVec.begin(), attrAverageVec.end(), [](const std::pair<std::string, double> &lhs, const std::pair<std::string, double> &rhs) {
		return lhs.second > rhs.second;
	});
	std::unordered_map<std::string, ui> attrRank;
	for(ui i = 0; i < attrAverageVec.size(); i++)
		attrRank[attrAverageVec[i].first] = i;

	std::vector<std::pair<int, int>> allPairs;
	for(int i = 0; i < table_A.row_no; i++)
		for(const auto &id : final_pairs[i])
			allPairs.emplace_back(i, id);
	size_t numRow = allPairs.size();
	std::vector<double> allValues(numRow, 0.0);
	std::vector<std::vector<double>> allValues2(attrAverageVec.size(), std::vector<double>(numRow, 0.0));

	for(ui i = 0; i < numRule; i++) {
		// std::cout << i << std::endl << std::flush;
		const std::string mapKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
		ui recPos = 0;
		if(datasets_map.find(mapKey) != datasets_map.end())
			recPos = datasets_map.at(mapKey);
		const auto &curRecordsA = recordsA[recPos];
		const auto &curRecordsB = recordsB[recPos];
		const auto &curWeightsA = recWeightsA[recPos];
		const auto &curWeightsB = recWeightsB[recPos];
		const auto &curWordwt = wordwt[recPos];

		ui lstrPos = table_A.inverted_schema.at(rules[i].attr);
		ui rstrPos = table_B.inverted_schema.at(rules[i].attr);
		const auto &curColumnA = table_A.cols[lstrPos];
		const auto &curColumnB = table_B.cols[rstrPos];

		std::vector<ui> curReverseIdMapA(id_mapA[recPos].size(), 0), curReverseIdMapB(id_mapB[recPos].size(), 0);
		for(ui j = 0; j < id_mapA[recPos].size(); j++)
			curReverseIdMapA[id_mapA[recPos][j]] = j;
		for(ui j = 0; j < id_mapB[recPos].size(); j++)
			curReverseIdMapB[id_mapB[recPos][j]] = j;

		std::vector<ui> curReverseIdMapAString(id_mapAString[lstrPos].size(), 0), curReverseIdMapBString(id_mapBString[rstrPos].size(), 0);
		for(ui j = 0; j < id_mapAString[lstrPos].size(); j++)
			curReverseIdMapAString[id_mapAString[lstrPos][j]] = j;
		for(ui j = 0; j < id_mapBString[rstrPos].size(); j++)
			curReverseIdMapBString[id_mapBString[rstrPos][j]] = j;

		ui attrPos = attrRank[rules[i].attr];
		// std::cout << "here: " << numRow << std::endl << std::flush;

		if(rules[i].sim == "lev") {
		#pragma omp parallel for
			for(size_t idx = 0; idx < numRow; idx++) {
				const auto &p = allPairs[idx];
				int lid = curReverseIdMapAString[p.first];
				int rid = curReverseIdMapBString[p.second];
				double val = 0.0;
				if(curColumnA[lid].empty() || curColumnB[rid].empty())
					val = 0.0;
				else
					val = SimFuncs::levSim(curColumnA[lid], curColumnB[rid]);
				allValues[idx] += (val * newWeights[i]);
				allValues2[attrPos][idx] += (val * newWeights[i]);
			}
		}
		else if(rules[i].sim == "exm") {
		#pragma omp parallel for
			for(size_t idx = 0; idx < numRow; idx++) {
				const auto &p = allPairs[idx];
				int lid = curReverseIdMapAString[p.first];
				int rid = curReverseIdMapBString[p.second];
				double val = (double)SimFuncs::exactMatch(curColumnA[lid], curColumnB[rid]);
				if(curColumnA[lid].empty() || curColumnB[rid].empty())
					val = 0.0;
				allValues[idx] += (val * newWeights[i]);
				allValues2[attrPos][idx] += (val * newWeights[i]);
			}
		}
		else if(rules[i].sim == "anm") {
		#pragma omp parallel for
			for(size_t idx = 0; idx < numRow; idx++) {
				const auto &p = allPairs[idx];
				int lid = curReverseIdMapAString[p.first];
				int rid = curReverseIdMapBString[p.second];
				double val = 0.0;
				if(curColumnA[lid].empty() || curColumnB[rid].empty())
					val = 0.0;
				else
					val = SimFuncs::absoluteNorm(curColumnA[lid], curColumnB[rid]);
				allValues[idx] += (val * newWeights[i]);
				allValues2[attrPos][idx] += (val * newWeights[i]);
			}
		}
		else if(rules[i].sim == "jac" || rules[i].sim == "cos" || rules[i].sim == "dice" || rules[i].sim == "overlap") {
			if(isWeighted) {
				double (*weightedSimJoin)(const std::vector<ui> &, const std::vector<ui> &, const std::vector<double> &, double, double) = nullptr;
				if(rules[i].sim == "jac") weightedSimJoin = SimFuncs::weightedJaccard;
				else if(rules[i].sim == "cos") weightedSimJoin = SimFuncs::weightedCosine;
				else if(rules[i].sim == "dice") weightedSimJoin = SimFuncs::weightedDice;
				else weightedSimJoin = SimFuncs::weightedOverlapCoeff;

			#pragma omp parallel for
				for(size_t idx = 0; idx < numRow; idx++) {
					const auto &p = allPairs[idx];
					int lid = curReverseIdMapA[p.first];
					int rid = curReverseIdMapB[p.second];
					double val = 0.0;
					if(curRecordsA[lid].empty() || curRecordsB[rid].empty())
						val = 0.0;
					else
						val = weightedSimJoin(curRecordsA[lid], curRecordsB[rid], curWordwt, curWeightsA[lid], curWeightsB[rid]);
					allValues[idx] += (val * newWeights[i]);
					allValues2[attrPos][idx] += (val * newWeights[i]);
				}
			}
			else {
				double (*simJoin)(const std::vector<ui> &, const std::vector<ui> &) = nullptr;
				if(rules[i].sim == "jac") simJoin = SimFuncs::jaccard;
				else if(rules[i].sim == "cos") simJoin = SimFuncs::cosine;
				else if(rules[i].sim == "dice") simJoin = SimFuncs::dice;
				else simJoin = SimFuncs::overlapCoeff;

			#pragma omp parallel for
				for(size_t idx = 0; idx < numRow; idx++) {
					const auto &p = allPairs[idx];
					int lid = curReverseIdMapA[p.first];
					int rid = curReverseIdMapB[p.second];
					double val = 0.0;
					if(curRecordsA[lid].empty() || curRecordsB[rid].empty())
						val = 0.0;
					else
						val = simJoin(curRecordsA[lid], curRecordsB[rid]);
					allValues[idx] += (val * newWeights[i]);
					allValues2[attrPos][idx] += (val * newWeights[i]);
				}
			}
		}
		else {
			std::cerr << "no such sim functions: " << rules[i].sim << std::endl;
			exit(1);
		}
	}
	// std::cout << "done adding" << std::endl << std::flush;

	std::vector<ui> sortedIdMap(allPairs.size());
	std::vector<ui> sortedIdMap2(allPairs.size());
	std::iota(sortedIdMap.begin(), sortedIdMap.end(), 0);
	std::iota(sortedIdMap2.begin(), sortedIdMap2.end(), 0);

	// method 1
	// according to weighted sum
	std::sort(sortedIdMap.begin(), sortedIdMap.end(), [&allValues](ui lhs, ui rhs) {
		return allValues[lhs] > allValues[rhs];
	});

	// method 2
	// define partial order in terms of attributes average weights, then use weighted sum of corresponding attributes
	std::sort(sortedIdMap2.begin(), sortedIdMap2.end(), [&allValues2](ui lhs, ui rhs) {
		for(ui i = 0; i < allValues2.size(); i++) {
			if(std::abs(allValues2[i][lhs] - allValues2[i][rhs]) < 1e-2)	
				continue;
			if(allValues2[i][lhs] > allValues2[i][rhs])
				return true;
			else if(allValues2[i][lhs] < allValues2[i][rhs])
				return false;
		}
		return lhs < rhs;
	});

	std::cout << "start report now" << std::endl << std::flush;
	if(mode == "replace") {
		updateFinalPairs(sortedIdMap, allPairs, final_pairs, K);
		return;
	}
	else {
		// ground truth
		std::vector<std::pair<int, int>> golds;
		for(int i = 0; i < groundTruth.row_no; i++) {
			int lid = std::stoi(groundTruth.rows[i][0]);
			int rid = std::stoi(groundTruth.rows[i][1]);
			golds.emplace_back(lid, rid);
		}
		
		if(mode == "report") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
			getCurrentRecall(sortedIdMap, allPairs, golds, K, cartesian);
			return;
		}
		else if(mode == "hybrid") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
			getCurrentRecall(sortedIdMap, allPairs, golds, K, cartesian);
			updateFinalPairs(sortedIdMap, allPairs, final_pairs, K);
			return;
		}
		else if(mode == "exp") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
			getCurrentRecallExp(sortedIdMap, allPairs, golds, K, cartesian, statStream);
			getCurrentRecallExp(sortedIdMap2, allPairs, golds, K, cartesian, statStream);
			updateFinalPairs(sortedIdMap2, allPairs, final_pairs, K);
			return;
		}
		else {
			std::cerr << "no such top k mode: " << mode << std::endl;
			exit(1);
		}
	}
}


void TopK::topKviaAllSimilarityScoreSelf(const Table &table_A, const Rule *rules, const std::vector<double> &simWeights, 
										 const std::unordered_map<std::string, double> &attrAverage,
										 const std::vector<std::vector<std::vector<ui>>> &records, 
										 const std::vector<std::vector<double>> &recWeights, 
										 const std::vector<std::vector<double>> &wordwt, 
										 const std::unordered_map<std::string, ui> &datasets_map, 
										 const std::vector<std::vector<ui>> &id_map, 
										 const std::vector<std::vector<ui>> &id_mapString, 
										 std::vector<std::vector<int>> &final_pairs, 
										 const Table &groundTruth, u_int64_t K, ui numRule, 
										 std::ofstream &statStream, bool isWeighted, 
										 const std::string &mode)
{
	std::vector<double> newWeights;
	double tot = 0.0;

	// normalize
	for(const auto &val : simWeights)
		tot += val;
	for(const auto &val : simWeights)
		newWeights.emplace_back(val / tot);

	// sigmoid
	// double m = tot / simWeights.size();
	// int k = 10;
	// for(const auto &val : simWeights) {
	// 	double power = -1.0 * k * (val - m);
	// 	newWeights.emplace_back(1.0 / (1.0 + exp(power)));
	// }

	std::vector<std::pair<std::string, double>> attrAverageVec;
	for(const auto &iter : attrAverage)
		attrAverageVec.emplace_back(iter.first, iter.second);
	std::sort(attrAverageVec.begin(), attrAverageVec.end(), [](const std::pair<std::string, double> &lhs, const std::pair<std::string, double> &rhs) {
		return lhs.second > rhs.second;
	});
	std::unordered_map<std::string, ui> attrRank;
	for(ui i = 0; i < attrAverageVec.size(); i++)
		attrRank[attrAverageVec[i].first] = i;

	std::vector<std::pair<int, int>> allPairs;
	for(int i = 0; i < table_A.row_no; i++)
		for(const auto &id : final_pairs[i])
			allPairs.emplace_back(i, id);
	size_t numRow = allPairs.size();
	std::vector<double> allValues(numRow, 0.0);
	std::vector<std::vector<double>> allValues2(attrAverageVec.size(), std::vector<double>(numRow, 0.0));

	for(ui i = 0; i < numRule; i++) {
		const std::string mapKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
		ui recPos = 0;
		if(datasets_map.find(mapKey) != datasets_map.end())
			recPos = datasets_map.at(mapKey);
		const auto &curRecords = records[recPos];
		const auto &curWeights = recWeights[recPos];
		const auto &curWordwt = wordwt[recPos];

		ui strPos = table_A.inverted_schema.at(rules[i].attr);
		const auto &curColumn = table_A.cols[strPos];

		std::vector<ui> curReverseIdMap(id_map[recPos].size(), 0);
		for(ui j = 0; j < id_map[recPos].size(); j++)
			curReverseIdMap[id_map[recPos][j]] = j;

		std::vector<ui> curReverseIdMapString(id_mapString[strPos].size(), 0);
		for(ui j = 0; j < id_mapString[strPos].size(); j++)
			curReverseIdMapString[id_mapString[strPos][j]] = j;

		ui attrPos = attrRank[rules[i].attr];

		if(rules[i].sim == "lev") {
		#pragma omp parallel for
			for(size_t idx = 0; idx < numRow; idx++) {
				const auto &p = allPairs[idx];
				int lid = curReverseIdMapString[p.first];
				int rid = curReverseIdMapString[p.second];
				double val = 0.0;
				if(curColumn[lid].empty() || curColumn[rid].empty())
					val = 0.0;
				else
					val = SimFuncs::levSim(curColumn[lid], curColumn[rid]);
				allValues[idx] += (val * newWeights[i]);
				allValues2[attrPos][idx] += (val * newWeights[i]);
			}
		}
		else if(rules[i].sim == "exm") {
		#pragma omp parallel for
			for(size_t idx = 0; idx < numRow; idx++) {
				const auto &p = allPairs[idx];
				int lid = curReverseIdMapString[p.first];
				int rid = curReverseIdMapString[p.second];
				double val = (double)SimFuncs::exactMatch(curColumn[lid], curColumn[rid]);
				if(curColumn[lid].empty() || curColumn[rid].empty())
					val = 0.0;
				allValues[idx] += (val * newWeights[i]);
				allValues2[attrPos][idx] += (val * newWeights[i]);
			}
		}
		else if(rules[i].sim == "anm") {
		#pragma omp parallel for
			for(size_t idx = 0; idx < numRow; idx++) {
				const auto &p = allPairs[idx];
				int lid = curReverseIdMapString[p.first];
				int rid = curReverseIdMapString[p.second];
				double val = 0.0;
				if(curColumn[lid].empty() || curColumn[rid].empty())
					val = 0.0;
				else
					val = SimFuncs::absoluteNorm(curColumn[lid], curColumn[rid]);
				allValues[idx] += (val * newWeights[i]);
				allValues2[attrPos][idx] += (val * newWeights[i]);
			}
		}
		else if(rules[i].sim == "jac" || rules[i].sim == "cos" || rules[i].sim == "dice" || rules[i].sim == "overlap") {
			if(isWeighted) {
				double (*weightedSimJoin)(const std::vector<ui> &, const std::vector<ui> &, const std::vector<double> &, double, double) = nullptr;
				if(rules[i].sim == "jac") weightedSimJoin = SimFuncs::weightedJaccard;
				else if(rules[i].sim == "cos") weightedSimJoin = SimFuncs::weightedCosine;
				else if(rules[i].sim == "dice") weightedSimJoin = SimFuncs::weightedDice;
				else weightedSimJoin = SimFuncs::weightedOverlapCoeff;

			#pragma omp parallel for
				for(size_t idx = 0; idx < numRow; idx++) {
					const auto &p = allPairs[idx];
					int lid = curReverseIdMap[p.first];
					int rid = curReverseIdMap[p.second];
					double val = 0.0;
					if(curRecords[lid].empty() || curRecords[rid].empty())
						val = 0.0;
					else
						val = weightedSimJoin(curRecords[lid], curRecords[rid], curWordwt, curWeights[lid], curWeights[rid]);
					allValues[idx] += (val * newWeights[i]);
					allValues2[attrPos][idx] += (val * newWeights[i]);
				}
			}
			else {
				double (*simJoin)(const std::vector<ui> &, const std::vector<ui> &) = nullptr;
				if(rules[i].sim == "jac") simJoin = SimFuncs::jaccard;
				else if(rules[i].sim == "cos") simJoin = SimFuncs::cosine;
				else if(rules[i].sim == "dice") simJoin = SimFuncs::dice;
				else simJoin = SimFuncs::overlapCoeff;

			#pragma omp parallel for
				for(size_t idx = 0; idx < numRow; idx++) {
					const auto &p = allPairs[idx];
					int lid = curReverseIdMap[p.first];
					int rid = curReverseIdMap[p.second];
					double val = 0.0;
					if(curRecords[lid].empty() || curRecords[rid].empty())
						val = 0.0;
					else
						val = simJoin(curRecords[lid], curRecords[rid]);
					allValues[idx] += (val * newWeights[i]);
					allValues2[attrPos][idx] += (val * newWeights[i]);
				}
			}
		}
		else {
			std::cerr << "no such sim functions: " << rules[i].sim << std::endl;
			exit(1);
		}
	}

	std::vector<ui> sortedIdMap(allPairs.size());
	std::vector<ui> sortedIdMap2(allPairs.size());
	std::iota(sortedIdMap.begin(), sortedIdMap.end(), 0);
	std::iota(sortedIdMap2.begin(), sortedIdMap2.end(), 0);

	// method 1
	// according to weighted sum
	std::sort(sortedIdMap.begin(), sortedIdMap.end(), [&allValues](ui lhs, ui rhs) {
		return allValues[lhs] > allValues[rhs];
	});

	// method 2
	// define partial order in terms of attributes average weights, then use weighted sum of corresponding attributes
	std::sort(sortedIdMap2.begin(), sortedIdMap2.end(), [&allValues2](ui lhs, ui rhs) {
		for(ui i = 0; i < allValues2.size(); i++) {
			if(std::abs(allValues2[i][lhs] - allValues2[i][rhs]) < 1e-2)	
				continue;
			if(allValues2[i][lhs] > allValues2[i][rhs])
				return true;
			else if(allValues2[i][lhs] < allValues2[i][rhs])
				return false;
		}
		return lhs < rhs;
	});

	if(mode == "replace") {
		updateFinalPairs(sortedIdMap, allPairs, final_pairs, K);
		return;
	}
	else {
		// ground truth
		std::vector<std::pair<int, int>> golds;
		for(int i = 0; i < groundTruth.row_no; i++) {
			int lid = std::stoi(groundTruth.rows[i][0]);
			int rid = std::stoi(groundTruth.rows[i][1]);
			golds.emplace_back(lid, rid);
		}
		
		if(mode == "report") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)(table_A.row_no - 1) / 2;
			getCurrentRecall(sortedIdMap, allPairs, golds, K, cartesian);
			return;
		}
		else if(mode == "hybrid") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)(table_A.row_no - 1) / 2;
			getCurrentRecall(sortedIdMap, allPairs, golds, K, cartesian);
			updateFinalPairs(sortedIdMap, allPairs, final_pairs, K);
			return;
		}
		else if(mode == "exp") {
			uint64_t cartesian = (uint64_t)table_A.row_no * (uint64_t)(table_A.row_no - 1) / 2;
			getCurrentRecallExp(sortedIdMap, allPairs, golds, K, cartesian, statStream);
			getCurrentRecallExp(sortedIdMap2, allPairs, golds, K, cartesian, statStream);
			updateFinalPairs(sortedIdMap2, allPairs, final_pairs, K);
			return;
		}
		else {
			std::cerr << "no such top k mode: " << mode << std::endl;
			exit(1);
		}
	}
}


void TopK::topKviaTATable(const Table &matchRes, uint64_t K, std::vector<std::vector<std::string>> &newTable)
{
	// simple preparing
	TableEntry **valueTable{nullptr};
	double **backup{nullptr};
	// as the match result will have a small size
	// there will not be overflow
	uint64_t numEntity = (uint64_t)matchRes.row_no;
	ui numDimension = matchRes.schema.size() - 3;

	if(numEntity < K) {
		for(uint64_t i = 0; i < numEntity; i++)
			newTable.emplace_back(matchRes.rows[i]);
		return;
	}

	valueTable = new TableEntry*[numDimension];
	for(ui i = 0; i < numDimension; i++)
		valueTable[i] = new TableEntry[numEntity];
	backup = new double*[numDimension];
	for(ui i = 0; i < numDimension; i++)
		backup[i] = new double[numEntity];

	// fill in value table
#pragma omp parallel for
	for(uint64_t i = 0; i < numEntity; i++) {
		for(ui j = 0; j < numDimension; j++) {
			auto &e = valueTable[j][i];
			e.id = i;
			e.val = std::stod(matchRes.rows[i][j+3]);
			backup[j][i] = e.val;
		}
	}
	// sort
	for(ui i = 0; i < numDimension; i++) 
		__gnu_parallel::sort(valueTable[i], valueTable[i] + numEntity, 
		[](const TableEntry &e1, const TableEntry &e2) {
			return e1.val > e2.val;
		});

	// TA
	std::unordered_set<uint64_t> quickRef;
	std::priority_queue<TableEntry, std::vector<TableEntry>, PQComparison> q;

	for(uint64_t i = 0; i < numEntity; i++) {
		double threshold = 0.0;
		for(ui j = 0; j < numDimension; j++) {
			const auto &e = valueTable[j][i];
			threshold += e.val;

			if(quickRef.find(e.id) == quickRef.end()) {
				quickRef.insert(e.id);

				uint64_t qid = e.id;
				double qval = 0.0;
				for(ui l = 0; l < numDimension; l++)
					qval += backup[l][qid];

				if((uint64_t)q.size() < K)
					q.emplace(qid, qval);
				else {
					q.pop();
					q.emplace(qid, qval);
				}
			}
		}

		if((uint64_t)q.size() >= K && q.top().val >= threshold)
			break;
	}

	// save & flush
	for(uint64_t i = 0; i < K; i++) {
		newTable.emplace_back(matchRes.rows[q.top().id]);
		q.pop();
	}

	// release
	for(ui i = 0; i < numDimension; i++)
		delete[] valueTable[i];
	delete[] valueTable;
	for(ui i = 0; i < numDimension; i++)
		delete[] backup[i];
	delete[] backup;
}