/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "blocker/simjoin_blocker.h"


void SimJoinBlocker::selfSimilarityJoinParallel(uint64_t K, const std::string &topKattr, 
										        const std::string &attrType, bool ifWeighted)
{
	for(ui i = 0; i < num_rules; i++) {
		double newT = rules[i].threshold;

		printf("---------------------------\n");
		printf("Start similarity join, Attributes: %s  Functions: %s\nTokenizers: %s  Settings: %s  Threshold: %.2lf\n", 
				rules[i].attr.c_str(), rules[i].sim.c_str(), rules[i].tok.c_str(), rules[i].tok_settings.c_str(), 
				newT);
		printf("---------------------------\n");
		fflush(stdout);

		if(K > 0)
			BlockerUtil::pretopKviaTASelf(K, topKattr, attrType, ifWeighted);

		// set-join
		if(rules[i].sim == "jac" || rules[i].sim == "cos" || rules[i].sim == "dice") {
			std::vector<std::pair<int, int>> sim_pairs;
			std::string tokKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			ui pos = datasets_map[tokKey];

			// join
            SetJoinParallel *joiner = new SetJoinParallel(recordsA[pos], weightsA[pos], wordwt[pos], newT, 0, ifWeighted);
            if(rules[i].sim == "jac") joiner->simFType = SimFuncType::JACCARD;
            else if(rules[i].sim == "cos") joiner->simFType = SimFuncType::COSINE;
            else if(rules[i].sim == "dice") joiner->simFType = SimFuncType::DICE;

            joiner->index(newT);
            joiner->findSimPairsSelf();
            joiner->mergeResults(sim_pairs);

			std::cout << "### size: " << sim_pairs.size() << std::endl;
			fflush(stdout);

			// synethis
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 0);

			sim_pairs.clear();
			delete joiner;
		}
		// ovlp-join
		else if(rules[i].sim == "overlap") {
			std::vector<std::pair<int, int>> sim_pairs;
			std::string tokKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			ui pos = datasets_map[tokKey];

			// join & synethis
            int othres = ceil(rules[i].threshold - 1e-5);
            OvlpSelfJoinParallel *joiner = new OvlpSelfJoinParallel(recordsA[pos], weightsA[pos], wordwt[pos], 0, ifWeighted);
            joiner->overlapjoin(othres, sim_pairs);

			std::cout << "### size: " << sim_pairs.size() << std::endl;
			fflush(stdout);

            // synethsis
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 0);

			sim_pairs.clear();
			delete joiner;
		}
		// string-join
		else if(rules[i].sim == "lev") {
			std::vector<std::pair<int, int>> sim_pairs;
			ui pos = table_A.inverted_schema[rules[i].attr];
			
			// join & synethis
            int sthres = floor(rules[i].threshold + 1e-5);

            if(sthres > 0) {
                StringJoinParallel *joiner = new StringJoinParallel(table_A.cols[pos], sthres);
                joiner->selfJoin(sim_pairs);
				std::cout << "### size: " << sim_pairs.size() << std::endl;
                BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 1);
                delete joiner;
				sim_pairs.clear();
            }
            else {
				bool found = false;
				for(ui r = 0; r < num_rules; r++) {
					if(rules[r].sim == "exm" && rules[r].attr == rules[i].attr) {
						found = true;
						break;
					}
				}

				if(found)
					continue;

                // join
                ExactJoinParallel::exactJoinSelf(table_A.cols[pos], sim_pairs);
				std::cout << "### size: " << sim_pairs.size() << std::endl;
                // synethis
				BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 1);
                sim_pairs.clear();
            }
		}
		// others
		else if(rules[i].sim == "exm") {
			std::vector<std::pair<int, int>> sim_pairs;
			ui pos = table_A.inverted_schema[rules[i].attr];
		
			// join
			ExactJoinParallel::exactJoinSelf(table_A.cols[pos], sim_pairs);
			std::cout << "### size: " << sim_pairs.size() << std::endl;
			// synethis
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 1);
			sim_pairs.clear();
		}
		else if(rules[i].sim == "anm") {
			std::vector<std::pair<int, int>> sim_pairs;
			std::vector<std::pair<int, int>> result_pairs[MAXTHREADNUM];
			ui pos = table_A.inverted_schema[rules[i].attr];
			ui sizeA = table_A.cols[pos].size();

			// join
			double threshold = rules[i].threshold;

			// a large threshold we just use exm
			if(threshold > 0.90) {
				bool found = false;
				for(ui r = 0; r < num_rules; r++) {
					if(rules[r].sim == "exm" && rules[r].attr == rules[i].attr) {
						found = true;
						break;
					}
				}

				if(found)
					continue;
				
				ExactJoinParallel::exactJoinSelf(table_A.cols[pos], sim_pairs);
			}
			else {			
				int eralyTerminated[MAXTHREADNUM] = { 0 };
				// ui maxHeapSize = K == 0 ? MAX_PAIR_SIZE : (ui)K;
#pragma omp parallel for
				for(ui ii = 0; ii < sizeA; ii++) {
					int tid = omp_get_thread_num();
					if(eralyTerminated[tid] == 1)
						continue;

					for(ui jj = ii + 1; jj < sizeA; jj++) 
						if(SimFuncs::absoluteNorm(table_A.cols[pos][ii], table_A.cols[pos][jj]) >= threshold)
							result_pairs[tid].emplace_back(ii, jj);

					if(result_pairs[tid].size() >= MAX_PAIR_SIZE)
						eralyTerminated[tid] = 1;
				}
				// merge
				for(int tid = 0; tid < MAXTHREADNUM; tid++)
					sim_pairs.insert(sim_pairs.end(), result_pairs[tid].begin(), result_pairs[tid].end());
			}

			// synethis
			std::cout << "### size: " << sim_pairs.size() << std::endl;
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 1);
			sim_pairs.clear();
		}
		else {
			std::cerr << "No such sim funcs: " << rules[i].sim << std::endl;
			exit(1);
		}

		// inspect memory
		double vm, rss; // virtual memory & resident set size
		SetJoinUtil::processMemUsage(vm, rss);
		printf("\033[1;34mVirtual memory: %.1lf\nResident set size: %.1lf\033[0m\n", vm, rss);
	}
}


void SimJoinBlocker::RSSimilarityJoinSerial(uint64_t K, const std::string &topKattr, 
									        const std::string &attrType, bool ifWeighted, 
									        bool isJoinTopK)
{
    std::string sim_pair_file = "./buffer/sim_pairs.txt";
	uint64_t Kbackup = K;
	K = isJoinTopK ? K * 2 : 0;

#if USING_PARALLEL == 1
#pragma omp parallel for 
#endif
	for(ui i = 0; i < num_rules; i++) {
		double newT = rules[i].threshold;
	
		printf("---------------------------\n");
		printf("Start similarity join, Attributes: %s  Functions: %s\nTokenizers: %s  Settings: %s  Threshold: %.2lf\n", 
				rules[i].attr.c_str(), rules[i].sim.c_str(), rules[i].tok.c_str(), rules[i].tok_settings.c_str(), 
				newT);
		printf("---------------------------\n");
		fflush(stdout);

		if(K > 0)
			BlockerUtil::pretopKviaTARS(Kbackup, topKattr, attrType, ifWeighted);

		// set-join
		if(rules[i].sim == "jac" || rules[i].sim == "cos" || rules[i].sim == "dice") {
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
#pragma omp critical
{
#endif
#endif
			std::vector<std::pair<int, int>> sim_pairs;
			std::string tokKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			ui pos = datasets_map[tokKey];

			// join
			SetJoin *joiner = new SetJoin(recordsB[pos], recordsA[pos], weightsB[pos], weightsA[pos], wordwt[pos], sim_pair_file, newT, (ui)K, true);
			if(rules[i].sim == "jac") joiner->simFType = SimFuncType::JACCARD;
			else if(rules[i].sim == "cos") joiner->simFType = SimFuncType::COSINE;
			else if(rules[i].sim == "dice") joiner->simFType = SimFuncType::DICE;

			joiner->setRSJoin(newT, sim_pairs);

			delete joiner;

			// synethis
			BlockerUtil::synthesizePairsRS(pos, sim_pairs, 0);
			std::cout << "### size: " << sim_pairs.size() << std::endl;
			sim_pairs.clear();
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
}
#endif
#endif
		}
		// ovlp-join
		else if(rules[i].sim == "overlap") {
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
#pragma omp critical
{
#endif
#endif
			std::vector<std::pair<int, int>> sim_pairs;
			std::string tokKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			ui pos = datasets_map[tokKey];

			// join & synethis
			int othres = ceil(rules[i].threshold - 1e-5);
			OvlpRSJoin *ojoiner = new OvlpRSJoin(recordsA[pos], recordsB[pos], weightsA[pos], weightsB[pos], wordwt[pos], (ui)K, true);

			ojoiner->overlapjoin(othres, sim_pairs);
			BlockerUtil::synthesizePairsRS(pos, sim_pairs, 0);
			std::cout << "### size: " << sim_pairs.size() << std::endl;
			sim_pairs.clear();
			delete ojoiner;
			
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
}
#endif
#endif
		}
		// string-join
		else if(rules[i].sim == "lev") {
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
#pragma omp critical
{
#endif
#endif
			std::vector<std::pair<int, int>> sim_pairs;
			ui pos = table_A.inverted_schema[rules[i].attr];
			
			// join & synethis
			int sthres = floor(rules[i].threshold + 1e-5);

			if(sthres > 0) {
				StringJoin *sjoiner = new StringJoin(table_B.cols[pos], table_A.cols[pos], sthres, (ui)K);
				sjoiner->RSJoin(sim_pairs);
				BlockerUtil::synthesizePairsRS(pos, sim_pairs, 1);
				delete sjoiner;
				std::cout << "### size: " << sim_pairs.size() << std::endl;
				sim_pairs.clear();
			}
			else {
				// join
				ExactJoin::exactJoinRS(table_A.cols[pos], table_B.cols[pos], sim_pairs);
				// synethis
				BlockerUtil::synthesizePairsRS(pos, sim_pairs, 1);
				std::cout << "### size: " << sim_pairs.size() << std::endl;
				sim_pairs.clear();
			}
			
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
}
#endif
#endif
		}
		// others
		else if(rules[i].sim == "exm") {
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
#pragma omp critical
{
#endif
#endif
			std::vector<std::pair<int, int>> sim_pairs;
			ui pos = table_A.inverted_schema[rules[i].attr];
		
			// join
			ExactJoin::exactJoinRS(table_A.cols[pos], table_B.cols[pos], sim_pairs, (ui)K);
			// synethis
			BlockerUtil::synthesizePairsRS(pos, sim_pairs, 1);
			std::cout << "### size: " << sim_pairs.size() << std::endl;
			sim_pairs.clear();
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
}
#endif
#endif
		}
		else if(rules[i].sim == "anm") {
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
#pragma omp critical
{
#endif
#endif
			std::vector<std::pair<int, int>> sim_pairs;
			std::vector<WeightPair> result_pairs_;
			ui pos = table_A.inverted_schema[rules[i].attr];
			ui sizeA = table_A.cols[pos].size();
			ui sizeB = table_B.cols[pos].size();

			// join
			double threshold = rules[i].threshold;
			for(ui ii = 0; ii < sizeA; ii++) {
				for(ui jj = 0; jj < sizeB; jj++) {
					double val = SimFuncs::absoluteNorm(table_A.cols[pos][ii], table_B.cols[pos][jj]);
					if(val >= threshold)
						result_pairs_.emplace_back(ii, jj, val);
				}
			}

			std::sort(result_pairs_.begin(), result_pairs_.end(), [](const WeightPair &lhs, const WeightPair &rhs){
				return lhs.val > rhs.val;
			});

			ui maxHeapSize = K == 0 ? MAX_PAIR_SIZE_SERIAL : K;
			if(result_pairs_.size() >= maxHeapSize)
				result_pairs_.resize(maxHeapSize);

			for(const auto &p : result_pairs_)
				sim_pairs.emplace_back(p.id1, p.id2);
			
			// synethis
			BlockerUtil::synthesizePairsRS(pos, sim_pairs, 1);
			std::cout << "### size: " << sim_pairs.size() << std::endl;
			sim_pairs.clear();
#if USING_PARALLEL == 1
#if USING_CRITICAL == 1
}
#endif
#endif
		}
		else {
			std::cerr << "No such sim funcs: " << rules[i].sim << std::endl;
			exit(1);
		}

		// inspect memory
		double vm, rss; // virtual memory & resident set size
		SetJoinUtil::processMemUsage(vm, rss);
		printf("\033[1;34mVirtual memory: %.1lf\nResident set size: %.1lf\033[0m\n", vm, rss);
	}
}


// we give penalty to missing values
// missing values indicate this attribute contains fewer infos than others
void SimJoinBlocker::estimateDensity(bool isWeighted, std::vector<double> &densities, 
							         std::unordered_map<std::string, double> &attrAverage, 
									 const std::string &defaultSampleResDir)
{
	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultSampleResDir == "" ? directory + "../../output/buffer/"
										  : (defaultSampleResDir.back() == '/' ? defaultSampleResDir : defaultSampleResDir + "/");
	const std::string pathSampleRes = directory + "sample_res.csv";
	CSVReader reader;
	reader.reading_one_table(pathSampleRes, false);
	Table sampleRes = reader.tables[0];
	sampleRes.Profile();

	ui totalPositive = 0;
	ui posLabel = sampleRes.inverted_schema.at("label");
	for(const auto &row : sampleRes.rows) 
		if(std::stoi(row[posLabel]) == 1)
			++ totalPositive;

	ui numRow = sampleRes.rows.size();
	std::unordered_map<std::string, ui> mapRecords;
	std::vector<std::vector<std::vector<ui>>> estimateRecordsA, estimateRecordsB;
	std::vector<std::vector<double>> estimateWeightsA, estimateWeightsB, estimateWordwt;
	std::unordered_map<std::string, ui> attrNumRules;

	for(ui i = 0; i < num_rules; i++) {
		double newT = rules[i].threshold;
		bool isTokNone = rules[i].tok == "none";
		const std::string mapKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;

		// check point
		if(isTokNone && (rules[i].sim != "lev" && rules[i].sim != "exm" && rules[i].sim != "anm")) {
			std::cerr << "error in rule " << i << " : " << rules[i].sim << " " << rules[i].tok << std::endl;
			exit(1);
		}

		if(mapRecords.find(mapKey) == mapRecords.end() && !isTokNone) {
			ui newPos = estimateRecordsA.size();
			mapRecords.insert({mapKey, newPos});

			const std::string tmpLschema = "ltable_" + rules[i].attr;
			const std::string tmpRschema = "rtable_" + rules[i].attr;
			ui columnA = sampleRes.inverted_schema.at(tmpLschema);
			ui columnB = sampleRes.inverted_schema.at(tmpRschema);
			TokenizerType tokType = TokenizerType::Dlm;
			ui numWord = 0, q = 0;

			if(rules[i].tok == "dlm") {
				// do nothing
			}
			else if(rules[i].tok == "qgm") {
				tokType = TokenizerType::QGram;
				q = std::stoi(rules[i].tok_settings);
			}
			else if(rules[i].tok == "wspace")
				tokType = TokenizerType::WSpace;
			else if(rules[i].tok == "alphanumeric")
				tokType = TokenizerType::AlphaNumeric;
			else {
				std::cerr << "no such tokenizer: " << rules[i].tok << std::endl;
				exit(1);
			}

			estimateRecordsA.emplace_back();
			estimateRecordsB.emplace_back();
			estimateWeightsA.emplace_back();
			estimateWeightsB.emplace_back();
			estimateWordwt.emplace_back();
			Tokenizer::resTableAttr2IntVector(sampleRes, estimateRecordsA[newPos], estimateRecordsB[newPos], 
											  estimateWeightsA[newPos], estimateWeightsB[newPos], estimateWordwt[newPos], 
											  columnA, columnB, tokType, numWord, q);
		}

		ui recPos = isTokNone ? 0 : mapRecords.at(mapKey);
		const auto &curRecordsA = isTokNone ? std::vector<std::vector<ui>> () : estimateRecordsA[recPos];
		const auto &curRecordsB = isTokNone ? std::vector<std::vector<ui>> () : estimateRecordsB[recPos];
		const auto &curWeightsA = isTokNone ? std::vector<double> () : estimateWeightsA[recPos];
		const auto &curWeightsB = isTokNone ? std::vector<double> () : estimateWeightsB[recPos];
		const auto &curWordwt = isTokNone ? std::vector<double> () : estimateWordwt[recPos];
		const auto &curColumnA = sampleRes.cols[sampleRes.inverted_schema.at("ltable_" + rules[i].attr)];
		const auto &curColumnB = sampleRes.cols[sampleRes.inverted_schema.at("rtable_" + rules[i].attr)];

		if(attrAverage.find(rules[i].attr) == attrAverage.end()) {
			attrAverage[rules[i].attr] = 0.0;
			attrNumRules[rules[i].attr] = 1;
		}
		else
			++ attrNumRules[rules[i].attr];

		if(rules[i].sim == "lev") {
			int sthres = floor(rules[i].threshold + 1e-5);
			ui predict = 0, hit = 0, missing = 0;
			for(ui j = 0; j < numRow; j++) {
				if(curColumnA[j].empty() || curColumnB[j].empty()) {
					++ missing;
					continue;
				}
				if(SimFuncs::levDist(curColumnA[j], curColumnB[j]) <= sthres) {
					++ predict;
					if(std::stoi(sampleRes.rows[j][posLabel]) == 1)
						++ hit;
				}
			}
			double density = predict * 1.0 / numRow;
			double recall = hit * 1.0 / totalPositive;
			// double precision = hit * 1.0 / predict;
			double precision = std::abs(predict + missing - 0.0) <= 1e-4 ? 0.0 : hit * 1.0 / (predict + missing);
			double F1 = std::abs(precision + recall - 0.0) <= 1e-4 ? 0.0 : 2 * (precision * recall) / (precision + recall);
			double weights = F1;
			// double weights = 1 - density;
			densities.emplace_back(weights);
			std::cout << rules[i].attr << " " << rules[i].sim << " " << rules[i].tok << " " << rules[i].threshold << " "
					  << std::setprecision(4) << weights << " " << missing << " " << predict << " " << hit << std::endl;
			attrAverage[rules[i].attr] += weights;
		}
		else if(rules[i].sim == "exm") {
			ui predict = 0, hit = 0, missing = 0;
			for(ui j = 0; j < numRow; j++) {
				if(curColumnA[j].empty() || curColumnB[j].empty()) {
					++ missing;
					continue;
				}
				if((double)SimFuncs::exactMatch(curColumnA[j], curColumnB[j]) >= newT) {
					++ predict;
					if(std::stoi(sampleRes.rows[j][posLabel]) == 1)
						++ hit;
				}
			}
			double density = predict * 1.0 / numRow;
			double recall = hit * 1.0 / totalPositive;
			// double precision = hit * 1.0 / predict;
			double precision = std::abs(predict + missing - 0.0) <= 1e-4 ? 0.0 : hit * 1.0 / (predict + missing);
			double F1 = std::abs(precision + recall - 0.0) <= 1e-4 ? 0.0 : 2 * (precision * recall) / (precision + recall);
			double weights = F1;
			// double weights = 1 - density;
			densities.emplace_back(weights);
			std::cout << rules[i].attr << " " << rules[i].sim << " " << rules[i].tok << " " << rules[i].threshold << " "
					  << std::setprecision(4) << weights << " " << missing << " " << predict << " " << hit << std::endl;
			attrAverage[rules[i].attr] += weights;
		}
		else if(rules[i].sim == "anm") {
			ui predict = 0, hit = 0, missing = 0;
			for(ui j = 0; j < numRow; j++) {
				if(curColumnA[j].empty() || curColumnB[j].empty()) {
					++ missing;
					continue;
				}
				if(SimFuncs::absoluteNorm(curColumnA[j], curColumnB[j]) >= newT) {
					++ predict;
					if(std::stoi(sampleRes.rows[j][posLabel]) == 1)
						++ hit;
				}
			}
			double density = predict * 1.0 / numRow;
			double recall = hit * 1.0 / totalPositive;
			// double precision = hit * 1.0 / predict;
			double precision = std::abs(predict + missing - 0.0) <= 1e-4 ? 0.0 : hit * 1.0 / (predict + missing);
			double F1 = std::abs(precision + recall - 0.0) <= 1e-4 ? 0.0 : 2 * (precision * recall) / (precision + recall);
			double weights = F1;
			// double weights = 1 - density;
			densities.emplace_back(weights);
			std::cout << rules[i].attr << " " << rules[i].sim << " " << rules[i].tok << " " << rules[i].threshold << " "
					  << std::setprecision(4) << weights << " " << missing << " " << predict << " " << hit << std::endl;
			attrAverage[rules[i].attr] += weights;
		}
		else if(rules[i].sim == "overlap") {
			int othres = ceil(rules[i].threshold - 1e-5);
			ui predict = 0, hit = 0, missing = 0;
			for(ui j = 0; j < numRow; j++) {
				if(curRecordsA[j].empty() || curRecordsB[j].empty()) {
					++ missing;
					continue;
				}
				if(SimFuncs::overlap(curRecordsA[j], curRecordsB[j]) >= othres) {
					++ predict;
					if(std::stoi(sampleRes.rows[j][posLabel]) == 1)
						++ hit;
				}
			}
			double density = predict * 1.0 / numRow;
			double recall = hit * 1.0 / totalPositive;
			// double precision = hit * 1.0 / predict;
			double precision = std::abs(predict + missing - 0.0) <= 1e-4 ? 0.0 : hit * 1.0 / (predict + missing);
			double F1 = std::abs(precision + recall - 0.0) <= 1e-4 ? 0.0 : 2 * (precision * recall) / (precision + recall);
			double weights = F1;
			// double weights = 1 - density;
			densities.emplace_back(weights);
			std::cout << rules[i].attr << " " << rules[i].sim << " " << rules[i].tok << " " << rules[i].threshold << " "
					  << std::setprecision(4) << weights << std::endl;
			attrAverage[rules[i].attr] += weights;
		}
		else if(rules[i].sim == "jac" || rules[i].sim == "cos" || rules[i].sim == "dice") {
			if(isWeighted) {
				double (*weightedSimJoin)(const std::vector<ui> &, const std::vector<ui> &, const std::vector<double> &, double, double) = nullptr;
				if(rules[i].sim == "jac") weightedSimJoin = SimFuncs::weightedJaccard;
				else if(rules[i].sim == "cos") weightedSimJoin = SimFuncs::weightedCosine;
				else weightedSimJoin = SimFuncs::weightedDice;

				ui predict = 0, hit = 0, missing = 0;
				for(ui j = 0; j < numRow; j++) {
					if(curRecordsA[j].empty() || curRecordsB[j].empty()) {
						++ missing;
						continue;
					}
					if(weightedSimJoin(curRecordsA[j], curRecordsB[j], curWordwt, curWeightsA[j], curWeightsB[j]) >= newT) {
						++ predict;
						if(std::stoi(sampleRes.rows[j][posLabel]) == 1)
							++ hit;
					}
				}
				double density = predict * 1.0 / numRow;
				double recall = hit * 1.0 / totalPositive;
				// double precision = hit * 1.0 / predict;
				double precision = std::abs(predict + missing - 0.0) <= 1e-4 ? 0.0 : hit * 1.0 / (predict + missing);
				double F1 = std::abs(precision + recall - 0.0) <= 1e-4 ? 0.0 : 2 * (precision * recall) / (precision + recall);
				double weights = F1;
				// double weights = 1 - density;
				densities.emplace_back(weights);
				std::cout << rules[i].attr << " " << rules[i].sim << " " << rules[i].tok << " " << rules[i].threshold << " "
						  << std::setprecision(4) << weights << std::endl;
				attrAverage[rules[i].attr] += weights;
			}
			else {
				double (*simJoin)(const std::vector<ui> &, const std::vector<ui> &) = nullptr;
				if(rules[i].sim == "jac") simJoin = SimFuncs::jaccard;
				else if(rules[i].sim == "cos") simJoin = SimFuncs::cosine;
				else simJoin = SimFuncs::dice;

				ui predict = 0, hit = 0, missing = 0;
				for(ui j = 0; j < numRow; j++) {
					if(curRecordsA[j].empty() || curRecordsB[j].empty()) {
						++ missing;
						continue;
					}
					if(simJoin(curRecordsA[j], curRecordsB[j]) >= newT) {
						++ predict;
						if(std::stoi(sampleRes.rows[j][posLabel]) == 1)
							++ hit;
					}
				}
				double density = predict * 1.0 / numRow;
				double recall = hit * 1.0 / totalPositive;
				// double precision = hit * 1.0 / predict;
				double precision = std::abs(predict + missing - 0.0) <= 1e-4 ? 0.0 : hit * 1.0 / (predict + missing);
				double F1 = std::abs(precision + recall - 0.0) <= 1e-4 ? 0.0 : 2 * (precision * recall) / (precision + recall);
				double weights = F1;
				// double weights = 1 - density;
				densities.emplace_back(weights);
				std::cout << rules[i].attr << " " << rules[i].sim << " " << rules[i].tok << " " << rules[i].threshold << " "
						  << std::setprecision(4) << weights << std::endl;
				attrAverage[rules[i].attr] += weights;
			}
		}
		else {
			std::cerr << "no such sim function: " << rules[i].sim << std::endl;
			exit(1);
		}

	}

	std::cout << "--- report the average ---" << std::endl;
	for(auto &iter : attrAverage) {
		iter.second = iter.second / attrNumRules[iter.first];
		std::cout << iter.first << " " << iter.second << std::endl;
	}
}


void SimJoinBlocker::selfInterchangeableJoin(uint64_t K, const std::string &topKattr, 
									         const std::string &attrType, bool ifWeighted)
{
	const std::string grpStatPath = "buffer/grp_stat.txt";
	std::ifstream grpStatFile(grpStatPath.c_str(), std::ios::in);
	std::string info;
	getline(grpStatFile, info);
	int totalTable = std::stoi(info);
	getline(grpStatFile, info);
	int totalAttr = std::stoi(info);
	getline(grpStatFile, info);
	std::vector<std::string> statAttrs;
	feature_utils::stringSplit(info, ' ', statAttrs);
	grpStatFile.close();
	GroupInterchangeable group(totalTable, totalAttr, statAttrs);

	for(ui i = 0; i < num_rules; i++) {
		double newT = rules[i].threshold;
		
		printf("---------------------------\n");
		printf("Start similarity join, Attributes: %s  Functions: %s\nTokenizers: %s  Settings: %s  Threshold: %.2lf\n", 
				rules[i].attr.c_str(), rules[i].sim.c_str(), rules[i].tok.c_str(), rules[i].tok_settings.c_str(), 
				newT);
		printf("---------------------------\n");
		fflush(stdout);

		if(K > 0)
			BlockerUtil::pretopKviaTASelf(K, topKattr, attrType, ifWeighted);

		// set-join
		if(rules[i].sim == "jac" || rules[i].sim == "cos" || rules[i].sim == "dice") {
			std::vector<std::pair<int, int>> sim_pairs;
			std::string tokKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			ui pos = datasets_map[tokKey];

			std::vector<int> reverseIdMapA(id_mapA[pos].size(), 0);
			std::vector<int> reverseIdMapB(id_mapB[pos].size(), 0);
			ui idMapASize = id_mapA[pos].size();
			ui idMapBSize = id_mapB[pos].size();
			for(ui j = 0; j < idMapASize; j++)
				reverseIdMapA[id_mapA[pos][j]] = j;
			for(ui j = 0; j < idMapBSize; j++)
				reverseIdMapB[id_mapB[pos][j]] = j;

			// join
            SetJoinParallel *joiner = new SetJoinParallel(recordsA[pos], weightsA[pos], wordwt[pos], newT);
            if(rules[i].sim == "jac") joiner->simFType = SimFuncType::JACCARD;
            else if(rules[i].sim == "cos") joiner->simFType = SimFuncType::COSINE;
            else if(rules[i].sim == "dice") joiner->simFType = SimFuncType::DICE;

            joiner->index(newT);
            joiner->findSimPairsSelf();
            joiner->mergeResults(sim_pairs);

			printf("\033[31mPair Size: %ld\033[0m\n", sim_pairs.size());
			fflush(stdout);

			// synethis
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 0);
			sim_pairs.clear();
			delete joiner;


			// interchangeable join
			int attrPos = group.calAttrIndex(rules[i].attr);
			int numFea = group.calNumFeature(rules[i].attr);
			int feaPos = group.calCahceIndex(rules[i].sim, rules[i].tok, numFea);

			SetJoinParallel *joinerIC = new SetJoinParallel(recordsA[pos], weightsA[pos], wordwt[pos], newT);
			if(rules[i].sim == "jac") joinerIC->simFType = SimFuncType::JACCARD;
			else if(rules[i].sim == "cos") joinerIC->simFType = SimFuncType::COSINE;
			else if(rules[i].sim == "dice") joinerIC->simFType = SimFuncType::DICE;

			joiner->groupA = group.interCGroupsA[attrPos];
			joiner->grpIdA = group.interCGrpIdA[attrPos];
			joiner->idMapA = id_mapA[pos];
			joiner->revIdMapA.resize(id_mapA[pos].size());
			for(size_t idx = 0; idx < id_mapA[pos].size(); idx++)
				joiner->revIdMapA[id_mapA[pos][idx]] = idx;
			joiner->featureValueCache = group.featureValCache[attrPos][feaPos];
			joiner->discreteCacheIdx = group.discreteCacheIdx[attrPos];
		}
		// ovlp-join
		else if(rules[i].sim == "overlap") {
			std::vector<std::pair<int, int>> sim_pairs;
			std::string tokKey = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			ui pos = datasets_map[tokKey];

			// join & synethis
            int othres = ceil(rules[i].threshold - 1e-5);
            OvlpSelfJoinParallel *joiner = new OvlpSelfJoinParallel(recordsA[pos], weightsA[pos], wordwt[pos]);
            joiner->overlapjoin(othres, sim_pairs);

			printf("\033[31mPair Size: %ld\033[0m\n", sim_pairs.size());
			fflush(stdout);

            // synethsis
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 0);

			sim_pairs.clear();
			delete joiner;
		}
		// string-join
		else if(rules[i].sim == "lev") {
			std::vector<std::pair<int, int>> sim_pairs;
			ui pos = table_A.inverted_schema[rules[i].attr];
			
			// join & synethis
            int sthres = floor(rules[i].threshold + 1e-5);

            if(sthres > 0) {
                StringJoinParallel *joiner = new StringJoinParallel(table_A.cols[pos], sthres);
                joiner->selfJoin(sim_pairs);
                BlockerUtil::synthesizePairsRS(pos, sim_pairs, 1);
                delete joiner;
				sim_pairs.clear();
            }
            else {
				bool found = false;
				for(ui r = 0; r < num_rules; r++) {
					if(rules[r].sim == "exm" && rules[r].attr == rules[i].attr) {
						found = true;
						break;
					}
				}

				if(found)
					continue;

                // join
                ExactJoinParallel::exactJoinSelf(table_A.cols[pos], sim_pairs);
                // synethis
				BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 1);
                sim_pairs.clear();
            }
		}
		// others
		else if(rules[i].sim == "exm") {
			std::vector<std::pair<int, int>> sim_pairs;
			ui pos = table_A.inverted_schema[rules[i].attr];
		
			// join
			ExactJoinParallel::exactJoinSelf(table_A.cols[pos], sim_pairs);

			// synethis
			BlockerUtil::synthesizePairsSelf(pos, sim_pairs, 1);
			sim_pairs.clear();
		}
		else {
			std::cerr << "No such sim funcs: " << rules[i].sim << std::endl;
			exit(1);
		}

		// inspect memory
		double vm, rss; // virtual memory & resident set size
		SetJoinUtil::processMemUsage(vm, rss);
		printf("\033[1;34mVirtual memory: %.1lf\nResident set size: %.1lf\033[0m\n", vm, rss);
	}
}