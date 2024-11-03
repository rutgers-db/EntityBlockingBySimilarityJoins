/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "blocker/block.h"

ui num_word = 0;
std::vector<TokenizerType> tok_type;
std::vector<ui> q; 


void Block::clearBuffers()
{
	num_tables = 0;
	num_rules = 0;
	table_A = Table();
	table_B = Table();
	gold = Table();
	// cleared at the end
	rules = nullptr;
	id_mapA.clear();
	id_mapB.clear();
	idStringMapA.clear();
	idStringMapB.clear();
	recordsA.clear();
	recordsB.clear();
	weightsA.clear();
	weightsB.clear();
	wordwt.clear();
	datasets_map.clear();
	final_pairs.clear();
	passedRules.clear();

	num_word = 0;
	tok_type.clear();
	q.clear();
}


void Block::prepareRecordsRS(ui columnA, ui columnB, TokenizerType tt, ui q)
{
	recordsA.emplace_back();
	recordsB.emplace_back();
	id_mapA.emplace_back();
	id_mapB.emplace_back();
	weightsA.emplace_back();
	weightsB.emplace_back();
	wordwt.emplace_back();

	// tokenize & sort records
	ui pos = recordsA.size() - 1;
	Tokenizer::RStableAttr2IntVector(table_A, table_B, recordsA[pos], recordsB[pos], 
									 weightsA[pos], weightsB[pos], wordwt[pos], 
									 id_mapA[pos], id_mapB[pos], 
									 columnA, columnB, tt, num_word, q);
}


void Block::prepareRecordsSelf(ui columnA, TokenizerType tt, ui q)
{
	recordsA.emplace_back();
	recordsB.emplace_back();
	id_mapA.emplace_back();
	id_mapB.emplace_back();
	weightsA.emplace_back();
	weightsB.emplace_back();
	wordwt.emplace_back();

	ui pos = recordsA.size() - 1;
	Tokenizer::SelftableAttr2IntVector(table_A, recordsA[pos], weightsA[pos], 
									   wordwt[pos], id_mapA[pos], columnA, 
									   tt, num_word, q);
	recordsB[pos] = recordsA[pos];
	weightsB[pos] = weightsA[pos];
	id_mapB[pos] = id_mapA[pos];
}


void Block::sortColumns()
{
	ui schemaSize = table_A.schema.size();

	idStringMapA.resize(schemaSize);
	idStringMapB.resize(schemaSize);
	
	// skip attr "_id"
	for(ui i = 1; i < schemaSize; i++) {
		ui size = table_A.cols[i].size();
		for(ui j = 0; j < size; j++)
			idStringMapA[i].emplace_back(j);
		
		size = table_B.cols[i].size();
		for(ui j = 0; j < size; j++)
			idStringMapB[i].emplace_back(j);

		// map id
		auto mapSortFuncA = [i](const ui &a, const ui &b) {
			std::string s1(table_A.cols[i][a]);
			std::string s2(table_A.cols[i][b]);

			if (s1.length() < s2.length()) 
				return true;
			else if (s1.length() > s2.length()) 
				return false;
			return s1 < s2;
		};
		auto mapSortFuncB = [i](const ui &a, const ui &b) {
			std::string s1(table_B.cols[i][a]);
			std::string s2(table_B.cols[i][b]);

			if (s1.length() < s2.length()) 
				return true;
			else if (s1.length() > s2.length()) 
				return false;
			return s1 < s2;
		};

		std::sort(idStringMapA[i].begin(), idStringMapA[i].end(), mapSortFuncA);
		std::sort(idStringMapB[i].begin(), idStringMapB[i].end(), mapSortFuncB);

		// sort columns
		// std::sort(table_A.cols[i].begin(), table_A.cols[i].end(), StringJoinUtil::strLessT);
		// std::sort(table_B.cols[i].begin(), table_B.cols[i].end(), StringJoinUtil::strLessT);
	}
}


void Block::showPara(int jt, int js, uint64_t topK, const std::string &topKattr, 
					 const std::string &attrType, const std::string &pathTableA, 
					 const std::string &pathTableB, const std::string &pathGold, 
					 const std::string &pathRule, int tableSize)
{
	std::string joinType = jt == 0 ? "self" : "RS";
	std::string joinSetting = js == 0 ? "sequential" : "parallel";
	std::cout << "-- join type: " << joinType << std::endl;
	std::cout << "-- join setting: " << joinSetting << std::endl;
	std::cout << "-- top K: " << topK << std::endl;
	std::cout << "-- topk attribute: " << topKattr << " " << attrType << std::endl;
	std::cout << "-- data paths: " << pathTableA << " " << pathTableB << " "
								  << pathGold << std::endl;
	std::cout << "-- rule path: " << pathRule << std::endl; 
	std::cout << "-- match result table size: " << tableSize << std::endl;
	if(pathTableA == "")
		std::cout << "if data paths are not specified by input, set them as default as in \"buffer\" folder" << std::endl;
}


void Block::readCSVTables(int isRS, const std::string &pathTableA, const std::string &pathTableB, 
				   		  const std::string &pathGold)
{
	CSVReader reader;

	bool normalize = false;
	std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory += "../../output/";
	std::string cleanA = pathTableA == "" ? directory + "buffer/clean_A.csv" : pathTableA;
	std::string cleanB = pathTableB == "" ? directory + "buffer/clean_B.csv" : pathTableB;
	std::string cleanG = pathGold == "" ? directory + "buffer/gold.csv" : pathGold;

	reader.reading_one_table(cleanA, normalize);
	table_A = reader.tables[0];
	table_A.findPerfectEntity();
	if(isRS == 1) {
		reader.reading_one_table(cleanB, normalize);
		table_B = reader.tables[1];
		table_B.findPerfectEntity();
		reader.reading_one_table(cleanG, normalize);
		gold = reader.tables[2];
	}
	else {
		table_B = table_A;
		reader.reading_one_table(cleanG, normalize);
		gold = reader.tables[1];
	}
	
	table_A.Profile();
	table_B.Profile();
	gold.Profile();

	final_pairs.resize(table_A.rows.size());
	passedRules.resize(table_A.rows.size());

	table_A.printData();
}


void Block::readRules(const std::string &pathRule)
{
	RuleReader::readRules(num_rules, rules, pathRule);

#if PRINT_RULES == 1
	for(ui i = 0; i < num_rules; i++) {
		printf("%s %s %s %s %s %d %lf\n", rules[i].attr.c_str(), rules[i].sim.c_str(), 
										  rules[i].sim_measure.c_str(), rules[i].tok.c_str(), 
										  rules[i].tok_settings.c_str(), rules[i].sign, 
										  rules[i].threshold);
	}
#endif
}


void Block::tokenize(int isRS)
{
	// tokenize for set join
	for(ui i = 0; i < num_rules; i++) {
		ui columnA = table_A.inverted_schema[rules[i].attr];
		ui columnB = table_B.inverted_schema[rules[i].attr];
		if(!columnA || !columnB) {
			printf("Error in attrs: %d %d %s\n", columnA, columnB, rules[i].attr.c_str());
			exit(1);
		}

		// For set-join / overlap-join
		if(strcmp(rules[i].tok.c_str(), "none")) {
			std::string tok_all = rules[i].tok + "_" + rules[i].tok_settings + "_" + rules[i].attr;
			
			// Check if need new tokenized records
			ui pos = 0;
			if(datasets_map.find(tok_all) == datasets_map.end()) {
				printf("Start tokenize new records: %s %s %s\n", rules[i].tok.c_str(), 
																 rules[i].tok_settings.c_str(), 
																 rules[i].attr.c_str());
				// fflush(stdout);

				if(!strcmp(rules[i].tok.c_str(), "qgm")) {
					TokenizerType tt = TokenizerType::QGram;
					ui q = atoi(rules[i].tok_settings.c_str());
					// tokenize
					if(isRS == 1)
						prepareRecordsRS(columnA, columnB, tt, q);
					else
						prepareRecordsSelf(columnA, tt, q);
				}
				else if(!strcmp(rules[i].tok.c_str(), "dlm")) {
					TokenizerType tt = TokenizerType::Dlm;
					// tokenize
					if(isRS == 1)
						prepareRecordsRS(columnA, columnB, tt, 0);
					else
						prepareRecordsSelf(columnA, tt, 0);
				}
				else if(!strcmp(rules[i].tok.c_str(), "wspace")) {
					TokenizerType tt = TokenizerType::WSpace;
					// tokenize
					if(isRS == 1)
						prepareRecordsRS(columnA, columnB, tt, 0);
					else
						prepareRecordsSelf(columnA, tt, 0);
				}
				else if(!strcmp(rules[i].tok.c_str(), "alphanumeric")) {
					TokenizerType tt = TokenizerType::AlphaNumeric;
					// tokenize
					if(isRS == 1)
						prepareRecordsRS(columnA, columnB, tt, 0);
					else
						prepareRecordsSelf(columnA, tt, 0);
				}
				else {
					printf("No such tokenizer: %s\n", rules[i].tok.c_str());
					exit(1);
				}

				pos = recordsA.size() - 1;
				datasets_map[tok_all] = pos;
			}
		}
	}

	// sort the columns for string join
	sortColumns();
}


void Block::getRecall(int isRS)
{
	ui size = final_pairs.size();
	for(ui i = 0; i < size; i++) {
		// is ordered
		if(!std::is_sorted(final_pairs[i].begin(), final_pairs[i].end())) {
			std::cerr << "Final res " << i << " not ordered" << std::endl << std::flush;
			exit(1);
		}
		// is unique
		auto it = std::unique(final_pairs[i].begin(), final_pairs[i].end());
		if(it != final_pairs[i].end()) {
			std::cerr << "Duplicate results in " << i << " " << std::distance(it, final_pairs[i].end()) << std::endl << std::flush;
			exit(1);
		}
	}

	std::vector<std::pair<int, int>> verifySet, finalSet;

	ui num_gold = gold.rows.size();
	for(ui i = 0; i < num_gold; i++) {
		int idA = atoi(gold.rows[i][0].c_str());
		int idB = atoi(gold.rows[i][1].c_str());
		verifySet.emplace_back(idA, idB);
	}
	
	ui final_size = final_pairs.size();
	for(ui i = 0; i < final_size; i++)
		for(const auto &id : final_pairs[i])
			finalSet.emplace_back((int)i, id);

	std::vector<std::pair<int, int>> resSet;
	__gnu_parallel::sort(verifySet.begin(), verifySet.end());
	__gnu_parallel::sort(finalSet.begin(), finalSet.end());
	__gnu_parallel::set_intersection(verifySet.begin(), verifySet.end(), finalSet.begin(), finalSet.end(), 
									 std::back_inserter(resSet));

	uint64_t cartesian = isRS == 0 ? (uint64_t)table_A.row_no * uint64_t(table_A.row_no - 1) / 2
								   : (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;

	std::cout << "---------------------------" << std::endl;
	std::cout <<                         "cartesian: " << cartesian << " " << table_A.row_no << " " << isRS << std::endl;
	std::cout << std::setprecision(4) << "   recall: " << resSet.size() * 1.0 / verifySet.size() << std::endl;
	std::cout <<                         "      |C|: " << finalSet.size() << std::endl;
	std::cout << std::setprecision(4) << "     CSSR: " << finalSet.size() * 1.0 / cartesian << std::endl;
	std::cout << "---------------------------" << std::endl << std::flush;
}


void Block::getRecall4Rules(int isRS)
{
	ui maxPassedRules = 0;
	std::unordered_map<int, std::vector<std::pair<int, int>>> invertedBuckets;

	for(int i = 0; i < table_A.row_no; i++) {
		for(const auto &p : passedRules[i]) {
			invertedBuckets[p.second].emplace_back(i, p.first);
			maxPassedRules = maxPassedRules < p.second ? p.second : maxPassedRules;
		}
	}
	for(ui i = 2; i <= maxPassedRules; i++) {
		// invertedBuckets[i].insert(invertedBuckets[i].end(), invertedBuckets[i - 1].begin(), 
		// 						  invertedBuckets[i - 1].end());
		__gnu_parallel::sort(invertedBuckets[i].begin(), invertedBuckets[i].end());
	}

	std::cout << "num rules: " << num_rules << " max passed: " << maxPassedRules << std::endl;

	ui num_gold = gold.rows.size();
	uint64_t cartesian = isRS == 0 ? (uint64_t)table_A.row_no * uint64_t(table_A.row_no - 1) / 2
								   : (uint64_t)table_A.row_no * (uint64_t)table_B.row_no;
	std::vector<std::pair<int, int>> verifySet;

	for(ui i = 0; i < num_gold; i++) {
		int idA = atoi(gold.rows[i][0].c_str());
		int idB = atoi(gold.rows[i][1].c_str());
		verifySet.emplace_back(idA, idB);
	}
	__gnu_parallel::sort(verifySet.begin(), verifySet.end());

	std::vector<std::pair<int, int>> topkResults;
	for(int i = 0; i < table_A.row_no; i++)
		for(const auto &id : final_pairs[i])
			topkResults.emplace_back(i, id);
	__gnu_parallel::sort(topkResults.begin(), topkResults.end());

	ui total = 0;
	std::vector<std::pair<int, int>> tmp, tmp2, tmp3;
	std::vector<double> percentage;
	for(ui i = 1; i <= maxPassedRules; i++) {
		tmp.clear();
		__gnu_parallel::set_intersection(verifySet.begin(), verifySet.end(), 
										 invertedBuckets[i].begin(), invertedBuckets[i].end(), 
										 std::back_inserter(tmp));
		tmp2.clear();
		__gnu_parallel::set_intersection(topkResults.begin(), topkResults.end(), 
										 invertedBuckets[i].begin(), invertedBuckets[i].end(), 
										 std::back_inserter(tmp2));
		tmp3.clear();
		__gnu_parallel::set_intersection(verifySet.begin(), verifySet.end(), 
										 tmp2.begin(), tmp2.end(), 
										 std::back_inserter(tmp3));

		std::cout << "---------------------------" << std::endl;
		std::cout << "      result for # of passed rule (without previous): " << i << std::endl;
		std::cout << std::setprecision(4) << "          recall on raw data: " << tmp.size() * 1.0 / num_gold * 1.0 << std::endl;
		std::cout << std::setprecision(4) << "       recall on topk result: " << tmp3.size() * 1.0 / num_gold * 1.0 << std::endl;
		std::cout <<                         "					       |C|: " << invertedBuckets[i].size() << std::endl;
		std::cout <<                         "				        |tmp2|: " << tmp2.size() << std::endl;
		std::cout << std::setprecision(4) << "				          CSSR: " << invertedBuckets[i].size() * 1.0 / cartesian * 1.0 << std::endl;
		std::cout << std::setprecision(4) << "|tmp2| / |invertedBucket[i]|:" << tmp2.size() * 1.0 / invertedBuckets[i].size() << std::endl;
		std::cout << "---------------------------" << std::endl << std::flush;

		total += tmp.size();
		percentage.emplace_back(tmp2.size() * 1.0 / topkResults.size());
	}

	std::cout << std::setprecision(4) << "recall: " << total * 1.0 / num_gold << std::endl;
	std::cout << "percentage: ";
	for(const auto &per : percentage)
		std::cout << std::setprecision(4) << per << " ";
	std::cout << std::endl;
}


extern "C" 
{
	void sim_join_block(int join_type, int join_setting, uint64_t top_k, const char *topk_attr, const char *attr_type, 
						const char *path_table_A, const char *path_table_B, const char *path_gold, 
						const char *path_rule, int table_size, bool is_join_topk, bool is_idf_weighted, 
						const char *default_output_dir, const char *default_sample_res_dir) {

		Block::showPara(join_type, join_setting, top_k, topk_attr, attr_type, path_table_A, path_table_B, path_gold, 
						path_rule, table_size);

		// top_k *= 5;
		
		timeval readingBegin, tokenizeBegin, joinBegin;
		timeval readingEnd, tokenizeEnd, joinEnd;
		double readingTime, tokenizeTime, joinTime, allTime;

		// reading
		gettimeofday(&readingBegin, NULL);
		Block::readCSVTables(join_type, path_table_A, path_table_B, path_gold);
		Block::readRules(path_rule);
		gettimeofday(&readingEnd, NULL);
		printf("~~~Finish reading~~~\n");

		// txtnorm
		gettimeofday(&tokenizeBegin, NULL);
		Block::tokenize(join_type);
		gettimeofday(&tokenizeEnd, NULL);
		printf("~~~Finish tokenizing~~~\n");

		// blocking
		gettimeofday(&joinBegin, NULL);
		if(join_type == 0 && join_setting == 1) {
			printf("----------Self blocking with Parallel joiners----------\n");
			SimJoinBlocker::selfSimilarityJoinParallel(top_k, topk_attr, attr_type, is_idf_weighted);
		}
		else if(join_type == 1 && join_setting == 0) {
			printf("----------RS blocking with Sequential joiners----------\n");
			SimJoinBlocker::RSSimilarityJoinSerial(top_k, topk_attr, attr_type, is_idf_weighted, is_join_topk);
		}
		else {
			std::cerr << "Desired framework is not avaliable" << std::endl;
			exit(1);
		}
		gettimeofday(&joinEnd, NULL);
		printf("~~~Finish joining~~~\n");

		std::vector<double> simWeights;
		std::unordered_map<std::string, double> attrAverage;
		SimJoinBlocker::estimateDensity(false, simWeights, attrAverage, default_sample_res_dir);

		// topK
		double topKTime = 0.0;
		if(top_k != 0) {
			timeval topKBegin, topKEnd;
			gettimeofday(&topKBegin, NULL);

			std::vector<std::vector<int>> tempFinalPairs = final_pairs;
			std::vector<std::vector<int>> finalPairsBackup = final_pairs;

			std::string fullPath = __FILE__;
			size_t lastSlash = fullPath.find_last_of("/\\");
			std::string directory = fullPath.substr(0, lastSlash + 1);
			const std::string defaultOutputDir = default_output_dir;
			directory = defaultOutputDir == "" ? directory + "../../output/topk_stat/" 
											   : (defaultOutputDir.back() == '/' ? defaultOutputDir : defaultOutputDir + "/");
			const std::string pathTopKOutput = directory + "intermedia.txt";
			std::ofstream topKStream(pathTopKOutput, std::ios::out);

			if(join_type == 0) {
				TopK::topKviaTASelf(table_A, topk_attr, attr_type, 
									recordsA, weightsA, wordwt, 
									datasets_map, id_mapA, 
									tempFinalPairs, gold, top_k, topKStream, 
									is_idf_weighted, "exp");
				tempFinalPairs = finalPairsBackup;
				TopK::topKviaAllSimilarityScoreSelf(table_A, rules, simWeights, attrAverage, recordsA, 
													weightsA, wordwt, datasets_map, id_mapA, 
													idStringMapA, tempFinalPairs, gold, top_k, 
													num_rules, topKStream, is_idf_weighted, "exp");
			}
			else if(join_type == 1) {
				TopK::topKviaTARS(table_A, table_B, topk_attr, attr_type, 
								  recordsA, recordsB, weightsA, weightsB, 
								  wordwt, datasets_map, id_mapA, id_mapB, 
								  tempFinalPairs, gold, top_k, topKStream, 
								  is_idf_weighted, "exp");
				tempFinalPairs = finalPairsBackup;
				TopK::topKviaAllSimilarityScoresRS(table_A, table_B, rules, simWeights, attrAverage,
												   recordsA, recordsB, weightsA, weightsB, 
												   wordwt, datasets_map, id_mapA, id_mapB, 
												   idStringMapA, idStringMapB, tempFinalPairs, 
												   gold, top_k, num_rules, topKStream, is_idf_weighted, "exp");
			}

			gettimeofday(&topKEnd, NULL);
			topKTime = topKEnd.tv_sec - topKBegin.tv_sec + (topKEnd.tv_usec - topKBegin.tv_usec) / 1e6;
			printf("~~~Finish topK~~~\n");
			
			// topKStream.close();
		}
		
		// Result
		if(join_type == 0)
			table_B.row_no = (table_B.row_no - 1) / 2;
		Block::getRecall(join_type);
		printf("~~~Finish calculating recall~~~\n");

		// Flush
		MultiWriter::writeBlockResMegallenCSV(table_A, table_B, table_size, final_pairs, default_output_dir);

		// Release
		printf("Releasing buffers..........\n");
		delete[] rules;

		// Time
		readingTime = readingEnd.tv_sec - readingBegin.tv_sec + (readingEnd.tv_usec - readingBegin.tv_usec) / 1e6;
		tokenizeTime = tokenizeEnd.tv_sec - tokenizeBegin.tv_sec + (tokenizeEnd.tv_usec - tokenizeBegin.tv_usec) / 1e6;
		joinTime = joinEnd.tv_sec - joinBegin.tv_sec + (joinEnd.tv_usec - joinBegin.tv_usec) / 1e6;
		allTime = readingTime + tokenizeTime + joinTime;

		printf("###     Reading Time: %.4lf\n", readingTime);
		printf("###    Tokenize Time: %.4lf\n", tokenizeTime);
		printf("###        Join Time: %.4lf\n", joinTime);
		if(top_k != 0) {
			printf("###        TopK Time: %.4lf\n", topKTime);
			allTime += topKTime;
		}
		printf("###         All Time: %.4lf\n", allTime);

		Block::clearBuffers();
	}

	void knn_block() {
		std::cerr << "not establish" << std::endl;
		exit(1);
	}
}