/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/io.h"
#include "blocker/blocker_config.h"
#include "blocker/block.h"

extern ui num_word;
extern std::vector<TokenizerType> tok_type;
extern std::vector<ui> q; 


int main(int argc, char* argv[])
{
	int jt = atoi(argv[1]);
	int js = atoi(argv[2]);
	uint64_t topK = std::stoull(argv[3]);
	std::string topKattr = argv[4];
	std::string attrType = argv[5];
	std::string pathTableA = argv[6];
	std::string pathTableB = argv[7];
	std::string pathGold = argv[8];
	std::string pathRule = argv[9];
	int tableSize = atoi(argv[10]);
	bool isJoinTopK = atoi(argv[11]) == 1 ? true : false;
	bool isIdfWeighted = atoi(argv[12]) == 1 ? true : false;
	Block::showPara(jt, js, topK, topKattr, attrType, pathTableA, pathTableB, pathGold, pathRule, tableSize);

	topK *= 5;
	
	timeval readingBegin, tokenizeBegin, joinBegin;
	timeval readingEnd, tokenizeEnd, joinEnd;
	double readingTime, tokenizeTime, joinTime, allTime;

	// reading
	gettimeofday(&readingBegin, NULL);
	Block::readCSVTables(jt, pathTableA, pathTableB, pathGold);
	Block::readRules(pathRule);
	gettimeofday(&readingEnd, NULL);

	printf("~~~Finish reading~~~\n");
	fflush(stdout);

	// txtnorm
	gettimeofday(&tokenizeBegin, NULL);
	Block::tokenize(jt);
	gettimeofday(&tokenizeEnd, NULL);

	printf("~~~Finish tokenizing~~~\n");
	fflush(stdout);

	// blocking
	gettimeofday(&joinBegin, NULL);
	if(jt == 0 && js == 1) {
		printf("----------Self blocking with Parallel joiners----------\n");
		SimJoinBlocker::selfSimilarityJoinParallel(topK, topKattr, attrType, isIdfWeighted);
	}
	else if(jt == 1 && js == 0) {
		printf("----------RS blocking with Sequential joiners----------\n");
		SimJoinBlocker::RSSimilarityJoinSerial(topK, topKattr, attrType, isIdfWeighted, isJoinTopK);
	}
    else {
        std::cerr << "Desired framework is not avaliable" << std::endl;
        exit(1);
    }
    
	gettimeofday(&joinEnd, NULL);

	printf("~~~Finish joining~~~\n");
	fflush(stdout);

	std::vector<double> simWeights;
	std::unordered_map<std::string, double> attrAverage;
	SimJoinBlocker::estimateDensity(false, simWeights, attrAverage);

	// topK
	double topKTime = 0.0;
	if(topK != 0) {
		timeval topKBegin, topKEnd;
		gettimeofday(&topKBegin, NULL);

		std::vector<std::vector<int>> tempFinalPairs = final_pairs;
		std::vector<std::vector<int>> finalPairsBackup = final_pairs;
		const std::string pathTopKOutput = "output/topk_stat/intermedia.txt";
		std::ofstream topKStream(pathTopKOutput, std::ios::out);

		if(jt == 0) {
			TopK::topKviaTASelf(table_A, topKattr, attrType, 
								recordsA, weightsA, wordwt, 
								datasets_map, id_mapA, 
								tempFinalPairs, gold, topK, topKStream, 
								isIdfWeighted, "exp");
			tempFinalPairs = finalPairsBackup;
			TopK::topKviaAllSimilarityScoreSelf(table_A, rules, simWeights, attrAverage, recordsA, 
												weightsA, wordwt, datasets_map, id_mapA, 
												idStringMapA, tempFinalPairs, gold, topK, 
												num_rules, topKStream, isIdfWeighted, "exp");
		}
		else if(jt == 1) {
			TopK::topKviaTARS(table_A, table_B, topKattr, attrType, 
							  recordsA, recordsB, weightsA, weightsB, 
							  wordwt, datasets_map, id_mapA, id_mapB, 
							  tempFinalPairs, gold, topK, topKStream, 
							  isIdfWeighted, "exp");
			tempFinalPairs = finalPairsBackup;
			TopK::topKviaAllSimilarityScoresRS(table_A, table_B, rules, simWeights, attrAverage,
											   recordsA, recordsB, weightsA, weightsB, 
											   wordwt, datasets_map, id_mapA, id_mapB, 
											   idStringMapA, idStringMapB, tempFinalPairs, 
											   gold, topK, num_rules, topKStream, isIdfWeighted, "exp");
		}

		gettimeofday(&topKEnd, NULL);
		topKTime = topKEnd.tv_sec - topKBegin.tv_sec + (topKEnd.tv_usec - topKBegin.tv_usec) / 1e6;
		printf("~~~Finish topK~~~\n");
		fflush(stdout);
	}
	
	// Result
	if(jt == 0)
		table_B.row_no = (table_B.row_no - 1) / 2;
	Block::getRecall(jt);

	printf("~~~Finish calculating recall~~~\n");
	fflush(stdout);

	// Flush
    MultiWriter::writeBlockResMegallenCSV(table_A, table_B, tableSize, final_pairs);

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
	if(topK != 0) {
		printf("###        TopK Time: %.4lf\n", topKTime);
		allTime += topKTime;
	}
	printf("###         All Time: %.4lf\n", allTime);

	return 0;
}