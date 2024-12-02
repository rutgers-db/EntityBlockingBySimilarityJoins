/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "feature/feature.h"


std::string FeatureEngineering::getDefaultICVDir(const std::string &defaultICVDir)
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultICVDir == "" ? directory + "../../simjoin_entitymatching/value_matcher/ic_values/" 
									: (defaultICVDir.back() == '/' ? defaultICVDir : defaultICVDir + "/");

    return directory;
}


void FeatureEngineering::readGroups(int totalAttr, const std::vector<std::string> &attrVec, FeatureIndex::Groups &group, 
                                    FeatureIndex::GroupTokens &groupTokensDlm, FeatureIndex::GroupTokens &groupTokensQgm, 
                                    FeatureIndex::Cluster &cluster, std::vector<int> &keyLength, 
                                    const std::string &defaultICVDir)
{
    std::string directory = getDefaultICVDir(defaultICVDir);

    for(int i = 0; i < totalAttr; i ++) {
        
        std::string grpPath = directory + "interchangeable_grp_" + attrVec[i] + ".txt";
        auto &curGrp = group[i];
        auto &curGrpDlm = groupTokensDlm[i];
        auto &curGrpQgm = groupTokensQgm[i];
        auto &curClt = cluster[i];

        // grp
        std::string entity;
        std::vector<std::string> entityVec;

        std::ifstream grpFile(grpPath.c_str(), std::ios::in);

        getline(grpFile, entity);
        int totalKey = std::stoi(entity); 
        keyLength.emplace_back(totalKey);

        for(int j = 0; j < totalKey; j++) {
            entityVec.clear();
            getline(grpFile, entity);
            FeatureUtils::stringSplit(entity, ' ', entityVec);
            int keyId = std::stoi(entityVec[0]);
            int length = std::stoi(entityVec[1]);

            if(length <= 1) 
                continue;
            
            for(int l = 0; l < length; l++) {
                std::string doc;
                getline(grpFile, doc);
                curGrp[keyId].emplace_back(doc);
                std::vector<std::string> tokensDlm;
                std::vector<std::string> tokensQgm;
                FeatureUtils::tokenize(doc, TokenizerType::Dlm, tokensDlm);
                FeatureUtils::tokenize(doc, TokenizerType::QGram, tokensQgm);
                curGrpDlm[keyId].emplace_back(tokensDlm);
                curGrpQgm[keyId].emplace_back(tokensQgm);
                curClt[doc] = keyId;
            }
        }

        grpFile.close();
    }
}


void FeatureEngineering::readGraphs(int totalAttr, const std::vector<std::string> &attrVec, Graphs &semanticGraph, 
                                    const std::string &defaultICVDir = "")
{
    std::string directory = getDefaultICVDir(defaultICVDir);

    for(int i = 0; i < totalAttr; i ++) {
        // build
        std::string graphPath = directory + "interchangeable_graph_" + attrVec[i] + ".txt";
        semanticGraph.emplace_back();
        semanticGraph.back().buildSemanticGraph(graphPath);

        // tokenize
        auto &curGraph = semanticGraph.back();
        std::vector<std::vector<std::string>> dlm;
        std::vector<std::vector<std::string>> qgm;

        for(const auto &doc : curGraph.docs) {
            std::vector<std::string> tokensDlm;
            std::vector<std::string> tokensQgm;
            FeatureUtils::tokenize(doc, TokenizerType::Dlm, tokensDlm);
            FeatureUtils::tokenize(doc, TokenizerType::QGram, tokensQgm);
            dlm.emplace_back(std::move(tokensDlm));
            qgm.emplace_back(std::move(tokensQgm));
        }

        curGraph.updateTokenizedDocs(dlm, qgm);
    }
}


void FeatureEngineering::readFeatures(ui &numFeatures, Rule *&featureNames, std::vector<std::string> &nameCopy, 
                                      const std::string &defaultFeatureNamesDir)
{
    // check this! the directory is incorrect
    RuleReader::readFeatureNames(numFeatures, featureNames, defaultFeatureNamesDir);

    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultFeatureNamesDir == "" ? directory + "../../output/buffer/" 
											 : (defaultFeatureNamesDir.back() == '/' ? defaultFeatureNamesDir 
																				     : defaultFeatureNamesDir + "/");
	const std::string pathFeatureName = directory + "feature_names.txt";

    FILE *copyfp = fopen(pathFeatureName.c_str(), "r");
    int dummy1 = 0;
    fscanf(copyfp, "%d\n", &dummy1);
    for(int i = 0; i < dummy1; i++) {
        char nameBuffer[200];
        fscanf(copyfp, "%s\n", nameBuffer);
        nameCopy.emplace_back(nameBuffer);
    }
    fclose(copyfp);
}


void FeatureEngineering::extractFeatures4Matching(int isInterchangeable, bool flagConsistent, int totalTable,
                                                  const FeatureArguments *attrs, const std::string &defaultFeatureVecDir, 
                                                  const std::string &defaultResTableName, const std::string &defaultICVDir, 
                                                  const std::string &defaultFeatureNamesDir)
{
    int totalAttr = attrs->totalAttr;
    std::vector<std::string> attrVec;
    std::vector<int> featureLength;
    for(int i = 0; i < totalAttr; i++) {
        attrVec.emplace_back(attrs->attributes[i]);
        featureLength.emplace_back(CalculateFeature::index.calNumFeature(attrs->attributes[i]));
    }

    printf("is interchangeable: %d\tflag consistent: %d\n", isInterchangeable, (int)flagConsistent);
    printf("number of tables: %d\tnumber of attrs: %d\n", totalTable, totalAttr);
    for(const auto &attr : attrVec)
        printf("%s\t", attr.c_str());
    printf("\n");

    CSVReader reader;
    // grp_id -> a set of interchangeable strings
    std::vector<std::unordered_map<int, std::vector<std::string>>> group;
    std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>> groupTokensDlm;
    std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>> groupTokensQgm;
    // id -> grp_id
    std::vector<std::unordered_map<std::string, int>> cluster;

    group.resize(totalAttr);
    groupTokensDlm.resize(totalAttr);
    groupTokensQgm.resize(totalAttr);
    cluster.resize(totalAttr);

    // io
    std::vector<int> keyLength;
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultFeatureVecDir == "" ? directory + "../../output/blk_res/" 
										   : (defaultFeatureVecDir.back() == '/' ? defaultFeatureVecDir 
																				 : defaultFeatureVecDir + "/");

    std::string resTableName = defaultResTableName == "" ? "blk_res" : defaultResTableName;
    for(int i = 0; i < totalTable; i ++) {
        std::string blkResPath = directory + resTableName + std::to_string(i) + ".csv";
        bool success = reader.reading_one_table(blkResPath, false);
    }

    if(isInterchangeable)
        readGroups(totalAttr, attrVec, group, groupTokensDlm, groupTokensQgm, cluster, keyLength, defaultICVDir);
    printf("done io\n");
    fflush(stdout);

    // fast group
    timeval initBegin, initEnd;
    gettimeofday(&initBegin, NULL);
    if(!flagConsistent && isInterchangeable)
        CalculateFeature::index.globalInit(keyLength, attrVec, group, groupTokensDlm, groupTokensQgm, false);
    gettimeofday(&initEnd, NULL);
    printf("done global init\n");

    // get feature table
    // assign function
    ui numFeatures;
    Rule *featureNames;
    std::vector<std::string> nameCopy;
    readFeatures(numFeatures, featureNames, nameCopy, defaultFeatureNamesDir);

    // group
    timeval begin, end;
    gettimeofday(&begin, NULL);

#pragma omp parallel for
    for(int i = 0; i < totalTable; i++) {
        Table blkRes = reader.tables[i];
        blkRes.Profile();
        std::vector<std::vector<double>> featureValues;

        // cal
        if(isInterchangeable)
            CalculateFeature::calAll(numFeatures, featureNames, attrVec, blkRes, featureValues, 
                                     group, groupTokensDlm, groupTokensQgm, cluster, featureLength, 
                                     flagConsistent, false);
        else
            CalculateFeature::calAllWithoutInterchangeable(numFeatures, featureNames, attrVec, blkRes, 
                                                           featureValues, featureLength, false);

        // flush
        std::string featureVecPath = directory + "feature_vec" + std::to_string(i) + ".csv";
        FILE *fvfile = fopen(featureVecPath.c_str(), "w");

        // header
        fprintf(fvfile, "id,ltable_id,rtable_id");
        for(const auto &copy : nameCopy)
            fprintf(fvfile, ",%s", copy.c_str());
        fprintf(fvfile, "\n");

        // no need for escaping
        for(int ridx = 0; ridx < blkRes.row_no; ridx ++) {
            const auto &curRow = blkRes.rows[ridx];
            int _id = std::stoi(curRow[0]);
            int lid = std::stoi(curRow[1]);
            int rid = std::stoi(curRow[2]);
            fprintf(fvfile, "%d,%d,%d", _id, lid, rid);
            for(const auto &val : featureValues[ridx]) {
                if(std::abs(std::abs(val) - std::abs(FeatureUtils::NaN)) < 1e-5)
                    fprintf(fvfile, ",");
                else
                    fprintf(fvfile, ",%.10lf", val);
            }
            fprintf(fvfile, "\n");
        }

        fflush(fvfile);
        fclose(fvfile);
    }

    gettimeofday(&end, NULL);
    double time = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
    double initTime = initEnd.tv_sec - initBegin.tv_sec + (initEnd.tv_usec - initBegin.tv_usec) / 1e6;
    std::cout << initTime << " " << time << std::endl;

    CalculateFeature::index.releaseMemory();
    return;
}


void FeatureEngineering::extractFeatures4TopK(int isInterchangeable, bool flagConsistent, int totalTable, 
                                              const FeatureArguments *attrs, const std::string &defaultFeatureVecDir, 
                                              const std::string &defaultICVDir, const std::string &defaultFeatureNamesDir)
{
    int totalAttr = attrs->totalAttr;
    std::vector<std::string> attrVec;
    std::vector<int> featureLength;
    for(int i = 0; i < totalAttr; i++) {
        attrVec.emplace_back(attrs->attributes[i]);
        featureLength.emplace_back(CalculateFeature::index.calNumFeature(attrs->attributes[i]));
    }
    // bool flagConsistent = flagICVal == 1 ? false : true;

    printf("number of tables: %d\tnumber of attrs: %d\n", totalTable, totalAttr);
    for(const auto &attr : attrVec)
        printf("%s\t", attr.c_str());
    printf("\n");

    CSVReader reader;
    // grp_id -> a set of interchangeable strings
    std::vector<std::unordered_map<int, std::vector<std::string>>> group;
    std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>> groupTokensDlm;
    std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>> groupTokensQgm;
    // id -> grp_id
    std::vector<std::unordered_map<std::string, int>> cluster;

    group.resize(totalAttr);
    groupTokensDlm.resize(totalAttr);
    groupTokensQgm.resize(totalAttr);
    cluster.resize(totalAttr);

    // io
    std::vector<int> keyLength;
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultFeatureVecDir == "" ? defaultFeatureVecDir + "../../output/match_res/" 
										   : (defaultFeatureVecDir.back() == '/' ? defaultFeatureVecDir 
																			 : defaultFeatureVecDir + "/");
    std::string matchresPath = directory + "match_res.csv";
    bool success = reader.reading_one_table(matchresPath, false);
    Table matchRes = reader.tables[0];
    matchRes.Profile();
    
    if(isInterchangeable)
        readGroups(totalAttr, attrVec, group, groupTokensDlm, groupTokensQgm, 
                   cluster, keyLength, defaultICVDir);
    printf("done io\n");

    // fast group
    if(!flagConsistent && isInterchangeable)
        CalculateFeature::index.globalInit(keyLength, attrVec, group, groupTokensDlm, groupTokensQgm, true);

    // get feature table
    // assign function
    ui numFeatures;
    Rule *featureNames;
    std::vector<std::string> nameCopy;
    readFeatures(numFeatures, featureNames, nameCopy, defaultFeatureNamesDir);

    // group
    timeval begin, end;
    gettimeofday(&begin, NULL);

    std::vector<std::vector<double>> featureValues;
    if(isInterchangeable)
        CalculateFeature::calAll(numFeatures, featureNames, attrVec, matchRes, featureValues, 
                                  group, groupTokensDlm, groupTokensQgm, cluster, featureLength, 
                                  flagConsistent, true);
    else
        CalculateFeature::calAllWithoutInterchangeable(numFeatures, featureNames, attrVec, matchRes, 
                                                        featureValues, featureLength, true);

    // flush
    std::string featureVecPath = directory + "feature_vec.csv";
    FILE *fvfile = fopen(featureVecPath.c_str(), "w");

    // header
    fprintf(fvfile, "id,ltable_id,rtable_id");
    for(const auto &copy : nameCopy)
        fprintf(fvfile, ",%s", copy.c_str());
    fprintf(fvfile, "\n");

    // no need for escaping
    // for top k, we need to fill in NaN
    std::vector<double> means(numFeatures, 0.0);
    std::vector<int> length(numFeatures, 0);
    for(int ridx = 0; ridx < matchRes.row_no; ridx ++) {
        const auto &curRow = featureValues[ridx];
        int rsize = (int)curRow.size();
        for(int j = 0; j < rsize; j++) {
            double val = curRow[j];
            if(std::abs(std::abs(val) - std::abs(FeatureUtils::NaN)) < 1e-5)   
                continue;
            ++ length[j];
            means[j] += val;
        }
    }
    for(int i = 0; i < numFeatures; i++)
        means[i] /= length[i];
    

    for(int ridx = 0; ridx < matchRes.row_no; ridx ++) {
        const auto &curRow = matchRes.rows[ridx];
        int _id = std::stoi(curRow[0]);
        int lid = std::stoi(curRow[1]);
        int rid = std::stoi(curRow[2]);
        fprintf(fvfile, "%d,%d,%d", _id, lid, rid);
        for(int j = 0; j < numFeatures; j++) {
            double val = featureValues[ridx][j];
            if(std::abs(std::abs(val) - std::abs(FeatureUtils::NaN)) < 1e-5)
                fprintf(fvfile, ",%.10lf", means[j]);
            else
                fprintf(fvfile, ",%.10lf", val);
        }
        fprintf(fvfile, "\n");
    }

    fclose(fvfile);

    gettimeofday(&end, NULL);
    double time = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
    std::cout << time << std::endl;

    CalculateFeature::index.releaseMemory();
    return;
}


extern "C"
{
    void extract_features_4_matching(int is_interchangeable, bool flag_consistent, int total_table, 
                                     const FeatureArguments *attrs, const char *default_fea_vec_dir, 
                                     const char *default_res_tab_name, const char *default_icv_dir, 
                                     const char *default_fea_names_dir) {
        FeatureEngineering::extractFeatures4Matching(is_interchangeable, flag_consistent, total_table, attrs, 
                                                     default_fea_vec_dir, default_res_tab_name, default_icv_dir, 
                                                     default_fea_names_dir);
    }

    void extract_features_4_topk(int is_interchangeable, bool flag_consistent, int total_table, 
                                 const FeatureArguments *attrs, const char *default_fea_vec_dir, 
                                 const char *default_icv_dir, const char *default_fea_names_dir) {
        FeatureEngineering::extractFeatures4TopK(is_interchangeable, flag_consistent, total_table, attrs, 
                                                 default_fea_vec_dir, default_icv_dir, default_fea_names_dir);
    }
}