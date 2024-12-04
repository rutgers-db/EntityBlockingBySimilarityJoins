/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
// main entrance for feature
#include "feature/feature.h"


int main(int argc, char *argv[])
{
    std::string usage = argv[1];
    int isInterchangeable = atoi(argv[2]);
    bool flagConsistent = atoi(argv[3]) == 1 ? true : false;
    int totalTable = atoi(argv[4]);
    FeatureArguments *attrs = new FeatureArguments;
    attrs->totalAttr = atoi(argv[5]);
    for(int i = 0; i < attrs->totalAttr; i++) {
        attrs->attributes[i] = new char[strlen(argv[6 + i])];
        strcpy(attrs->attributes[i], argv[6 + i]);
    }
    std::string defaultFeatureVecDir = argv[6 + attrs->totalAttr];
    std::string defaultResTableName = argv[7 + attrs->totalAttr];
    std::string defaultICVDir = argv[8 + attrs->totalAttr];
    std::string defaultFeatureNamesDir = argv[9 + attrs->totalAttr];
    
    if(usage == "match")
        // FeatureEngineering::extractFeatures4Matching(isInterchangeable, flagConsistent, totalTable, attrs, 
        //                                              defaultFeatureVecDir, defaultICVDir, defaultFeatureNamesDir);
        FeatureEngineering::extractFeatures4MatchingGraph(isInterchangeable, totalTable, attrs, defaultFeatureVecDir, 
                                                          defaultResTableName, defaultICVDir, defaultFeatureNamesDir);
    else if(usage == "topk")
        FeatureEngineering::extractFeatures4TopK(isInterchangeable, flagConsistent, totalTable, attrs, 
                                                 defaultFeatureVecDir, defaultICVDir, defaultFeatureNamesDir);
    else {
        std::cerr << "no such usage" << std::endl;
        exit(1);
    }

    delete attrs;

    return 0;
}