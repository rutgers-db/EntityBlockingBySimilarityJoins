/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _FEATURE_H_
#define _FEATURE_H_

#include "common/io.h"
#include "feature/feature_utils.h"
#include "feature/feature_index.h"
#include "feature/cal_feature.h"
#include <fstream>
#include <sys/time.h>
#include <omp.h>

// arguments for feature api
struct FeatureArguments
{
    int totalAttr;
    char *attributes[20];

    FeatureArguments() = default;
    ~FeatureArguments() {
        for(int i = 0; i < 20; i++)
            delete[] attributes[i];
    }
};


class FeatureEngineering
{
public:
    FeatureEngineering() = default;
    ~FeatureEngineering() = default;
    FeatureEngineering(const FeatureEngineering &other) = delete;
    FeatureEngineering(FeatureEngineering &&other) = delete;

public:
    static void readGroups(int totalAttr, const std::vector<std::string> &attrVec, FeatureIndex::Groups &group, 
                           FeatureIndex::GroupTokens &groupTokensDlm, FeatureIndex::GroupTokens &groupTokensQgm, 
                           FeatureIndex::Cluster &cluster, std::vector<int> &keyLength, 
                           const std::string &defaultICVDir = "");

    static void readFeatures(ui &numFeatures, Rule *&featureNames, std::vector<std::string> &nameCopy, 
                             const std::string &defaultFeatureNamesDir = "");
    /*
     * extract features with interchangeable values
     * this is used for matching with updated features
     */
    static void extractFeatures4Matching(int isInterchangeable, bool flagConsistent, int totalTable,
                                         const FeatureArguments *attrs, const std::string &defaultFeatureVecDir = "", 
                                         const std::string &defaultResTableName = "", const std::string &defaultICVDir = "", 
                                         const std::string &defeaultFeatureNamesDir = "");
    /*
     * extract features for top k
     * will use interchangeable values depending on input
     */
    static void extractFeatures4TopK(int isInterchangeable, bool flagConsistent, int totalTable, 
                                     const FeatureArguments *attrs, const std::string &defaultFeatureVecDir = "", 
                                     const std::string &defaultICVDir = "", const std::string &defaultFeatureNamesDir = "");
};


#endif // _FEATURE_H_