/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
/*
 * this file can only be included in the feature.cpp & related .hpp files
 * this file contains the methods for accelerating extracting features 
 * from interchangeable values (e.g. sim joins/searchs)
 * this is the version for vector<string> rather than vector<ui>
 */
#ifndef _FEATURE_INDEX_H_
#define _FEATURE_INDEX_H_

#include "feature/feature_utils.h"
#include <string>
#include <vector>


class FeatureIndex
{
public:
    // tokens are stored as string or (unsigned) integer
    // the integer tokens are for similarity join to boost index init
    // e.g., set a threshold 0.8 to quickly filter pairs in two interchangeable groups
    using Group = std::unordered_map<int, std::vector<std::string>>;
    using Groups = std::vector<std::unordered_map<int, std::vector<std::string>>>;
    using GroupToken = std::unordered_map<int, std::vector<std::vector<std::string>>>;
    using GroupTokens = std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>>;
    using GroupTokenInt = std::unordered_map<int, std::vector<std::vector<ui>>>;
    using GroupTokensInt = std::vector<std::unordered_map<int, std::vector<std::vector<ui>>>>;
    using Cluster = std::vector<std::unordered_map<std::string, int>>;

public:
    std::vector<std::string> str_gt_10w = {"name", "title", "description"};
    std::vector<std::string> str_bt_1w_5w = {};
    std::vector<std::string> str_bt_5w_10w = {};
    std::vector<std::string> str_eq_1w = {"brand", "category", "manufacturer"};

    // for each attr, cache all the values for each two groups with size larger than MIN_CACHED_LENGTH
    double ****featureValCache = nullptr;
    int **discreteCacheIdx = nullptr;
    // for releasing buffers only
    std::vector<int> attrCahceLength;

    // min length of the group to be cached
    const int MIN_CACHED_LENGTH = 10000;
    // det used for length filter
    const double LENGTH_FILTER_DET = 0.1;
    // length filter
    const double universalDet = LENGTH_FILTER_DET;
    const double cosLengthFilter = universalDet * universalDet;
    const double diceLengthFilter = universalDet / (2 - universalDet);

public:
    FeatureIndex() = default;
    ~FeatureIndex() {
        int attrSize = (int)attrCahceLength.size();
        for(int i = 0; i < attrSize; i++) {
            for(int j = 0; j < attrCahceLength[i]; j++) {
                for(int k = 0; k < attrCahceLength[i]; k++)
                    delete[] featureValCache[i][j][k];
                delete[] featureValCache[i][j];
            }
            delete[] featureValCache[i];
            delete[] discreteCacheIdx[i];
        }
        delete[] featureValCache;
        delete[] discreteCacheIdx;
    }
    FeatureIndex(const FeatureIndex &other) = delete;
    FeatureIndex(FeatureIndex &&other) = delete;

public:
    // get the column index for a specific feature
    int calCahceIndex(const std::string &func, const std::string &tok, int numFeature);

private:
    // sort according to idf
    void normalizeTokens(const GroupTokens &grpToks, GroupTokensInt &grpToksInt);

    void calIndexLength4(int curAttrIdx, const std::vector<int> &grpid, const GroupToken &curGrpDlm, const GroupTokenInt &curGrpDlmInt, 
                         bool isCoeff);
    void calIndexLength6(int curAttrIdx, const std::vector<int> &grpid, const Group &curGrp, const GroupToken &curGrpQgm, 
                         const GroupTokenInt &curGrpQgmInt, bool isCoeff);
    void calIndexLength8(int curAttrIdx, const std::vector<int> &grpid, const GroupToken &curGrpDlm, const GroupTokenInt &curGrpDlmInt, 
                         const GroupToken &curGrpQgm, const GroupTokenInt &curGrpQgmInt, bool isCoeff);
    
public:
    // get number of features according to attr
    int calNumFeature(const std::string attr);

    void globalInit(const std::vector<int> &keyNum, const std::vector<std::string> &attrs, Groups &groups, const GroupTokens &grpdlm, 
                    const GroupTokens &grpqgm, bool isCoeff = false);
};


#endif // _FEATURE_INDEX_H_