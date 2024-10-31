// TODO: using FeatureIndex class
// the interchangeable value in blocking may be depracted
#ifndef _GROUP_H_
#define _GROUP_H_

#include "common/tokenizer.h"
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

using Groups = std::vector<std::unordered_map<int, std::vector<std::string>>>;
using GroupTokens = std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>>;
using Clusters = std::vector<std::unordered_map<std::string, int>>;

namespace feature_utils
{

// sims
inline int overlap(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    int ovlp = 0;

    std::vector<std::string> res;
    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    // __gnu_parallel::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    ovlp = res.size();

    return ovlp;
}

inline int tripletMin(int a, int b, int c)
{
    return (a <= b && a <= c) ? a : (b <= c ? b : c);
}

inline double levDist(const std::string &v1, const std::string &v2)
{
    ui v1_size = v1.size();
    ui v2_size = v2.size();

    if (!v1_size)
        return v2_size;
    else if (!v2_size)
        return v1_size;

    int dist[v1_size + 1][v2_size + 1];
    for (ui i = 0; i <= v1_size; i++)
        std::fill(dist[i], dist[i] + v2_size + 1, 0);

    for (ui i = 0; i <= v1_size; i++)
        dist[i][0] = int(i);
    for (ui i = 0; i <= v2_size; i++)
        dist[0][i] = int(i);

    // std::cout << v1 << v2 << std::endl;
    for (ui i = 1; i <= v1_size; i++)
    {
        for (ui j = 1; j <= v2_size; j++)
        {
            int cost = (v1[i - 1] == v2[j - 1]) ? 0 : 1;
            dist[i][j] = tripletMin(dist[i - 1][j] + 1,
                                    dist[i][j - 1] + 1,
                                    dist[i - 1][j - 1] + cost);
        }
    }

    return dist[v1_size][v2_size] * 1.0;
}

inline double jaccard(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp) * 1.0;
}

inline double cosine(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);
    double dot_product = v1.size() * v2.size() * 1.0;
    // double inv_denominator = SimFuncs::inverseSqrt(dot_product);

    // return ovlp * 1.0 * inv_denominator;
    return ovlp * 1.0 / sqrt(dot_product);
}

inline double dice(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);

    return ovlp * 2.0 / (int)(v1.size() + v2.size()) * 1.0;
}

inline double exactMatch(const std::string &s1, const std::string &s2)
{
    return s1 == s2 ? 1.0 : 0.0;
}

inline double absoluteNorm(const std::string &s1, const std::string &s2)
{
    if (s1 == " " || s2 == " " || s1.empty() || s2.empty())
        return -1.0;

    double d1 = stod(s1);
    double d2 = stod(s2);

    if (std::abs(d1) < 1e-5 || std::abs(d2) < 1e-5)
        return 0.0;

    double diff = std::abs(d1 - d2);
    double maxVal = std::max(std::abs(d1), std::abs(d2));

    if (diff / maxVal <= 1e-5)
        return 1.0;

    return 1.0 - diff / maxVal;
}

inline void stringSplit(std::string str, char delim, std::vector<std::string> &res) 
{
    std::istringstream iss(str);
    std::string token;
    while(getline(iss, token, delim))
        res.emplace_back(token);
};

inline void tokenize(const std::string &str, const std::string &type, std::vector<std::string> &tokens, 
              const std::string &delims) 
{
    tokens.clear();
    if(type == "dlm") Tokenizer::string2TokensDlm(str, tokens, delims);
    else if(type == "qgm") Tokenizer::string2TokensQGram(str, tokens, 3);
    std::sort(tokens.begin(), tokens.end());
    auto iter = std::unique(tokens.begin(), tokens.end());
    tokens.resize(std::distance(tokens.begin(), iter));
}

};


class GroupInterchangeable
{
public:
    int totalTable{0};
    int totalAttr{0};
    std::vector<std::string> attrVec;
    std::vector<int> featureLength;

    Groups group;
    Clusters cluster;
    GroupTokens groupTokensDlm, groupTokensQgm;
    std::vector<std::vector<std::vector<int>>> interCGroupsA, interCGroupsB;
    std::vector<std::vector<int>> interCGrpIdA, interCGrpIdB;
    std::vector<int> keyLength;

    // for each attr, cache all the values for each two groups with size >= 2
    double ****featureValCache{nullptr};
    int **discreteCacheIdx{nullptr};
    // for release only
    std::vector<int> attrCahceLength;

    std::vector<std::string> str_gt_10w = {"name", "title", "description"};
    std::vector<std::string> str_bt_1w_5w = {};
    std::vector<std::string> str_bt_5w_10w = {};
    std::vector<std::string> str_eq_1w = {"brand", "category"};

public:
    GroupInterchangeable() = default;
    GroupInterchangeable(int _totalTable, int _totalAttr,  const std::vector<std::string> &_attrVec)
    : totalTable(_totalTable), totalAttr(_totalAttr), attrVec(_attrVec) {
        printf("number of tables: %d\tnumber of attrs: %d\n", totalTable, totalAttr);
        for(const auto &attr : attrVec)
            printf("%s\t", attr.c_str());
        printf("\n");
    }
    ~GroupInterchangeable() = default;
    GroupInterchangeable(const GroupInterchangeable &other) = delete;
    GroupInterchangeable(GroupInterchangeable &&other) = delete;

public:
    // get number of features according to attr
    int calNumFeature(const std::string attr) {
        if(std::count(str_gt_10w.begin(), str_gt_10w.end(), attr) != 0 
        || std::count(str_bt_5w_10w.begin(), str_bt_5w_10w.end(), attr) != 0)
            return 4;
        else if(std::count(str_eq_1w.begin(), str_eq_1w.end(), attr) != 0)
            return 6;
        else if(std::count(str_bt_1w_5w.begin(), str_bt_1w_5w.end(), attr) != 0)
            return 8;

        return 0;
    }

    // get the column index for a specific feature
    int calCahceIndex(const std::string &func, const std::string &tok, int numFeature) {
        int idx = 0;

        if(func == "exm" || func == "lev") {
            idx = func == "lev" ? 4 : 5;
            return idx;
        }

        if(tok == "dlm") idx = 0;
        else if(tok == "qgm") idx = numFeature == 6 ? 0 : 1;

        if(func == "jac") idx = idx * 4 + 0;
        else if(func == "cos") idx = idx * 4 + 1;
        else if(func == "dice") idx = idx * 4 + 2;
        else if(func == "overlap") idx = idx * 4 + 3;

        return idx;
    }

    int calAttrIndex(const std::string &attr) {
        auto iter = std::find(attrVec.begin(), attrVec.end(), attr);
        return std::distance(attrVec.begin(), iter);
    }

    void globalInit(const std::vector<int> &keyNum, const std::vector<std::string> &attrs, Groups &groups, const GroupTokens &grpdlm, const GroupTokens &grpqgm) {
        featureValCache = new double***[totalAttr];
        discreteCacheIdx = new int*[totalAttr];

        for(int i = 0; i < totalAttr; i++) {
            const auto &curGrp = groups[i];
            const auto &curGrpDlm = grpdlm[i];
            const auto &curGrpQgm = grpqgm[i];
            int numFeature = calNumFeature(attrs[i]);
            if(numFeature == 0) {
                std::cerr << "no such attr: " << attrs[i] << std::endl;
                exit(1);
            }

            // length > 1
            std::vector<int> grpid;
            for(const auto &grpit : curGrp) {
                int keyId = grpit.first;
                if(grpit.second.size() > 1)
                    grpid.emplace_back(keyId);
            }

            int bucketSize = (int)grpid.size();
            // cache
            featureValCache[i] = new double**[numFeature];
            for(int j = 0; j < numFeature; j++) {
                featureValCache[i][j] = new double*[bucketSize];
                for(int l = 0; l < bucketSize; l++) {
                    featureValCache[i][j][l] = new double[numFeature];
                    std::fill(featureValCache[i][j][l], featureValCache[i][j][l] + numFeature, 0.0);
                }
            }
            // cache index
            discreteCacheIdx[i] = new int[keyNum[i]];
            std::fill(discreteCacheIdx[i], discreteCacheIdx[i] + keyNum[i], 0);
            for(int didx = 0; didx < bucketSize; didx++)
                discreteCacheIdx[i][grpid[didx]] = didx;
            
            attrCahceLength.emplace_back(bucketSize);

            // calculate
            for(int id1 = 0; id1 < bucketSize; id1++) {
                int lid = grpid[id1];
                const auto &ldocs = curGrp.at(lid);
                const auto &ldlms = curGrpDlm.at(lid);
                const auto &lqgms = curGrpQgm.at(lid);
                for(int id2 = id1 + 1; id2 < bucketSize; id2++) {
                    int rid = grpid[id2];
                    const auto &rdocs = curGrp.at(rid);
                    const auto &rdlms = curGrpDlm.at(rid);
                    const auto &rqgms = curGrpQgm.at(rid);

                    switch(numFeature) {
                        case 4 : {
                            double maxJacVal = 0.0;
                            double maxCosVal = 0.0;
                            double maxDiceVal = 0.0;
                            int maxOvlpVal = 0;
                            for(const auto &ldoc : ldlms) {
                                for(const auto &rdoc : rdlms) {
                                    maxJacVal = std::max(maxJacVal, feature_utils::jaccard(ldoc, rdoc));
                                    maxCosVal = std::max(maxCosVal, feature_utils::cosine(ldoc, rdoc));
                                    maxDiceVal = std::max(maxDiceVal, feature_utils::dice(ldoc, rdoc));
                                    maxOvlpVal = std::max(maxOvlpVal, feature_utils::overlap(ldoc, rdoc));
                                }
                            }
                            featureValCache[i][0][id1][id2] = maxJacVal;
                            featureValCache[i][0][id2][id1] = maxJacVal;
                            featureValCache[i][1][id1][id2] = maxCosVal;
                            featureValCache[i][1][id2][id1] = maxCosVal;
                            featureValCache[i][2][id1][id2] = maxDiceVal;
                            featureValCache[i][2][id2][id1] = maxDiceVal;
                            featureValCache[i][3][id1][id2] = maxOvlpVal * 1.0;
                            featureValCache[i][3][id2][id1] = maxOvlpVal * 1.0;
                            break;
                        }
                        case 6 : {
                            double maxJacVal = 0.0;
                            double maxCosVal = 0.0;
                            double maxDiceVal = 0.0;
                            int maxOvlpVal = 0;
                            double minLevVal = 0.0;
                            double maxExmVal = 0.0;
                            for(const auto &ldoc : lqgms) {
                                for(const auto &rdoc : rqgms) {
                                    maxJacVal = std::max(maxJacVal, feature_utils::jaccard(ldoc, rdoc));
                                    maxCosVal = std::max(maxCosVal, feature_utils::cosine(ldoc, rdoc));
                                    maxDiceVal = std::max(maxDiceVal, feature_utils::dice(ldoc, rdoc));
                                    maxOvlpVal = std::max(maxOvlpVal, feature_utils::overlap(ldoc, rdoc));
                                }
                            }
                            for(const auto &ldoc : ldocs) {
                                for(const auto &rdoc : rdocs) {
                                    minLevVal = std::min(minLevVal, feature_utils::levDist(ldoc, rdoc));
                                    maxExmVal = std::max(maxExmVal, feature_utils::exactMatch(ldoc, rdoc));
                                }
                            }
                            featureValCache[i][0][id1][id2] = maxJacVal;
                            featureValCache[i][0][id2][id1] = maxJacVal;
                            featureValCache[i][1][id1][id2] = maxCosVal;
                            featureValCache[i][1][id2][id1] = maxCosVal;
                            featureValCache[i][2][id1][id2] = maxDiceVal;
                            featureValCache[i][2][id2][id1] = maxDiceVal;
                            featureValCache[i][3][id1][id2] = maxOvlpVal * 1.0;
                            featureValCache[i][3][id2][id1] = maxOvlpVal * 1.0;
                            featureValCache[i][4][id1][id2] = minLevVal;
                            featureValCache[i][4][id2][id1] = minLevVal;
                            featureValCache[i][5][id1][id2] = maxExmVal;
                            featureValCache[i][5][id2][id1] = maxExmVal;
                            break;
                        }
                        case 8 : {
                            double maxJacDlmVal = 0.0, maxJacQgmVal = 0.0;
                            double maxCosDlmVal = 0.0, maxCosQgmVal = 0.0;
                            double maxDiceDlmVal = 0.0, maxDiceQgmVal = 0.0;
                            int maxOvlpDlmVal = 0, maxOvlpQgmVal = 0;
                            for(const auto &ldoc : ldlms) {
                                for(const auto &rdoc : rdlms) {
                                    maxJacDlmVal = std::max(maxJacDlmVal, feature_utils::jaccard(ldoc, rdoc));
                                    maxCosDlmVal = std::max(maxCosDlmVal, feature_utils::cosine(ldoc, rdoc));
                                    maxDiceDlmVal = std::max(maxDiceDlmVal, feature_utils::dice(ldoc, rdoc));
                                    maxOvlpDlmVal = std::max(maxOvlpDlmVal, feature_utils::overlap(ldoc, rdoc));
                                }
                            }
                            for(const auto &ldoc : lqgms) {
                                for(const auto &rdoc : rqgms) {
                                    maxJacQgmVal = std::max(maxJacQgmVal, feature_utils::jaccard(ldoc, rdoc));
                                    maxCosQgmVal = std::max(maxCosQgmVal, feature_utils::cosine(ldoc, rdoc));
                                    maxDiceQgmVal = std::max(maxDiceQgmVal, feature_utils::dice(ldoc, rdoc));
                                    maxOvlpQgmVal = std::max(maxOvlpQgmVal, feature_utils::overlap(ldoc, rdoc));
                                }
                            }
                            featureValCache[i][0][id1][id2] = maxJacDlmVal;
                            featureValCache[i][0][id2][id1] = maxJacDlmVal;
                            featureValCache[i][1][id1][id2] = maxCosDlmVal;
                            featureValCache[i][1][id2][id1] = maxCosDlmVal;
                            featureValCache[i][2][id1][id2] = maxDiceDlmVal;
                            featureValCache[i][2][id2][id1] = maxDiceDlmVal;
                            featureValCache[i][3][id1][id2] = maxOvlpDlmVal * 1.0;
                            featureValCache[i][3][id2][id1] = maxOvlpDlmVal * 1.0;
                            featureValCache[i][4][id1][id2] = maxJacQgmVal;
                            featureValCache[i][4][id2][id1] = maxJacQgmVal;
                            featureValCache[i][5][id1][id2] = maxCosQgmVal;
                            featureValCache[i][5][id2][id1] = maxCosQgmVal;
                            featureValCache[i][6][id1][id2] = maxDiceQgmVal;
                            featureValCache[i][6][id2][id1] = maxDiceQgmVal;
                            featureValCache[i][7][id1][id2] = maxOvlpQgmVal * 1.0;
                            featureValCache[i][7][id2][id1] = maxOvlpQgmVal * 1.0;
                            break;
                        }
                    }
                }
            }
        }
    }

    void IO() {
        std::string delims = " \"\',\\\t\r\n";

        for(int i = 0; i < totalAttr; i ++) {
            std::string grpPath = "buffer/interchangeable_grp_" + attrVec[i] + ".txt";
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
                feature_utils::stringSplit(entity, ' ', entityVec);
                int keyId = std::stoi(entityVec[0]);
                int length = std::stoi(entityVec[1]);

                if(length <= 1)
                    continue;
                
                for(int l = 0; l < length; l++) {
                    std::string doc;
                    getline(grpFile, doc);
                    // doc
                    curGrp[keyId].emplace_back(doc);
                    // tokenized doc
                    std::vector<std::string> tokensDlm;
                    std::vector<std::string> tokensQgm;
                    std::string tok1 = "dlm", tok2 = "qgm";
                    feature_utils::tokenize(doc, tok1, tokensDlm, delims);
                    feature_utils::tokenize(doc, tok2, tokensQgm, delims);
                    curGrpDlm[keyId].emplace_back(tokensDlm);
                    curGrpQgm[keyId].emplace_back(tokensQgm);
                    curClt[doc] = keyId;
                }
            }
            grpFile.close();
        }
    }

    void buildIndex(const Table &tableA, const Table &tableB) {
        interCGroupsA.resize(totalAttr);
        interCGroupsB.resize(totalAttr);
        interCGrpIdA.resize(totalAttr);
        interCGrpIdB.resize(totalAttr);
        
        for(int i = 0; i < totalAttr; i++) {
            std::string curAttr = attrVec[i];
            int attrPos = (int)tableA.inverted_schema.at(curAttr);
            const auto &curColA = tableA.cols[attrPos];
            const auto &curColB = tableB.cols[attrPos];
            int colASize = (int)curColA.size();
            int colBSize = (int)curColB.size();

            const auto &curClt = cluster[i];
            // const auto &curGrp = group[i];
            auto &curGrpA = interCGroupsA[i];
            auto &curGrpB = interCGroupsB[i];
            auto &curGrpIdA = interCGrpIdA[i];
            auto &curGrpIdB = interCGrpIdB[i];

            for(int row = 0; row < colASize; row++) {
                const auto &doc = curColA[row];
                bool haskey = curClt.find(doc) != curClt.end();
                
                if(haskey) {
                    int keyId = curClt.at(doc);
                    curGrpIdA.emplace_back(keyId);
                    curGrpA[keyId].emplace_back(row);
                }
                else
                    curGrpIdA.emplace_back(-1);
            }

            for(int row = 0; row < colBSize; row++) {
                const auto &doc = curColB[row];
                bool haskey = curClt.find(doc) != curClt.end();

                if(haskey) {
                    int keyId = curClt.at(doc);
                    curGrpIdB.emplace_back(keyId);
                    curGrpB[keyId].emplace_back(row);
                }
                else
                    curGrpB.emplace_back(-1);
            }
        }
    }

    void releaseBuffer() {
        int attrSize = (int)attrCahceLength.size();
        for(int i = 0; i < attrSize; i++) {
            int numFeature = calNumFeature(attrVec[i]);
            for(int j = 0; j < numFeature; j++) {
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
};





#endif // _GROUP_H_