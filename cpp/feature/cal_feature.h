/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
/*
 * this file can only be included in the feature.cpp & related .hpp files
 * this file contains the calculation of feature values
 */
#ifndef _CAL_FEATURE_H_
#define _CAL_FEATURE_H_

#include "group/graph.h"
#include "feature/feature_index.h"
#include "feature/feature_utils.h"


class CalculateFeature
{
public:
    // there are two versions of set join funcs in feature_utils.hpp
    // the version with 2 arguments are used in feature.cpp
    // thus the pointer only capture this version
    typedef double (*SetJoinFunc)(const std::vector<std::string> &, const std::vector<std::string> &);
    typedef double (*StringJoinFunc)(const std::string &, const std::string &);

public:
    static FeatureIndex index;

public:
    CalculateFeature() = default;
    ~CalculateFeature() = default;
    CalculateFeature(const CalculateFeature &other) = delete;
    CalculateFeature(CalculateFeature &&other) = delete;

private:
    // calculate original features (i.e., without interchangeable values)
    static void calOriginalFeatures(std::vector<std::vector<double>> &featureValues, const std::string &func, 
                                    const std::string &lstr, const std::string &rstr, 
                                    const std::vector<std::string> &ltokens, 
                                    const std::vector<std::string> &rtokens, 
                                    bool isCoeff = false);

    // calculate features with on side has interchangeable values
    // tokens, cltid: the entity does not have interchangeable values
    // ictokens, iccltid: the entity has interchangeable values
    static void calOneSideFeatures(std::vector<std::vector<double>> &featureValues, SetJoinFunc setJoinP, const std::string &tok, 
                                   const std::vector<std::string> &tokens, const std::vector<std::string> &ictokens, 
                                   const FeatureIndex::GroupToken &curGrpDlm, const FeatureIndex::GroupToken &curGrpQgm, 
                                   int cltid, int iccltid);

    static void calOneSideFeatures(std::vector<std::vector<double>> &featureValues, SetJoinFunc setJoinP, const std::string &tok, 
                                   const std::string &str, const std::vector<std::string> &tokens, 
                                   const std::vector<std::string> &ictokens, 
                                   const Graph &semanticGraph);

    static void calOneSideFeatures(std::vector<std::vector<double>> &featureValues, StringJoinFunc stringJoinP, const std::string &tok, 
                                   const std::string &str, const std::string &icstr, const FeatureIndex::Group &curGrp,
                                   int cltid, int iccltid, const std::string &func);

    static void calOneSideFeatures(std::vector<std::vector<double>> &featureValues, StringJoinFunc stringJoinP, const std::string &tok, 
                                   const std::string &str, const std::string &icstr, const Graph &semanticGraph, const std::string &func);

    // double side
    static void calDoubleSideFeatures(std::vector<std::vector<double>> &featureValues, SetJoinFunc setjoinP, const std::string &tok, 
                                      const std::vector<std::string> &ltokens, const std::vector<std::string> &rtokens, 
                                      const FeatureIndex::GroupToken &curGrpDlm, const FeatureIndex::GroupToken &curGrpQgm, 
                                      int lcltid, int rcltid, int *const &curDCIdx, double ***const &curCache, 
                                      const std::vector<int> &featureLength, const std::string &func, ui attrpos);

    static void calDoubleSideFeatures(std::vector<std::vector<double>> &featureValues, SetJoinFunc setJoinP, const std::string &tok, 
                                      const std::string &lstr, const std::string &rstr, const std::vector<std::string> &ltokens,
                                      const std::vector<std::string> &rtokens, const Graph &semanticGraph, const std::string &func);

    static void calDoubleSideFeatures(std::vector<std::vector<double>> &featureValues, StringJoinFunc stringJoinP, const std::string &tok, 
                                      const std::string &lstr, const std::string &rstr, const FeatureIndex::Group &curGrp, 
                                      int lcltid, int rcltid, int *const &curDCIdx, double ***const &curCache, 
                                      const std::vector<int> &featureLength, const std::string &func, ui attrpos);

    static void calDoubleSideFeatures(std::vector<std::vector<double>> &featureValues, StringJoinFunc stringJoinP, const std::string &tok, 
                                      const std::string &lstr, const std::string &rstr, const Graph &semanticGraph, 
                                      const std::string &func);

public:
    // isTopK indicates whether calculating features is used for top K on matchRes
    // if true, then the value needed to be normalized, that is, use the overlap coeff
    static void calAll(int numFeatures, Rule *featureNames, const std::vector<std::string> &attrVec, const Table &resTable, 
                       std::vector<std::vector<double>> &featureValues, const FeatureIndex::Groups &group, 
                       const FeatureIndex::GroupTokens &groupTokensDlm, const FeatureIndex::GroupTokens &groupTokensQgm, 
                       const FeatureIndex::Cluster &cluster, const std::vector<int> &featureLength, bool flagConsistent, 
                       bool isTopK);
    
    static void calAllWithoutInterchangeable(int numFeatures, Rule *featureNames, const std::vector<std::string> &attrVec, const Table &resTable, 
                                             std::vector<std::vector<double>> &featureValues, const std::vector<int> &featureLength, bool isTopK);
};


#endif // _CAL_FEATURE_H_