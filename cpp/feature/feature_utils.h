/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
/*
 * this file can only be included in the feature.cpp & related .hpp files
 * functions are not inlined
 * this file contains the overloading sim funcs of SimFunc.cpp
 * the version for vector<string> rather than vector<ui>
 */
#ifndef _FEATURE_UTILS_H_
#define _FEATURE_UTILS_H_

#include "common/type.h"
#include "common/tokenizer.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>


class FeatureUtils
{
public:
    static std::string delims;
    static double NaN;

public:
    FeatureUtils() = default;
    ~FeatureUtils() = default;
    FeatureUtils(const FeatureUtils &other) = delete;
    FeatureUtils(FeatureUtils &&other) = delete;

public:
    /*
     * the implementation is according to the py_entitymatching package
     * for sim funcs, if one of the input records is empty
     * the funcs will return NaN which is a big negative value
     * this is different from simfunc.cc
     */
    static int overlap(const std::vector<std::string> &v1, const std::vector<std::string> &v2);
    // this should only be used in feature value calculation in feature.cpp
    // captured by SetJoinFunc
    static double overlapD(const std::vector<std::string> &v1, const std::vector<std::string> &v2);

    static int tripletMin(int a, int b, int c);
    static double levDist(const std::string &v1, const std::string &v2);

    static double jaccard(const std::vector<std::string> &v1, const std::vector<std::string> &v2);
    static double jaccard(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp);

    static double cosine(const std::vector<std::string> &v1, const std::vector<std::string> &v2);
    static double cosine(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp);

    static double dice(const std::vector<std::string> &v1, const std::vector<std::string> &v2);
    static double dice(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp);

    static double overlapCoeff(const std::vector<std::string> &v1, const std::vector<std::string> &v2);
    static double overlapCoeff(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp);

    static double exactMatch(const std::string &s1, const std::string &s2);

    static double absoluteNorm(const std::string &s1, const std::string &s2);

    static void stringSplit(std::string str, char delim, std::vector<std::string> &res);

    static void tokenize(const std::string &str, TokenizerType type, std::vector<std::string> &tokens);
};


#endif // _FEATURE_UTILS_H_