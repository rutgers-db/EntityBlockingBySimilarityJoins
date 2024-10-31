/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _SIM_FUNC_H_
#define _SIM_FUNC_H_

#include "type.h"
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>
#include <sstream>
#include <sys/time.h>
#include <parallel/algorithm>

#define ISZERO(X) std::abs(X) <= 1e-5


/*
 * A set of similarity functions
 */
class SimFuncs
{
public:
    SimFuncs() = default;
    ~SimFuncs() = default;
    SimFuncs(const SimFuncs &other) = delete;
    SimFuncs(SimFuncs &&other) = delete;

public:
    // helpers
    static int overlap(const std::vector<ui> &v1, const std::vector<ui> &v2);
    static double weightedOverlap(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                  const std::vector<double> &wordwt);

    static int levDist(const std::string &v1, const std::string &v2);
    static double inverseSqrt(double number);
    static int tripletMin(int a, int b, int c);

    // four main funcs
    // different from the classical weighted func
    // wordwt: word weights
    // v1rw, v2rw: weight for each record, which is the sum of its words weights

    // jaccard : (|A| \cap |B|) / (|A| + |B| - |A| \cap |B|)
    static double jaccard(const std::vector<ui> &v1, const std::vector<ui> &v2);
    static double jaccard(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp);
    static double weightedJaccard(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                  const std::vector<double> &wordwt,
                                  double v1rw, double v2rw);
    static double weightedJaccard(double v1rw, double v2rw, double ovlp);

    // cosine: (|A| \cap |B|) / sqrt(|A| * |B|)
    static double cosine(const std::vector<ui> &v1, const std::vector<ui> &v2);
    static double cosine(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp);
    static double weightedCosine(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                 const std::vector<double> &wordwt,
                                 double v1rw, double v2rw);
    static double weightedCosine(double v1rw, double v2rw, double ovlp);
    
    // dice: 2 * (|A| \cap |B|) / (|A| + |B|)
    static double dice(const std::vector<ui> &v1, const std::vector<ui> &v2);
    static double dice(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp);
    static double weightedDice(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                               const std::vector<double> &wordwt,
                               double v1rw, double v2rw);
    static double weightedDice(double v1rw, double v2rw, double ovlp);

    // overlap coefficient: (|A| \cap |B|) / min(|A|, |B|)
    static double overlapCoeff(const std::vector<ui> &v1, const std::vector<ui> &v2);
    static double overlapCoeff(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp);
    static double weightedOverlapCoeff(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                       const std::vector<double> &wordwt,
                                       double v1rw, double v2rw);
    static double weightedOverlapCoeff(double v1rw, double v2rw, double ovlp);

    // other funcs based on string without tokenization
    // these can not be applied on long strings: str_bt_1w_5w to str_gt_10w
    // since most of funcs(joins) time will increase sharply with length
    static double levSim(const std::string &v1, const std::string &v2);
    static bool exactMatch(const std::string &s1, const std::string &s2);
    static double absoluteNorm(const std::string &s1, const std::string &s2);
    // following not used in blocking
    // code: https://github.com/sanket143/Jaro-Winkler/blob/master/cpp/jwdistance.h
    static double jaroWinkler(const std::string &s1, const std::string &s2);
    static double mongeElkan(const std::string &s1, const std::string &s2);
};


/*
 * The records are vectors of unsigned int for most of the time
 * but in some parts, like feature extraction, the four set similarity functions
 * may accept the records that are vectors of strings
 * thus, we only overload these four sim funcs as template, which will not affect
 * for other parts
 */
class SimFuncsTemplate
{
public:
    SimFuncsTemplate() = default;
    ~SimFuncsTemplate() = default;
    SimFuncsTemplate(const SimFuncsTemplate &other) = delete;
    SimFuncsTemplate(SimFuncsTemplate &&other) = delete;

public:
    template<typename T>
    static int overlap(const std::vector<T> &v1, const std::vector<T> &v2);

    template<typename T>
    static double jaccard(const std::vector<T> &v1, const std::vector<T> &v2);
    template<typename T>
    static double jaccard(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp);

    template<typename T>
    static double cosine(const std::vector<T> &v1, const std::vector<T> &v2);
    template<typename T>
    static double cosine(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp);

    template<typename T>
    static double dice(const std::vector<T> &v1, const std::vector<T> &v2);
    template<typename T>
    static double dice(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp);

    template<typename T>
    static double overlapCoeff(const std::vector<T> &v1, const std::vector<T> &v2);
    template<typename T>
    static double overlapCoeff(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp);
};


// TODO:
class TFIDF
{

};


#endif // _SIM_FUNC_H_