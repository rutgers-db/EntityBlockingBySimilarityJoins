/*
 * author: Dong Deng 
 * modified: Zhencan Peng in rutgers-db/RedPajama_Analysis
 * modified: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "common/simfunc.h"


int SimFuncs::overlap(const std::vector<ui> &v1, const std::vector<ui> &v2)
{
    int ovlp = 0;

#if OVLP_STRATEGY == 0
    auto it1 = v1.begin();
    auto it2 = v2.begin();

    while (it1 != v1.end() && it2 != v2.end())
    {
        if (*it1 == *it2)
        {
            ovlp++;
            ++it1, ++it2;
        }
        else
        {
            if (*it1 < *it2)
                ++it1;
            else
                ++it2;
        }
    }
#elif OVLP_STRATEGY == 1
    std::vector<ui> res;
    // std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    __gnu_parallel::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    ovlp = res.size();
#endif

    return ovlp;
}

double SimFuncs::weightedOverlap(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                 const std::vector<double> &wordwt)
{
    double ovlp = 0.0;

#if OVLP_STRATEGY == 0
    auto it1 = v1.begin();
    auto it2 = v2.begin();

    while(it1 != v1.end() && it2 != v2.end()) {
        if(*it1 == *it2) {
            ovlp += wordwt[*it1];
            ++ it1;
            ++ it2;
        }
        else {
            if(*it1 < *it2) ++ it1;
            else ++it2;
        }
    }
#elif OVLP_STRATEGY == 1
    std::vector<ui> res;
    // std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    __gnu_parallel::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    for(const auto &e : res)
        ovlp += wordwt[e];
#endif

    return ovlp;
}

// From: https://mrober.io/papers/rsqrt.pdf, page 49
double SimFuncs::inverseSqrt(double number)
{
    uint64_t i;
    double x2, y;
    x2 = number * 0.5;
    y = number;
    i = *(uint64_t *)&y;
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    y = *(double *)&i;
    y = y * (1.5 - (x2 * y * y));
    return y;
}

int SimFuncs::levDist(const std::string &v1, const std::string &v2)
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
            dist[i][j] = SimFuncs::tripletMin(dist[i - 1][j] + 1,
                                              dist[i][j - 1] + 1,
                                              dist[i - 1][j - 1] + cost);
        }
    }

    return dist[v1_size][v2_size];
}

int SimFuncs::tripletMin(int a, int b, int c)
{
    return (a <= b && a <= c) ? a : (b <= c ? b : c);
}

double SimFuncs::jaccard(const std::vector<ui> &v1, const std::vector<ui> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = SimFuncs::overlap(v1, v2);

    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp) * 1.0;
}

double SimFuncs::jaccard(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp) * 1.0;
}

double SimFuncs::weightedJaccard(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                 const std::vector<double> &wordwt,
                                 double v1rw, double v2rw)
{
    if (ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    double ovlp = SimFuncs::weightedOverlap(v1, v2, wordwt);

    return ovlp / (v1rw + v2rw - ovlp);
}

double SimFuncs::weightedJaccard(double v1rw, double v2rw, double ovlp)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    return ovlp / (v1rw + v2rw - ovlp);
}

double SimFuncs::cosine(const std::vector<ui> &v1, const std::vector<ui> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = SimFuncs::overlap(v1, v2);
    double dot_product = v1.size() * v2.size() * 1.0;
    // double inv_denominator = SimFuncs::inverseSqrt(dot_product);

    // return ovlp * 1.0 * inv_denominator;
    return ovlp * 1.0 / sqrt(dot_product);
}

double SimFuncs::cosine(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    return ovlp * 1.0 / sqrt(v1.size() * v2.size() * 1.0);
}

double SimFuncs::weightedCosine(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                const std::vector<double> &wordwt,
                                double v1rw, double v2rw)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    double ovlp = SimFuncs::weightedOverlap(v1, v2, wordwt);
    double dot_product = v1rw * v2rw * 1.0;
    // double inv_denominator = SimFuncs::inverseSqrt(dot_product);

    // return ovlp * 1.0 * inv_denominator;
    return ovlp * 1.0 / sqrt(dot_product);
}

double SimFuncs::weightedCosine(double v1rw, double v2rw, double ovlp)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    return ovlp / sqrt(v1rw * v2rw);
}

double SimFuncs::dice(const std::vector<ui> &v1, const std::vector<ui> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = SimFuncs::overlap(v1, v2);

    return ovlp * 2.0 / (int)(v1.size() + v2.size()) * 1.0;
}

double SimFuncs::dice(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    return ovlp * 2.0 / (int)(v1.size() + v2.size()) * 1.0;
}

double SimFuncs::weightedDice(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                              const std::vector<double> &wordwt,
                              double v1rw, double v2rw)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    double ovlp = SimFuncs::weightedOverlap(v1, v2, wordwt);

    return ovlp * 2.0 / (v1rw + v2rw);
}

double SimFuncs::weightedDice(double v1rw, double v2rw, double ovlp)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    return ovlp * 2.0 / (v1rw + v2rw);
}

double SimFuncs::overlapCoeff(const std::vector<ui> &v1, const std::vector<ui> &v2)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    int ovlp = SimFuncs::overlap(v1, v2);

    return ovlp * 1.0 / std::min(v1.size(), v2.size()) * 1.0;
}

double SimFuncs::overlapCoeff(const std::vector<ui> &v1, const std::vector<ui> &v2, int ovlp)
{
    if (v1.empty() && v2.empty())
        return 1.0;

    return ovlp * 1.0 / std::min(v1.size(), v2.size()) * 1.0;
}

double SimFuncs::weightedOverlapCoeff(const std::vector<ui> &v1, const std::vector<ui> &v2, 
                                      const std::vector<double> &wordwt,
                                      double v1rw, double v2rw)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    double ovlp = SimFuncs::weightedOverlap(v1, v2, wordwt);

    return ovlp / std::min(v1rw, v2rw);
}

double SimFuncs::weightedOverlapCoeff(double v1rw, double v2rw, double ovlp)
{
    if(ISZERO(v1rw) && ISZERO(v2rw))
        return 1.0;

    return ovlp / std::min(v1rw, v2rw);
}

double SimFuncs::levSim(const std::string &v1, const std::string &v2)
{
    ui lev_dist = SimFuncs::levDist(v1, v2);

    return 1 - (int(lev_dist) * 1.0 / std::max(v1.size(), v2.size()) * 1.0);
}

bool SimFuncs::exactMatch(const std::string &s1, const std::string &s2)
{
    return s1 == s2;
}

double SimFuncs::absoluteNorm(const std::string &s1, const std::string &s2)
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


double SimFuncs::jaroWinkler(const std::string &s1, const std::string &s2)
{
    float m = 0;
    int low, high, range;
    int k = 0, numTrans = 0;

    // Exit early if either are empty
    if (s1.empty() || s2.empty())
        return 0.0;
    // Exit early if they're an exact match.
    if (s1 == s2)
        return 1.0;

    range = (std::max(s1.length(), s2.length()) / 2) - 1;
    int s1Matches[s1.length()] = {};
    int s2Matches[s2.length()] = {};
    int len1 = s1.length();
    int len2 = s2.length();

    for (int i = 0; i < len1; i++) {
        // Low & High;
        low = i >= range ? i - range : 0;
        high = i + range <= len2 - 1 ? i + range : len2 - 1;

        for (int j = low; j <= high; j++) {
            if (s1Matches[i] != 1 && s2Matches[j] != 1 && s1[i] == s2[j]) {
                ++ m;
                s1Matches[i] = 1;
                s2Matches[j] = 1;
                break;
            }
        }
    }

    // Exit early if no matches were found
    if (m == 0)
        return 0.0;

    // Count the transpositions.
    for (int i = 0; i < len1; i++) {
        if (s1Matches[i] == 1) {
            int j;
            for (j = k; j < len2; j++) {
                if (s2Matches[j] == 1) {
                    k = j + 1;
                    break;
                }
            }

            if (s1[i] != s2[j]) {
                numTrans += 1;
            }
        }
    }

    double weight = (m / len1 + m / len2 + (m - (numTrans / 2)) / m) / 3;
    double l = 0;
    double p = 0.1;
    if (weight > 0.7) {
        while (s1[l] == s2[l] && l < 4)
            l += 1;

        weight += l * p * (1 - weight);
    }

    return weight;
}


double SimFuncs::mongeElkan(const std::string &s1, const std::string &s2)
{
    // split
    const char split = ' ';
    std::istringstream iss1(s1);
    std::istringstream iss2(s2);
    std::string token1, token2;
    std::vector<std::string> res1, res2;
    while(getline(iss1, token1, split)) {
        if(token1.empty() || token1 == " ")
            continue;
        res1.emplace_back(token1);
    }
    while(getline(iss2, token2, split)) {
        if(token2.empty() || token2 == " ")
            continue;
        res2.emplace_back(token2);
    }

    // using jaro-winkler as inner functions according to Falcon
    // could also use levstein-distance
    double cummax = 0.0;
    for(const auto &tok : res1) {
        double maxscore = 0.0;
        for(const auto &tok2 : res2)
            maxscore = std::max(maxscore, jaroWinkler(tok, tok2));
        cummax += maxscore;
    }

    return cummax / (double)res1.size() * 1.0;
}


template<typename T>
int SimFuncsTemplate::overlap(const std::vector<T> &v1, const std::vector<T> &v2)
{
    std::vector<T> res;
    __gnu_parallel::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), 
                                     std::back_inserter(res));
    return (int)res.size();
}


template<typename T>
double SimFuncsTemplate::jaccard(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if(v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp);
}


template<typename T>
double SimFuncsTemplate::jaccard(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp)
{
    if(v1.empty() && v2.empty())
        return 1.0;
    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp);
}


template<typename T>
double SimFuncsTemplate::cosine(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if(v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / sqrt(v1.size() * v2.size());
}


template<typename T>
double SimFuncsTemplate::cosine(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp)
{
    if(v1.empty() && v2.empty())
        return 1.0;
    return ovlp * 1.0 / sqrt(v1.size() * v2.size());
}


template<typename T>
double SimFuncsTemplate::dice(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if(v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / (2 * (v1.size() + v2.size()));
}


template<typename T>
double SimFuncsTemplate::dice(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp)
{
     if(v1.empty() && v2.empty())
        return 1.0;
    return ovlp * 2.0 / (v1.size() + v2.size());
}


template<typename T>
double SimFuncsTemplate::overlapCoeff(const std::vector<T> &v1, const std::vector<T> &v2)
{
    if(v1.empty() && v2.empty())
        return 1.0;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / std::min(v1.size(), v2.size());
}


template<typename T>
double SimFuncsTemplate::overlapCoeff(const std::vector<T> &v1, const std::vector<T> &v2, int ovlp)
{
    if(v1.empty() && v2.empty())
        return 1.0;
    return ovlp * 1.0 / std::min(v1.size(), v2.size());
}
