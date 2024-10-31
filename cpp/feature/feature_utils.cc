/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "feature/feature_utils.h"

std::string FeatureUtils::delims = " \"\',\\\t\r\n";
double FeatureUtils::NaN = -19260817.0;


int FeatureUtils::overlap(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if(v1.empty() || v2.empty())
        return (int)NaN;

    std::vector<std::string> res;
    
    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    // __gnu_parallel::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));

    return (int)res.size();
}


double FeatureUtils::overlapD(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if(v1.empty() || v2.empty())
        return (int)NaN;

    std::vector<std::string> res;
    
    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));
    // __gnu_parallel::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(res));

    return (int)res.size();
}


int FeatureUtils::tripletMin(int a, int b, int c)
{
    return (a <= b && a <= c) ? a : (b <= c ? b : c);
}


double FeatureUtils::levDist(const std::string &v1, const std::string &v2)
{
    ui v1_size = v1.size();
    ui v2_size = v2.size();

    if(!v1_size || !v2_size)
        return NaN;

    int dist[v1_size + 1][v2_size + 1];
    for (ui i = 0; i <= v1_size; i++)
        std::fill(dist[i], dist[i] + v2_size + 1, 0);

    for (ui i = 0; i <= v1_size; i++)
        dist[i][0] = int(i);
    for (ui i = 0; i <= v2_size; i++)
        dist[0][i] = int(i);

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


double FeatureUtils::jaccard(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if(v1.empty() || v2.empty())
        return NaN;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp);
}


double FeatureUtils::jaccard(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp)
{
    if(v1.empty() || v2.empty())
        return NaN;

    return ovlp * 1.0 / (v1.size() + v2.size() - ovlp);
}


double FeatureUtils::cosine(const std::vector<std::string> &v1, const std::vector<std::string> &v2) 
{
    if(v1.empty() || v2.empty())
        return NaN;

    int ovlp = overlap(v1, v2);

    double dot_product = v1.size() * v2.size() * 1.0;
    return ovlp * 1.0 / sqrt(dot_product);
}


double FeatureUtils::cosine(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp)
{
    if(v1.empty() || v2.empty())
        return NaN;

    double dot_product = v1.size() * v2.size() * 1.0;
    return ovlp * 1.0 / sqrt(dot_product);
}


double FeatureUtils::dice(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if(v1.empty() || v2.empty())
        return NaN;

    int ovlp = overlap(v1, v2);

    return ovlp * 2.0 / (int)(v1.size() + v2.size());
}


double FeatureUtils::dice(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp)
{
    if(v1.empty() || v2.empty())
        return NaN;

    return ovlp * 2.0 / (int)(v1.size() + v2.size());
}


double FeatureUtils::overlapCoeff(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
    if(v1.empty() || v2.empty())
        return NaN;

    int ovlp = overlap(v1, v2);

    return ovlp * 1.0 / std::min(v1.size(), v2.size());
}


double FeatureUtils::overlapCoeff(const std::vector<std::string> &v1, const std::vector<std::string> &v2, int ovlp)
{
    if(v1.empty() || v2.empty())
        return NaN;

    return ovlp * 1.0 / std::min(v1.size(), v2.size());
}


double FeatureUtils::exactMatch(const std::string &s1, const std::string &s2)
{
    if(!s1.length() || !s2.length())
        return NaN;

    return s1 == s2 ? 1.0 : 0.0;
}


double FeatureUtils::absoluteNorm(const std::string &s1, const std::string &s2)
{
    if (s1 == " " || s2 == " " || s1.empty() || s2.empty())
        return NaN;

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


void FeatureUtils::stringSplit(std::string str, char delim, std::vector<std::string> &res) 
{
    std::istringstream iss(str);
    std::string token;
    while(getline(iss, token, delim))
        res.emplace_back(token);
};


void FeatureUtils::tokenize(const std::string &str, TokenizerType type, std::vector<std::string> &tokens) 
{
    tokens.clear();

    switch(type) {
        case TokenizerType::Dlm : 
            Tokenizer::string2TokensDlm(str, tokens, delims); 
            break;
        case TokenizerType::QGram : 
            Tokenizer::string2TokensQGram(str, tokens, 3); 
            break;
        case TokenizerType::WSpace :
            Tokenizer::string2TokensWSpace(str, tokens);
            break;
        case TokenizerType::AlphaNumeric : 
            Tokenizer::string2TokensAlphaNumeric(str, tokens);
            break;
    }
    
    // sort and unique
    std::sort(tokens.begin(), tokens.end());
    auto iter = std::unique(tokens.begin(), tokens.end());
    tokens.resize(std::distance(tokens.begin(), iter));
}