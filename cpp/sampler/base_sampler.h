/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _BASE_SAMPLER_H_
#define _BASE_SAMPLER_H_

#include "common/tokenizer.h"
#include "common/dataframe.h"
#include "sampler/sampler_util.h"
#include <vector>


class Sampler
{
public:
    std::string blkAttr;
    bool isRS{false};
    ui numWord{0};
    std::string pathTableA;
    std::string pathTableB;
    Table tableA;
    Table tableB;
    std::vector<std::vector<ui>> recordsA;
    std::vector<std::vector<ui>> recordsB;
    std::vector<double> weightsA;
    std::vector<double> weightsB;
    std::vector<double> wordwt;
    std::vector<ui> idMapA;
    std::vector<ui> idMapB;
    std::vector<std::pair<int, int>> pairs;

public:
    Sampler() = default;
    Sampler(std::string _blkAttr, bool _isRS) : blkAttr(_blkAttr), isRS(_isRS) { }
    ~Sampler() = default;
    Sampler(const Sampler &other) = delete;
    Sampler(Sampler &&other) = delete;

public:
    void readTable(const std::string &_pathTableA, const std::string &_pathTableB);
    void prepareRecords(ui columnA, ui columnB, TokenizerType tt, ui q);
    // virtual api
    virtual void sample(const std::string &pathTableA, const std::string &pathTableB) = 0;
};


#endif // _BASE_SAMPLER_H_