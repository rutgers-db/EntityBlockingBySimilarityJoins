/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _DOWN_SAMPLER_H_
#define _DOWN_SAMPLER_H_

#include "sampler/base_sampler.h"
#include <random>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <sys/time.h>


class DownSampler : public Sampler
{
public:
    using Sampler::Sampler;
    ui n{0};                                                            // |S|
    ui y{0};                                                            // parameter y according to falcon-sigmod2017
    std::unordered_map<ui, std::vector<ui>> tokenIndex;                 // inverted index I
    std::vector<std::pair<int, int>> samplePairs[MAXTHREADNUM];

public:
    DownSampler() = default;
    DownSampler(ui _n, ui _y, std::string _blkAttr, bool _isRS) : Sampler(_blkAttr, _isRS), n(_n), y(_y) {
        std::cout << "spawn down sampler: " << blkAttr << " n: " << n << " y: " << y << " is RS Join: " << isRS << std::endl;
    }
    ~DownSampler() = default;
    DownSampler(const DownSampler &other) = delete;
    DownSampler(DownSampler &&other) = delete;

public:
    void sample(const std::string &pathTableA, const std::string &pathTableB);
};


#endif // _DOWN_SAMPLER_H_