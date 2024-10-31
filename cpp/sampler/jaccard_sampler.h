/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _JACCARD_SAMPLER_H_
#define _JACCARD_SAMPLER_H_

#include "common/setjoin_parallel.h"
#include "sampler/base_sampler.h"


/*
 * To minimize the sampling time, we employ the parallel join
 * on small datasets (i.e., size less than 10 k), the parallel algorithm may be slower (~0.02s)
 * on large datasets (i.e., size larger than 1m), the parallel algorithm is significantly faster
 */
class JacSampler : public Sampler
{
public:
    using Sampler::Sampler;
    double det{0.0};

public:
    JacSampler() = default;
    JacSampler(double _det, std::string _blkAttr, bool _isRS) : Sampler(_blkAttr, _isRS), det(_det) {
        std::cout << "spawn jaccard sampler: " << blkAttr << " " << det << " is RS Join: " << isRS << std::endl;
    }
    ~JacSampler() = default;
    JacSampler(const JacSampler &other) = delete;
    JacSampler(JacSampler &&other) = delete;

public:
    void sample(const std::string &pathTableA, const std::string &pathTableB);
};


#endif // _JACCARD_SAMPLER_H_