/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _OVERLAP_SAMPLER_H_
#define _OVERLAP_SAMPLER_H_

#include "common/ovlpjoin_parallel.h"
#include "sampler/base_sampler.h"


/*
 * To minimize the sampling time, we use the parallel join
 */
class OvlpSampler : public Sampler
{
public:
    using Sampler::Sampler;
    int det{0};

public:
    OvlpSampler() = default;
    OvlpSampler(int _det, std::string _blkAttr, bool _isRS) : Sampler(_blkAttr, _isRS), det(_det) {
        std::cout << "spawn overlap sampler: " << blkAttr << " " << det << " is RS Join: " << isRS << std::endl;
    }
    ~OvlpSampler() = default;
    OvlpSampler(const OvlpSampler &other) = delete;
    OvlpSampler(OvlpSampler &&other) = delete;

public:
    void sample(const std::string &pathTableA, const std::string &pathTableB);
};


#endif // _OVERLAP_SAMPLER_H_