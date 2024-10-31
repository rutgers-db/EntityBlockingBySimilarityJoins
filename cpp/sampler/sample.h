/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _SAMPLE_H_
#define _SAMPLE_H_

#include "common/simfunc.h"
#include "common/io.h"
#include "sampler/base_sampler.h"
#include "sampler/down_sampler.h"
#include "sampler/jaccard_sampler.h"
#include "sampler/overlap_sampler.h"
#include <cstdlib>
#include <time.h>
#include <parallel/algorithm>
#include <omp.h>


class Sample
{
private:
    static int CLUSTER_SAMPLE_SIZE;

    struct dsu
    {
        std::vector<int> fa;

        dsu(ui size): fa(size) { 
            std::iota(fa.begin(), fa.end(), 0); 
        }

        int find(int x) {
            return fa[x] == x ? x : fa[x] = find(fa[x]);
        }

        void unite(int x, int y) {
            fa[find(x)] = find(y);
        }
    };

public:
    Sample() = default;
    ~Sample() = default;
    Sample(const Sample &other) = delete;
    Sample(Sample &&other) = delete;

private:
    // 2-step cluster
    static std::pair<double, double> getStat(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
                                             const std::vector<ui> &idMapB);
    static void step2Sample(const std::string &blkAttr, double step2Tau, std::vector<std::pair<int, int>> &pairs, 
                            const Table &tableA, const Table &tableB, const std::vector<ui> &idMapA, 
                            const std::vector<ui> &idMapB, bool isRS);

public:
    // cluster
    static void clusterSampleSelf(const std::string &blkAttr, double clusterTau, double blkTau, 
                                  const std::string &pathTableA, const std::string &pathTableB, 
                                  const std::string &defaultOutputDir = "");
    /*
     * it is not appropriate to apply clustering on RS-join
     * thus we only adopt the normal 2-step jaccard sampling
     */
    static void clusterSampleRS(const std::string &blkAttr, double clusterTau, double blkTau, double step2Tau,
                                const std::string &pathTableA, const std::string &pathTableB, 
                                const std::string &defaultOutputDir = "");
    
    // down
    static void downSample(ui n, ui y, const std::string &blkAttr, bool isRS, 
                           const std::string &pathTableA, const std::string &pathTableB, 
                           const std::string &defaultOutputDir = "");
    
    // pre, may be depracted
    /*
     * Directly read the provided sampled subset
     * this can only be used on synthetic datasets
     * the real-wolrd datasets has no pre-sampled subset
     */
    static void preSample(ui n, int datanum, const std::string &blkAttr, const std::string &pathZ, 
                          const std::string &pathY, const std::string &defaultOutputDir = "");
};

#endif // _SAMPLE_H_