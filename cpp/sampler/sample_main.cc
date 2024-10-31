/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
/*
 * main entrance for sampler
 * the pre sample has not been tested, use at your own risk
 * also, the pre sample can only be used on SIGMOD Programming Contest 2022 datasets
 * where a subset of entities as well as labels are provided
 */
#include "sampler/sample.h"


int main(int argc, char **argv)
{
    std::string blkAttr = argv[1];
    int num_data = atoi(argv[2]);
    std::string sampleStrategy = argv[3];
    bool isRS = num_data == 1 ? false : true;

    if(sampleStrategy == "cluster") {
        double sameTau = atof(argv[4]);
        double blkTau = atof(argv[5]);
        double step2Tau = atof(argv[6]);
        std::string pathTableA = argv[7];
        std::string pathTableB = argv[8];
        std::string pathDefaultOutputDir = argv[9];
        printf("Blocking attribute: %s\tNumber of data: %d\tClustering tau: %.4lf\tBlocking tau: %.4lf\n", 
            blkAttr.c_str(), num_data, sameTau, blkTau);

        if(isRS)
            Sample::clusterSampleRS(blkAttr, sameTau, blkTau, step2Tau, pathTableA, pathTableB, pathDefaultOutputDir);
        else
            Sample::clusterSampleSelf(blkAttr, sameTau, blkTau, pathTableA, pathTableB, pathDefaultOutputDir);
    }
    else if(sampleStrategy == "down") {
        int n = atoi(argv[4]);
        int y = atoi(argv[5]);
        std::string pathTableA = argv[6];
        std::string pathTableB = argv[7];
        std::string pathDefaultOutputDir = argv[8];
        printf("Blocking attribute: %s\tNumber of data: %d\tn:%d\ty:%d\n", blkAttr.c_str(), num_data, n, y);

        Sample::downSample((ui)n, (ui)y, blkAttr, isRS, pathTableA, pathTableB, pathDefaultOutputDir);
    }
    else if(sampleStrategy == "pre") {
        int n = atoi(argv[4]);
        int datanum = atoi(argv[5]);
        std::string pathZ = argv[6];
        std::string pathY = argv[7];
        printf("Loading secret data %d pre-sampled with size: %u\tblock attr: %ssample Z: %s\tsample Y: %s\n", datanum, n, blkAttr.c_str(), 
               pathZ.c_str(), pathY.c_str());

        Sample::preSample((ui)n, datanum, blkAttr, pathZ, pathY);
    }
    else {
        std::cerr << "No such strategy: " << sampleStrategy << std::endl;
        exit(1);
    }
    
    return 0;
}