/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "sampler/base_sampler.h"
#include "sampler/down_sampler.h"
#include "sampler/jaccard_sampler.h"
#include "sampler/overlap_sampler.h"


void Sampler::readTable(const std::string &_pathTableA, const std::string &_pathTableB) 
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory += "../../output/";
    std::string defaultPathA = directory + "buffer/clean_A.csv";
    std::string defaultPathB = directory + "buffer/clean_B.csv";

    pathTableA = _pathTableA == "" ? defaultPathA : _pathTableA;
    pathTableB = _pathTableB == "" ? defaultPathB : _pathTableB;
    SamplerUtil::readCSVTables(pathTableA, pathTableB, tableA, tableB, isRS);
    std::cout << std::endl << "reading table a: " << pathTableA << std::endl;
    if(isRS == true)
        std::cout << "reading table b: " << pathTableB << std::endl; 
}


void Sampler::prepareRecords(ui columnA, ui columnB, TokenizerType tt, ui q) 
{
    if(isRS == true) {
        // tokenize & sort records
        Tokenizer::RStableAttr2IntVector(tableA, tableB, recordsA, recordsB, 
                                            weightsA, weightsB, wordwt,
                                            idMapA, idMapB, columnA, columnB, 
                                            tt, numWord, q);
    }
    else {
        // tokenize & sort records
        Tokenizer::SelftableAttr2IntVector(tableA, recordsA, weightsA, wordwt, 
                                            idMapA, columnA, tt, numWord, q);
        tableB = tableA;
        recordsB = recordsA;
        weightsB = weightsA;
        idMapB = idMapA;
    }
}


void DownSampler::sample(const std::string &pathTableA, const std::string &pathTableB) 
{
   readTable(pathTableA, pathTableB);

    auto iter = tableA.inverted_schema.find(blkAttr);
    if(iter == tableA.inverted_schema.end()) {
        std::cerr << "No such key: " << blkAttr << std::endl;
        exit(1);
    }

    prepareRecords(iter->second, iter->second, TokenizerType::Dlm, 0);

    timeval begin, end;
    gettimeofday(&begin, NULL);

    // construct inverted index I
    auto &sampleRecordsA = recordsA;
    auto &sampleRecordsB = recordsB;
    for(ui i = 0; i < tableA.row_no; i++)
        for(const auto &word : sampleRecordsA[i]) 
            tokenIndex[word].emplace_back(i);

    // random select n / y tuples from B
    ui randomNum = std::ceil(n * 1.0 / y - 1e-5);
    if(randomNum > (ui)tableB.row_no)
        randomNum = (ui)tableB.row_no;

    std::vector<ui> idx(tableB.row_no);
    std::iota(idx.begin(), idx.end(), 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
    idx.resize(randomNum);
    std::vector<bool> tag(tableB.row_no, false);
    for(const auto &e : idx)
        tag[e] = true;

    // for each tuple b in idx, select y tuples in A to pair
    // firstly using index I to find tuples a to form set X that shares at least one tokens with b
    // then sort X in decreasing order of number of shared tokens with b
    // select top y1 = min(y/2, |X|) tuples from X
    // finally select y-y1 tuples from the remaining tuples in A
    struct SampleEntity {
        ui id{0};
        ui sharing{0};

        SampleEntity() = default;
        SampleEntity(ui _id, ui _sharing) 
        : id(_id), sharing(_sharing) { }

        bool operator==(const SampleEntity &rhs) const {
            return this->id == rhs.id;
        }
    };

    std::default_random_engine drng[MAXTHREADNUM];
#pragma omp parallel for
    for(int ii = 0; ii < MAXTHREADNUM; ii++) {
        int tid = omp_get_thread_num();
        unsigned seed2 = std::chrono::system_clock::now().time_since_epoch().count() + (unsigned)tid;
        drng[tid] = std::default_random_engine{seed2};
    }

#pragma omp parallel for
    for(const auto &b : idx) { 
        int tid = omp_get_thread_num();
        std::vector<SampleEntity> X;
        std::vector<ui> secPartIdx(tableA.row_no);
        std::vector<bool> quickRef(tableA.row_no, false);
        std::iota(secPartIdx.begin(), secPartIdx.end(), 0);

        for(const auto &word : sampleRecordsB[b]) {
            auto iter = tokenIndex.find(word);
            if(iter == tokenIndex.end())
                continue;

            std::vector<ui> res;
            for(const auto &vec : iter->second) {
                if((!isRS && vec == b) || quickRef[vec])
                    continue;
                res.clear();
                std::set_intersection(sampleRecordsB[b].begin(), sampleRecordsB[b].end(), 
                                        sampleRecordsA[vec].begin(), sampleRecordsA[vec].end(), 
                                        std::back_inserter(res));
                X.emplace_back(vec, res.size());
                quickRef[vec] = true;
            }
        }

        // no more deduplication needed
        // std::sort(X.begin(), X.end(), [](const SampleEntity &se1, const SampleEntity &se2) {
        //     return se1.id < se2.id;
        // });
        // auto uiter = std::unique(X.begin(), X.end());
        // X.resize(std::distance(X.begin(), uiter));
        
        std::sort(X.begin(), X.end(), [](const SampleEntity &se1, const SampleEntity &se2) {
            return se1.sharing > se2.sharing;
        });

        ui ub = std::min(y/2, (ui)X.size());
        if(isRS == true) {
            std::vector<ui> firPartIdx;
            // first part
            for(ui i = 0; i < ub; i++) {
                samplePairs[tid].emplace_back((int)X[i].id, (int)b);
                firPartIdx.emplace_back(X[i].id);
            }

            // second part
            std::shuffle(secPartIdx.begin(), secPartIdx.end(), drng[tid]);
            ui count = 0;
            for(const auto &e : secPartIdx) {
                if(std::count(firPartIdx.begin(), firPartIdx.end(), e) != 0)
                    continue;
                samplePairs[tid].emplace_back((int)e, (int)b);
                ++ count;
                if(count >= y - ub)
                    break;
            }
        }
        else {
            std::vector<ui> firPartIdx;
            firPartIdx.emplace_back(b);
            // first part
            ui count = 0;
            for(const auto &e : X) {
                if(tag[e.id])
                    continue;
                firPartIdx.emplace_back(e.id);
                int minid = std::min(b, e.id);
                int maxid = std::max(b, e.id);
                samplePairs[tid].emplace_back(minid, maxid);
                ++ count;
                if(count >= ub)
                    break;
            }
            
            // second part
            std::shuffle(secPartIdx.begin(), secPartIdx.end(), drng[tid]);
            count = 0;
            for(const auto &e : secPartIdx) {
                if(std::count(firPartIdx.begin(), firPartIdx.end(), e) != 0 || tag[e])
                    continue;
                int minid = std::min(b, e);
                int maxid = std::max(b, e);
                samplePairs[tid].emplace_back(minid, maxid);
                ++ count;
                if(count >= y - ub)
                    break;
            }
        }
    }

    gettimeofday(&end, NULL);

    // merge
    for(int tid = 0; tid < MAXTHREADNUM; tid++)
        pairs.insert(pairs.end(), samplePairs[tid].begin(), samplePairs[tid].end());
    std::sort(pairs.begin(), pairs.end());
    auto uuiter = std::unique(pairs.begin(), pairs.end());
    if(uuiter != pairs.end()) {
        std::cerr << "Duplicate in down sampling results" << std::endl;
        for(auto iit = uuiter; iit != pairs.end(); iit++)
            printf("%d %d\n", iit->first, iit->second);
    }

    // stat
    double time = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
    printf("Down sampling: %zu\tTime: %.4lf\n", pairs.size(), time);
}


void JacSampler::sample(const std::string &pathTableA, const std::string &pathTableB) 
{
    readTable(pathTableA, pathTableB);

    auto iter = tableA.inverted_schema.find(blkAttr);
    if(iter == tableA.inverted_schema.end()) {
        std::cerr << "No such key: " << blkAttr << std::endl;
        exit(1);
    }

    prepareRecords(iter->second, iter->second, TokenizerType::Dlm, 0);
    // std::cout << "here" << std::endl << std::flush;

    timeval begin, end;
    double st;

    if(isRS == true) {
        SetJoinParallel *joiner = new SetJoinParallel(recordsB, recordsA, weightsB, weightsA, wordwt, det);
        joiner->simFType = SimFuncType::JACCARD;
        gettimeofday(&begin, NULL);
        joiner->index(det);
        joiner->findSimPairsRS();
        joiner->mergeResults(pairs);
        gettimeofday(&end, NULL);
        st = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
        delete joiner;
    }
    else {
        SetJoinParallel *joiner = new SetJoinParallel(recordsA, weightsA, wordwt, det);
        joiner->simFType = SimFuncType::JACCARD;
        gettimeofday(&begin, NULL);
        joiner->index(det);
        joiner->findSimPairsSelf();
        joiner->mergeResults(pairs);
        gettimeofday(&end, NULL);
        st = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
        delete joiner;
    }

    printf("~~~Sample(join) time~~~: %.4lf\n", st);
}


void OvlpSampler::sample(const std::string &pathTableA, const std::string &pathTableB) 
{
    readTable(pathTableA, pathTableB);

    auto iter = tableA.inverted_schema.find(blkAttr);
    if(iter == tableA.inverted_schema.end()) {
        std::cerr << "No such key: " << blkAttr << std::endl;
        exit(1);
    }

    prepareRecords(iter->second, iter->second, TokenizerType::Dlm, 0);
    // std::cout << "here" << std::endl << std::flush;

    timeval begin, end;
    double st;

    if(isRS == true) {
        OvlpRSJoinParallel *joiner = new OvlpRSJoinParallel(recordsA, recordsB, weightsA, weightsB, wordwt);
        gettimeofday(&begin, NULL);
        joiner->overlapjoin(det, pairs);
        gettimeofday(&end, NULL);
        st = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
        delete joiner;
    }
    else {
        OvlpSelfJoinParallel *joiner = new OvlpSelfJoinParallel(recordsA, weightsA, wordwt);
        gettimeofday(&begin, NULL);
        joiner->overlapjoin(det, pairs);
        gettimeofday(&end, NULL);
        st = end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6;
        delete joiner;
    }

    printf("~~~Sample(join) time~~~: %.4lf\n", st);
}