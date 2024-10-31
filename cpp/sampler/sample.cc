/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "sampler/sample.h"

int Sample::CLUSTER_SAMPLE_SIZE = 100000;


void Sample::clusterSampleSelf(const std::string &blkAttr, double clusterTau, double blkTau, 
                               const std::string &pathTableA, const std::string &pathTableB, 
                               const std::string &defaultOutputDir)
{
    // cluster
    JacSampler jacCluster(clusterTau, blkAttr, false);
    jacCluster.sample(pathTableA, pathTableB);

    dsu cluster(jacCluster.tableA.row_no);
    ui rawSize = jacCluster.pairs.size();
    for(ui i = 0; i < rawSize; i++) {
        int id1 = jacCluster.pairs[i].first;
        int id2 = jacCluster.pairs[i].second;
        cluster.unite(id1, id2);
        if(id1 == id2) {
            std::cerr << "Jac error: " << id1 << " " << id2 << std::endl;
            exit(1);
        }
    }

    std::unordered_map<int, int> cidMap;
    std::vector<std::vector<int>> clusterBucket;

    for(ui i = 0; i < jacCluster.tableA.row_no; i++) {
        if(cluster.find(i) == i) {
            clusterBucket.emplace_back();
            ui cid = clusterBucket.size() - 1;
            cidMap[i] = cid;
            clusterBucket[cid].emplace_back(i);
        }
    }
    for(ui i = 0; i < jacCluster.tableA.row_no; i++) {
        if(cluster.find(i) != i) {
            int cid = cidMap[cluster.find(i)];
            clusterBucket[cid].emplace_back(i);
        }
    }

    // sample
    JacSampler jacSampler(blkTau, blkAttr, false);
    jacSampler.sample(pathTableA, pathTableB);

    // cluster pair
    std::vector<std::pair<int, int>> clusterPairs;

    ui firstSize = jacSampler.pairs.size();
    for(ui i = 0; i < firstSize; i++) {
        int id1 = jacSampler.pairs[i].first;
        int id2 = jacSampler.pairs[i].second;
        int cid1 = cidMap[cluster.find(id1)];
        int cid2 = cidMap[cluster.find(id2)];

        if(count(clusterBucket[cid1].begin(), clusterBucket[cid1].end(), id1) == 0) {
            std::cerr << "Error in clustering" << std::endl;
            clusterBucket[cid1].emplace_back(id1);
        }
        if(count(clusterBucket[cid2].begin(), clusterBucket[cid2].end(), id2) == 0) {
            std::cerr << "Error in clustering" << std::endl;
            clusterBucket[cid2].emplace_back(id2);
        }

        // build
        if(cid1 == cid2)
            continue;
        else {
            int cidMin = std::min(cid1, cid2);
            int cidMax = std::max(cid1, cid2);
            clusterPairs.emplace_back(cidMin, cidMax);
        }
    }
   
    std::sort(clusterPairs.begin(), clusterPairs.end());
    auto uiter = std::unique(clusterPairs.begin(), clusterPairs.end());
    clusterPairs.resize(std::distance(clusterPairs.begin(), uiter));

    std::vector<std::pair<int, int>> samplePairs;
    for(const auto &p : clusterPairs) {
        int oneid1 = clusterBucket[p.first][0];
        int oneid2 = clusterBucket[p.second][0];
        samplePairs.emplace_back(oneid1, oneid2);
    }

    // random
    std::sort(samplePairs.begin(), samplePairs.end());
    if(samplePairs.size() > CLUSTER_SAMPLE_SIZE) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(samplePairs.begin(), samplePairs.end(), std::default_random_engine(seed));
        samplePairs.resize(CLUSTER_SAMPLE_SIZE);
    }

    std::vector<int> label(samplePairs.size(), 0);

#ifdef ARROW_INSTALLED
    MultiWriter::writeSampleResMegallenParquet(samplePairs, jacSampler.idMapA, jacSampler.idMapA, jacSampler.tableA, jacSampler.tableA, 
                                               label, defaultOutputDir);
#endif
#ifndef ARROW_INSTALLED
    MultiWriter::writeSampleResMegallenCSV(samplePairs, jacSampler.idMapA, jacSampler.idMapA, jacSampler.tableA, jacSampler.tableA, 
                                           label, defaultOutputDir);
#endif
}


// 2 step sampling
std::pair<double, double> Sample::getStat(const std::vector<std::pair<int, int>> &pairs, const std::vector<ui> &idMapA, 
                                          const std::vector<ui> &idMapB)
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory += "../../output/buffer/";

    std::string pathGold = directory + "gold.csv";
    CSVReader reader;
    bool normalize = false;
    bool readRes = reader.reading_one_table(pathGold, normalize);
    if(readRes == false) {
        std::cerr << "no gold presents in buffer or no buffer" << std::endl;
        return std::make_pair(0.0, 0.0);
    }

    Table goldTable = reader.tables[0];
    goldTable.Profile();
    std::vector<std::pair<int, int>> result;
    for(const auto &p : pairs)
        result.emplace_back(idMapA[p.first], idMapB[p.second]);
    std::vector<std::pair<int, int>> golds;

    for(int i = 0; i < goldTable.row_no; i++) {
        int lid = std::stoi(goldTable.rows[i][0]);
        int rid = std::stoi(goldTable.rows[i][1]);
        golds.emplace_back(lid, rid);
    }

    std::sort(golds.begin(), golds.end());
    std::sort(result.begin(), result.end());
    std::vector<std::pair<int, int>> res;
    std::set_intersection(golds.begin(), golds.end(), 
                          result.begin(), result.end(), 
                          std::back_inserter(res));

    double recall = res.size() * 1.0 / golds.size();
    double density = res.size() * 1.0 / result.size();

    std::cout << "--- recall: " << recall << std::endl;
    std::cout << "--- density: " << density << std::endl;
    return std::make_pair(recall, density);
}


void Sample::step2Sample(const std::string &blkAttr, double step2Tau, std::vector<std::pair<int, int>> &pairs, 
                         const Table &tableA, const Table &tableB, const std::vector<ui> &idMapA, 
                         const std::vector<ui> &idMapB, bool isRS)
{
    std::vector<std::pair<int, int>> tempPairs;
    for(const auto &p : pairs) 
        tempPairs.emplace_back((int)idMapA[p.first], (int)idMapB[p.second]);

    std::vector<std::vector<ui>> recordsAQgm, recordsBQgm;
    std::vector<double> weightsAQgm, weightsBQgm;
    std::vector<double> wordwt;
    std::vector<ui> idMapAQgm, idMapBQgm;
    ui numWord = 0;
    ui columnA = tableA.inverted_schema.at(blkAttr);
    ui columnB = columnA;
    
    if(isRS) {
        Tokenizer::RStableAttr2IntVector(tableA, tableB, recordsAQgm, recordsBQgm, weightsAQgm, weightsBQgm, 
                                         wordwt, idMapAQgm, idMapBQgm, columnA, columnB, TokenizerType::QGram,
                                         numWord, 3);
    }
    else {
        Tokenizer::SelftableAttr2IntVector(tableA, recordsAQgm, weightsAQgm, wordwt, idMapAQgm, 
                                           columnA, TokenizerType::QGram, numWord, 3);
        recordsBQgm = recordsAQgm;
        weightsBQgm = weightsAQgm;
        idMapBQgm = idMapAQgm;
    }

    std::vector<ui> reverseIdMapA(idMapAQgm.size(), 0);
    std::vector<ui> reverseIdMapB(idMapBQgm.size(), 0);
    for(size_t i = 0; i < idMapAQgm.size(); i++)
        reverseIdMapA[idMapAQgm[i]] = i;
    for(size_t i = 0; i < idMapBQgm.size(); i++)
        reverseIdMapB[idMapBQgm[i]] = i;

    for(auto &p : tempPairs) {
        p.first = (int)reverseIdMapA[p.first];
        p.second = (int)reverseIdMapB[p.second];
    }

    std::vector<ui> revIdMapA(idMapA.size(), 0);
    std::vector<ui> revIdMapB(idMapB.size(), 0);
    for(size_t i = 0; i < idMapA.size(); i++)
        revIdMapA[idMapA[i]] = i;
    for(size_t i = 0; i < idMapB.size(); i++)
        revIdMapB[idMapB[i]] = i;

    if(revIdMapA.size() != reverseIdMapA.size() || revIdMapB.size() != reverseIdMapB.size()) {
        std::cerr << "error in tokenizing: " << revIdMapA.size() << " " << reverseIdMapA.size() 
                  << " " << revIdMapB.size() << " " << reverseIdMapB.size() << std::endl;
    }

    pairs.clear();
    std::vector<std::pair<int, int>> tmp[MAXTHREADNUM];

#pragma omp parallel for
    for(const auto &p : tempPairs) {
        int tid = omp_get_thread_num();
        int lid = p.first;
        int rid = p.second;
        double val = SimFuncs::jaccard(recordsAQgm[lid], recordsBQgm[rid]);
        if(val >= step2Tau) 
            tmp[tid].emplace_back(revIdMapA[idMapAQgm[p.first]], revIdMapB[idMapBQgm[p.second]]);
    }

    for(int tid = 0; tid < MAXTHREADNUM; tid++)
        pairs.insert(pairs.end(), tmp[tid].begin(), tmp[tid].end());
}


void Sample::clusterSampleRS(const std::string &blkAttr, double clusterTau, double blkTau, double step2Tau,
                             const std::string &pathTableA, const std::string &pathTableB, 
                             const std::string &defaultOutputDir)
{
    
    if(blkTau < 1.0) {
        JacSampler jacSampler(blkTau, blkAttr, true);
        jacSampler.sample(pathTableA, pathTableB);

        if(jacSampler.pairs.size() > CLUSTER_SAMPLE_SIZE)
            jacSampler.pairs.resize(CLUSTER_SAMPLE_SIZE);

        std::pair<double, double> stat = getStat(jacSampler.pairs, jacSampler.idMapA, jacSampler.idMapB);
        if(stat.second <= 0.1) {
            std::cout << "trigger 2 step sample" << std::endl;
            step2Sample(blkAttr, step2Tau, jacSampler.pairs, jacSampler.tableA, jacSampler.tableB, 
                        jacSampler.idMapA, jacSampler.idMapB, true);
            stat = getStat(jacSampler.pairs, jacSampler.idMapA, jacSampler.idMapB);
        }

        std::vector<int> label(jacSampler.pairs.size(), 0);

#ifdef ARROW_INSTALLED
        MultiWriter::writeSampleResMegallenParquet(jacSampler.pairs, jacSampler.idMapA, jacSampler.idMapB, jacSampler.tableA, jacSampler.tableB, 
                                                   label, defaultOutputDir);
#endif
#ifndef ARROW_INSTALLED
        MultiWriter::writeSampleResMegallenCSV(jacSampler.pairs, jacSampler.idMapA, jacSampler.idMapB, jacSampler.tableA, jacSampler.tableB, 
                                               label, defaultOutputDir);
#endif
    }
    else {
        int ovlpTau = ceil(blkTau - 1e-5);
        OvlpSampler ovlpSampler(ovlpTau, blkAttr, true);
        ovlpSampler.sample(pathTableA, pathTableB);

        if(ovlpSampler.pairs.size() > CLUSTER_SAMPLE_SIZE)
            ovlpSampler.pairs.resize(CLUSTER_SAMPLE_SIZE);

        std::pair<double, double> stat = getStat(ovlpSampler.pairs, ovlpSampler.idMapA, ovlpSampler.idMapB);
        if(stat.second <= 0.1) {
            std::cout << "trigger 2 step sample" << std::endl;
            step2Sample(blkAttr, step2Tau, ovlpSampler.pairs, ovlpSampler.tableA, ovlpSampler.tableB, 
                        ovlpSampler.idMapA, ovlpSampler.idMapB, true);
            stat = getStat(ovlpSampler.pairs, ovlpSampler.idMapA, ovlpSampler.idMapB);
        }

        std::vector<int> label(ovlpSampler.pairs.size(), 0);

#ifdef ARROW_INSTALLED
        MultiWriter::writeSampleResMegallenParquet(ovlpSampler.pairs, ovlpSampler.idMapA, ovlpSampler.idMapB, ovlpSampler.tableA, ovlpSampler.tableB, 
                                                   label, defaultOutputDir);
#endif
#ifndef ARROW_INSTALLED
        MultiWriter::writeSampleResMegallenCSV(ovlpSampler.pairs, ovlpSampler.idMapA, ovlpSampler.idMapB, ovlpSampler.tableA, ovlpSampler.tableB, 
                                               label, defaultOutputDir);
#endif
    }
}


void Sample::downSample(ui n, ui y, const std::string &blkAttr, bool isRS, 
                        const std::string &pathTableA, const std::string &pathTableB, 
                        const std::string &defaultOutputDir)
{
    DownSampler downSampler(n, y, blkAttr, isRS);
    downSampler.sample(pathTableA, pathTableB);

    std::vector<int> label(downSampler.pairs.size(), 0);

#ifdef ARROW_INSTALLED
    if(num_data == 2)
        MultiWriter::writeSampleResMegallenParquet(topK, jacSampler.idMapA[0], jacSampler.idMapB[0], jacSampler.tableA, jacSampler.tableB, 
                                                   label, defaultOutputDir);
    else if(num_data == 1)
        MultiWriter::writeSampleResMegallenParquet(topK, jacSampler.idMapA[0], jacSampler.idMapA[0], jacSampler.tableA, jacSampler.tableA, 
                                                   label, defaultOutputDir);
#endif
#ifndef ARROW_INSTALLED
    if(isRS == true) {
        MultiWriter::writeSampleResMegallenCSV(downSampler.pairs, downSampler.idMapA, downSampler.idMapB, downSampler.tableA, downSampler.tableB, 
                                               label, defaultOutputDir);
    }
    else {
        MultiWriter::writeSampleResMegallenCSV(downSampler.pairs, downSampler.idMapA, downSampler.idMapA, downSampler.tableA, downSampler.tableA, 
                                               label, defaultOutputDir);
    }
#endif
}


void Sample::preSample(ui n, int datanum, const std::string &blkAttr, const std::string &pathZ, const std::string &pathY, 
                       const std::string &defaultOutputDir)
{
    bool normalize = false;

    CSVReader reader;
    reader.reading_one_table(pathZ, normalize);
    reader.reading_one_table(pathY, normalize);
    Table Z = reader.tables[0];
    Table Y = reader.tables[1];
    Z.Profile();
    Y.Profile();

    std::vector<int> sampleid;
    std::unordered_map<int, int> id2rowno;
    int maxid = 0;
    for(int i = 0; i < Z.row_no; i++) {
        int id_ = stoi(Z.rows[i][0]);
        sampleid.emplace_back(id_);
        id2rowno[id_] = i;
        maxid = maxid >= id_ ? maxid : id_;
    }
    // sort(sampleid.begin(), sampleid.end());

    // get token index
    ui num_row = Z.rows.size();
    ui pos = Z.inverted_schema[blkAttr];
    std::cout << pos << std::endl;
	std::string delim = " \"\',\\\t\r\n";
    std::vector<std::vector<std::string>> bows;
    bows.resize(num_row);
	Tokenizer::updateBagDlm(Z, bows, pos, delim, 0);

    std::unordered_map<std::string, std::vector<int>> invIndex;
    for(ui i = 0; i < num_row; i++)
        for(const auto &tok : bows[i])
            invIndex[tok].emplace_back(stoi(Z.rows[i][0]));
    for(auto &it : invIndex) {
        sort(it.second.begin(), it.second.end());
        // printf("%s\n", it.first.c_str());
        // for(const auto &e : it.second) 
        //     printf("%d %s\n", e, Z.rows[id2rowno[e]][1].c_str());
    }
    std::cout << "index done" << std::endl;

    // first: id, second: label
    std::vector<std::vector<int>> matches(maxid); 
    std::vector<std::vector<int>> labels(maxid);
    for(int i = 0; i < Y.row_no; i++) {
        int lid = stoi(Y.rows[i][0]);
        int rid = stoi(Y.rows[i][1]);
        int lidx = id2rowno[lid];
        int ridx = id2rowno[rid];
        std::vector<std::string> res;
        std::set_intersection(bows[lidx].begin(), bows[lidx].end(), 
                              bows[ridx].begin(), bows[ridx].end(), 
                              std::back_inserter(res));
        double jac = (int)res.size() * 1.0 / ((int)bows[lidx].size() + (int)bows[ridx].size() - (int)res.size());
        // if(jac <= 0.57)
        //     continue;
        matches[lid].emplace_back(rid);
        labels[lid].emplace_back(1);
    }

    // select pairs that shares at least one tokens
    std::vector<std::vector<std::pair<int, double>>> bucket(num_row);
#pragma omp parallel for
    for(int idx = 0; idx < Z.row_no; idx++) {
        int lid = sampleid[idx];
        std::vector<bool> tag(num_row, false);
        for(const auto &tok : bows[idx]) {
            auto st = upper_bound(invIndex[tok].begin(), invIndex[tok].end(), lid);
            for(; st != invIndex[tok].end(); st++) {
                int rid = *st;
                if(count(matches[lid].begin(), matches[lid].end(), rid) != 0)
                    continue;
                std::vector<std::string> res;
                std::set_intersection(bows[idx].begin(), bows[idx].end(), 
                                      bows[id2rowno[rid]].begin(), bows[id2rowno[rid]].end(), 
                                      std::back_inserter(res));
                if(tag[id2rowno[rid]])
                    continue;
                double jac = (int)res.size() * 1.0 / ((int)bows[idx].size() + (int)bows[id2rowno[rid]].size() - (int)res.size());
                bucket[idx].emplace_back(rid, jac);
                tag[id2rowno[rid]] = true;
                // matches[lid].emplace_back(rid);
                // labels[lid].emplace_back(0);
            }
        }
    }
    for(auto &b : bucket)
        sort(b.begin(), b.end(), [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) {
            return lhs.second > rhs.second;
        });
    // const int selected = 10;
    for(int idx = 0; idx < Z.row_no; idx++) {
        int lid = sampleid[idx];
        // if((int)bucket[idx].size() > selected)
        //     bucket[idx].resize(selected);
        for(const auto &p : bucket[idx]) {
            // if(p.second >= 0.7 || p.second <= 0.5)
            //     continue;
            matches[lid].emplace_back(p.first);
            labels[lid].emplace_back(0);
        }
    }

    ui totsize = 0;
    for(const auto &vec : matches)
        totsize += vec.size();
    printf("Total: %u\tPositive: %d\tNegative: %d\n", totsize, Y.row_no, (int)totsize - Y.row_no);

    // flush
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory += "../../output/buffer/";
    std::string outputPath = directory + "sample_res.csv";
    FILE *csvfile = fopen(outputPath.c_str(), "w");

    // first three schemas
	fprintf(csvfile, "_id,ltable_id,rtable_id");
	// others
	for(ui i = 1; i < Z.schema.size(); i++) {
		std::string curAttr = Z.schema[i];
		std::string newAttr = "ltable_" + curAttr;
		fprintf(csvfile, ",%s", newAttr.c_str());
	}
	for(ui i = 1; i < Z.schema.size(); i++) {
		std::string curAttr = Z.schema[i];
		std::string newAttr = "rtable_" + curAttr;
		fprintf(csvfile, ",%s", newAttr.c_str());
	}
	fprintf(csvfile, ",label\n");

	// values
    long long l = 0;
	for(int i = 0; i < Z.row_no; i++) {
		int lid = stoi(Z.rows[i][0]);
        int idx = 0;
        for(const auto &rid : matches[lid]) {
            fprintf(csvfile, "%lld,%d,%d", l, lid, rid);
            // out attrs
            for(ui j = 1; j < Z.schema.size(); j++) {
                std::string curAttr = Z.rows[i][j];
                MultiWriter::escapeOneRow(curAttr);
                fprintf(csvfile, ",%s", curAttr.c_str());
            }
            for(ui j = 1; j < Z.schema.size(); j++) {
                std::string curAttr = Z.rows[id2rowno[rid]][j];
                MultiWriter::escapeOneRow(curAttr);
                fprintf(csvfile, ",%s", curAttr.c_str());
            }

            fprintf(csvfile, ",%d\n", labels[lid][idx]);
            ++ l;
            ++ idx;
        }
	}

    fclose(csvfile);
}


// ctype apis
extern "C"
{
    void cluster_sample_self(const char *blk_attr, double cluster_tau, double blk_tau, const char *path_table_A, 
                             const char *path_table_B, const char *defaultOutputDir) {
        Sample::clusterSampleSelf(blk_attr, cluster_tau, blk_tau, path_table_A, path_table_B, defaultOutputDir);
    }

    void cluster_sample_RS(const char *blk_attr, double cluster_tau, double blk_tau, double step2_tau, 
                           const char *path_table_A, const char *path_table_B, 
                           const char *defaultOutputDir) {
        Sample::clusterSampleRS(blk_attr, cluster_tau, blk_tau, step2_tau, path_table_A, path_table_B, defaultOutputDir);
    }

    void down_sample(unsigned int n, unsigned int y, const char *blk_attr, bool is_RS, 
                     const char *path_table_A, const char *path_table_B, 
                     const char *defaultOutputDir) {
        Sample::downSample(n, y, blk_attr, is_RS, path_table_A, path_table_B, defaultOutputDir);
    }

    void pre_sample(unsigned int n, int data_num, const char *blk_attr, const char *path_Z, 
                    const char *path_Y) {
        Sample::preSample(n, data_num, blk_attr, path_Z, path_Y);
    }
}