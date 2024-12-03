/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "group/group.h"


std::string Group::getICVDir(const std::string &defaultICVDir)
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultICVDir == "" ? directory + "../../simjoin_entitymatching/value_matcher/ic_values/" 
									: (defaultICVDir.back() == '/' ? defaultICVDir : defaultICVDir + "/");

    return directory;
}


void Group::readDocsAndVecs(std::vector<std::string> &docs, std::vector<std::vector<double>> &vecs, 
                            const std::string &defaultICVDir)
{
    // prefix
    std::string directory = getICVDir(defaultICVDir);

    // path
    std::string pathDocs = directory + "vec_interchangeable.txt";
    std::ifstream streamDocs(pathDocs.c_str(), std::ios::in);

    int totalDocs = 0;
    streamDocs >> totalDocs;
    docs.reserve(totalDocs);
    vecs.reserve(totalDocs);

    for(int i = 0; i < totalDocs; i++) {
        // read
        std::string doc = "";
        streamDocs >> doc;
        int vecSize = 0;
        streamDocs >> vecSize;
        std::vector<double> vec;
        vec.reserve(vecSize);
        for(int j = 0; j < vecSize; j++) {
            double val = 0.0;
            streamDocs >> val;
            vec.emplace_back(val);
        }

        // append
        std::sort(vec.begin(), vec.end());
        docs.emplace_back(doc);
        vecs.emplace_back(std::move(vec));
    }
}


void Group::readDocCandidatePairs(std::vector<std::pair<std::string, std::string>> &candidates, 
                                  const std::string &defaultICVDir)
{
    // prefix
    std::string directory = getICVDir(defaultICVDir);

    // path
    std::string pathPairs = directory + "pair_interchangeable.txt";
    std::ifstream streamPairs(pathPairs.c_str(), std::ios::in);

    int totalPairs = 0;
    streamPairs >> totalPairs;
    candidates.reserve(totalPairs);

    for(int i = 0; i < totalPairs; i++) {
        // read
        std::string lDoc = "";
        streamPairs >> lDoc;
        std::string rDoc = "";
        streamPairs >> rDoc;

        // append
        candidates.emplace_back(lDoc, rDoc);
    }
}


void Group::groupInterchangeableValuesByGraph(const std::string &groupAttribute, const std::string &groupStrategy, 
                                              double groupTau, bool isTransitiveClosure, 
                                              const std::string &defaultICVDir)
{
    // io
    std::vector<std::string> docs;
    std::vector<std::vector<double>> vecs;
    std::vector<std::pair<std::string, std::string>> candidates;

    Group::readDocsAndVecs(docs, vecs, defaultICVDir);
    Group::readDocCandidatePairs(candidates, defaultICVDir);

    // build
    Graph senmaticGraph(isTransitiveClosure, groupTau);
    senmaticGraph.buildSemanticGraph(docs, vecs, candidates);

    // write
    std::string directory = getICVDir(defaultICVDir);
    std::string pathGraph = directory + "interchangeable_graph_" 
                            + groupAttribute + ".txt";
    senmaticGraph.writeSemanticGraph(pathGraph);
}


extern "C"
{
    void group_interchangeable_values_by_graph(const char *group_attribute, const char *group_strategy, 
                                               double group_tau, bool is_transitive_closure, 
                                               const char *default_icv_dir) {
        Group::groupInterchangeableValuesByGraph(group_attribute, group_strategy, group_tau, is_transitive_closure, 
                                                 default_icv_dir);
    }

    void group_interchangeable_values_by_cluster() {
        std::cerr << "not established" << std::endl;
        exit(1);
    }
}