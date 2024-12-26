/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "group/group.h"


std::unordered_map<std::string, std::vector<double>> Group::saveNegEmbeddings(const std::vector<std::string> &docs, const DocEmbeddings &vecs)
{
    std::unordered_map<std::string, std::vector<double>> doc2Vec;
    for(size_t i = 0; i < docs.size(); i++)
        doc2Vec[docs[i]] = vecs[i];

    return doc2Vec;
}


std::unordered_map<std::string, std::vector<std::vector<double>>> Group::saveNegWordEmbeddings(const std::vector<std::string> &docs, const WordEmbeddings &vecs)
{
    std::unordered_map<std::string, std::vector<std::vector<double>>> doc2Vec;
    for(size_t i = 0; i < docs.size(); i++)
        doc2Vec[docs[i]] = vecs[i];

    return doc2Vec;
}


std::string Group::getICVDir(const std::string &defaultICVDir)
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultICVDir == "" ? directory + "../../simjoin_entitymatching/value_matcher/ic_values/" 
									: (defaultICVDir.back() == '/' ? defaultICVDir : defaultICVDir + "/");

    return directory;
}


std::string Group::getNegMatchDir(const std::string &defaultMatchResDir)
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultMatchResDir == "" ? directory + "../../simjoin_entitymatching/output/match_res/" 
                                         : (defaultMatchResDir.back() == '/' ? defaultMatchResDir : defaultMatchResDir + "/");

    return directory;
}


void Group::readDocsAndVecs(std::vector<std::string> &docs, DocEmbeddings &vecs, 
                            const std::string &defaultICVDir, 
                            const std::string &defaultTabName)
{
    // prefix
    std::string directory = getICVDir(defaultICVDir);

    // path
    std::string pathDocs = defaultTabName == "" ? directory + "vec_interchangeable.txt"
                                                : directory + defaultTabName;
    std::ifstream streamDocs(pathDocs.c_str(), std::ios::in);

    int totalDocs = 0;
    std::string header;
    getline(streamDocs, header);
    totalDocs = std::stoi(header);
    // allocate
    docs.reserve(totalDocs);
    vecs.reserve(totalDocs);

    for(int i = 0; i < totalDocs; i++) {
        // read doc
        std::string doc = "";
        getline(streamDocs, doc);

        // read vec
        std::string vecInfo;
        getline(streamDocs, vecInfo);
        std::istringstream iss(vecInfo);
        std::string token;

        // size
        getline(iss, token, ' ');
        int vecSize = std::stoi(token);
        std::vector<double> vec;
        vec.reserve(vecSize);

        // values
        for(int j = 0; j < vecSize; j++) {
            getline(iss, token, ' ');
            double val = std::stod(token);
            vec.emplace_back(val);
        }

        // append
        docs.emplace_back(doc);
        vecs.emplace_back(std::move(vec));
    }
}


void Group::readWordEmbeddingDocsAndVecs(std::vector<std::string> &docs, WordEmbeddings &vecs, 
                                         const std::string &defaultICVDir, 
                                         const std::string &defaultTabName)
{
    // prefix
    std::string directory = getICVDir(defaultICVDir);

    // path
    std::string pathDocs = defaultTabName == "" ? directory + "vec_interchangeable.txt"
                                                : directory + defaultTabName;
    std::ifstream streamDocs(pathDocs.c_str(), std::ios::in);

    int totalDocs = 0;
    std::string header;
    getline(streamDocs, header);
    totalDocs = std::stoi(header);
    // allocate
    docs.reserve(totalDocs);
    vecs.reserve(totalDocs);

    for(int i = 0; i < totalDocs; i++) {
        // read doc
        std::string doc = "";
        getline(streamDocs, doc);

        // read # of vecs
        std::string vecInfo;
        getline(streamDocs, vecInfo);
        int totalVecs = std::stoi(vecInfo);

        std::vector<std::vector<double>> vec;
        vec.reserve(totalVecs);

        for(int l = 0; l < totalVecs; l++) {
            getline(streamDocs, vecInfo);
            std::istringstream iss(vecInfo);
            std::string token;

            // size
            getline(iss, token, ' ');
            int vecSize = std::stoi(token);
            std::vector<double> wordVec;
            wordVec.reserve(vecSize);

            // values
            for(int j = 0; j < vecSize; j++) {
                getline(iss, token, ' ');
                double val = std::stod(token);
                wordVec.emplace_back(val);
            }

            vec.emplace_back(std::move(wordVec));
        }

        // append
        std::sort(vec.begin(), vec.end(), [](const std::vector<double> &lhs, const std::vector<double> &rhs) {
            size_t vecSize = lhs.size();
            for(size_t k = 0; k < vecSize; k++) {
                if(lhs[k] < rhs[k])
                    return true;
                else if(lhs[k] > rhs[k])
                    return false;
            }
        });
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
    std::string header;
    getline(streamPairs, header);
    totalPairs = std::stoi(header);
    // allocate
    candidates.reserve(totalPairs);

    for(int i = 0; i < totalPairs; i++) {
        // read
        std::string lDoc = "";
        getline(streamPairs, lDoc);
        std::string rDoc = "";
        getline(streamPairs, rDoc);

        // append
        candidates.emplace_back(lDoc, rDoc);
    }
}


void Group::groupInterchangeableValuesByGraph(const std::string &groupAttribute, const std::string &groupStrategy, 
                                              double groupTau, bool isTransitiveClosure, 
                                              const std::string &defaultICVDir)
{
    std::cout << "group interchangeable values on attribute : " << groupAttribute 
        << "\tstrategy : " << groupStrategy 
        << "\tdefault directory for output : " << defaultICVDir
        << "\tthreshold : " << groupTau
        << "\tis transitive closure enabled : " << isTransitiveClosure << std::endl;

    // io
    std::vector<std::string> docs;
    DocEmbeddings vecs;
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


void Group::groupInterchangeableValuesByWordGraph(const std::string &groupAttribute, const std::string &groupStrategy, 
                                                  double groupTau, bool isTransitiveClosure, 
                                                  const std::string &defaultICVDir)
{
    std::cout << "group interchangeable values on attribute : " << groupAttribute 
        << "\tstrategy : " << groupStrategy 
        << "\tdefault directory for output : " << defaultICVDir
        << "\tthreshold : " << groupTau
        << "\tis transitive closure enabled : " << isTransitiveClosure << std::endl;

    // io
    std::vector<std::string> docs;
    WordEmbeddings vecs;
    std::vector<std::pair<std::string, std::string>> candidates;

    Group::readWordEmbeddingDocsAndVecs(docs, vecs, defaultICVDir);
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


void Group::reformatMatchResTableDoc(const std::string &pathMatchTab, const std::string &groupAttribute, const Graph &semanticGraph,
                                     const std::unordered_map<std::string, std::vector<double>> &doc2Vec)
{
    CSVReader reader;
    reader.reading_one_table(pathMatchTab, false);

    Table matchRes = reader.tables[0];
    matchRes.Profile();

    const std::string lSchema = "ltable_" + groupAttribute;
    const std::string rschema = "rtable_" + groupAttribute;
    ui lpos = matchRes.inverted_schema.at(lSchema);
    ui rpos = matchRes.inverted_schema.at(rschema);

    for(auto &row : matchRes.rows) {
        std::string lStr = row[lpos];
        std::string rStr = row[rpos];
        bool lHasCandidate = !semanticGraph.isVertexIsolated(lStr);
        bool rHasCandidate = !semanticGraph.isVertexIsolated(rStr);

        if(lHasCandidate && rHasCandidate) {
            auto stringPair = semanticGraph.retrieveMostSimilarNeighborsDoc(lStr, rStr);
            row[lpos] = stringPair.first;
            row[rpos] = stringPair.second;
        }
        else if(lHasCandidate && !rHasCandidate) {
            row[lpos] = semanticGraph.retrieveMostSimilarNeighborsDoc(lStr, doc2Vec.at(rStr));
        }
        else if(!lHasCandidate && rHasCandidate) {
            row[rpos] = semanticGraph.retrieveMostSimilarNeighborsDoc(rStr, doc2Vec.at(lStr));
        }
    }

    MultiWriter::writeOneTable(matchRes, pathMatchTab);
}


void Group::reformatMatchResTableWord(const std::string &pathMatchTab, const std::string &groupAttribute, const Graph &semanticGraph,
                                      const std::unordered_map<std::string, std::vector<std::vector<double>>> &doc2Vec)
{
    CSVReader reader;
    reader.reading_one_table(pathMatchTab, false);

    Table matchRes = reader.tables[0];
    matchRes.Profile();

    const std::string lSchema = "ltable_" + groupAttribute;
    const std::string rschema = "rtable_" + groupAttribute;
    ui lpos = matchRes.inverted_schema.at(lSchema);
    ui rpos = matchRes.inverted_schema.at(rschema);

    for(auto &row : matchRes.rows) {
        std::string lStr = row[lpos];
        std::string rStr = row[rpos];
        bool lHasCandidate = !semanticGraph.isVertexIsolated(lStr);
        bool rHasCandidate = !semanticGraph.isVertexIsolated(rStr);

        if(lHasCandidate && rHasCandidate) {
            auto stringPair = semanticGraph.retrieveMostSimilarNeighborsWord(lStr, rStr);
            row[lpos] = stringPair.first;
            row[rpos] = stringPair.second;
        }
        else if(lHasCandidate && !rHasCandidate) {
            row[lpos] = semanticGraph.retrieveMostSimilarNeighborsWord(lStr, doc2Vec.at(rStr));
        }
        else if(!lHasCandidate && rHasCandidate) {
            row[rpos] = semanticGraph.retrieveMostSimilarNeighborsWord(rStr, doc2Vec.at(lStr));
        }
    }

    MultiWriter::writeOneTable(matchRes, pathMatchTab);
}


void Group::reformatTableByInterchangeableValuesByGraph(const std::string &groupAttribute, double groupTau, bool isTransitiveClosure, 
                                                        const std::string &defaultICVDir, const std::string &defaultMatchResDir)
{
    std::cout << "group interchangeable values on attribute : " << groupAttribute 
        << "\tdefault directory for output : " << defaultICVDir
        << "\tthreshold : " << groupTau
        << "\tis transitive closure enabled : " << isTransitiveClosure << std::endl;

    // io
    std::vector<std::string> docs, negDocs;
    DocEmbeddings vecs, negVecs;
    std::vector<std::pair<std::string, std::string>> candidates;

    Group::readDocsAndVecs(docs, vecs, defaultICVDir);
    Group::readDocsAndVecs(negDocs, negVecs, defaultICVDir, "vec_interchangeable_neg.txt");
    Group::readDocCandidatePairs(candidates, defaultICVDir);

    // build
    Graph senmaticGraph(isTransitiveClosure, groupTau);
    senmaticGraph.buildSemanticGraph(docs, vecs, candidates);

    // write
    std::string directory = getICVDir(defaultICVDir);
    std::string pathGraph = directory + "interchangeable_graph_" 
                            + groupAttribute + ".txt";
    senmaticGraph.writeSemanticGraph(pathGraph);

    // refactor
    std::string matchDirectory = getNegMatchDir(defaultMatchResDir);
    std::string pathNegMatchTab = matchDirectory + "neg_match_res0.csv";
    reformatMatchResTableDoc(pathNegMatchTab, groupAttribute, senmaticGraph, Group::saveNegEmbeddings(negDocs, negVecs));
}


void Group::reformatTableByInterchangeableValuesByWordGraph(const std::string &groupAttribute, double groupTau, bool isTransitiveClosure, 
                                                            const std::string &defaultICVDir, const std::string &defaultMatchResDir)
{
    std::cout << "group interchangeable values on attribute : " << groupAttribute 
        << "\tdefault directory for output : " << defaultICVDir
        << "\tthreshold : " << groupTau
        << "\tis transitive closure enabled : " << isTransitiveClosure << std::endl;

    // io
    std::vector<std::string> docs, negDocs;
    WordEmbeddings vecs, negVecs;
    std::vector<std::pair<std::string, std::string>> candidates;

    Group::readWordEmbeddingDocsAndVecs(docs, vecs, defaultICVDir);
    Group::readWordEmbeddingDocsAndVecs(negDocs, negVecs, defaultICVDir, "vec_interchangeable_neg.txt");
    Group::readDocCandidatePairs(candidates, defaultICVDir);

    // build
    Graph senmaticGraph(isTransitiveClosure, groupTau);
    senmaticGraph.buildSemanticGraph(docs, vecs, candidates);

    // write
    std::string directory = getICVDir(defaultICVDir);
    std::string pathGraph = directory + "interchangeable_graph_" 
                            + groupAttribute + ".txt";
    senmaticGraph.writeSemanticGraph(pathGraph);

    // refactor
    std::string matchDirectory = getNegMatchDir(defaultMatchResDir);
    std::string pathNegMatchTab = matchDirectory + "neg_match_res0.csv";
    reformatMatchResTableWord(pathNegMatchTab, groupAttribute, senmaticGraph, Group::saveNegWordEmbeddings(negDocs, negVecs));
}


extern "C"
{
    void group_interchangeable_values_by_graph(const char *group_attribute, const char *group_strategy, 
                                               double group_tau, bool is_transitive_closure, 
                                               const char *default_icv_dir) {
        // Group::groupInterchangeableValuesByGraph(group_attribute, group_strategy, group_tau, is_transitive_closure, 
        //                                          default_icv_dir);
        Group::groupInterchangeableValuesByWordGraph(group_attribute, group_strategy, group_tau, is_transitive_closure, 
                                                     default_icv_dir);
    }

    void refactor_neg_match_res_by_graph(const char *group_attribute, double group_tau, bool is_transitive_closure, 
                                         const char *default_icv_dir, const char *default_match_res_dir) {
        Group::reformatTableByInterchangeableValuesByWordGraph(group_attribute, group_tau, is_transitive_closure, 
                                                               default_icv_dir, default_match_res_dir);
    }

    void group_interchangeable_values_by_cluster() {
        std::cerr << "not established" << std::endl;
        exit(1);
    }
}


/*
 * check if vectors are sorted before set_intersection
 */