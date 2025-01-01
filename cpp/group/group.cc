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


void Group::updateNegEmbeddings(const std::vector<std::string> &docs, const DocEmbeddings &vecs, 
                                std::unordered_map<std::string, std::vector<double>> &doc2Vec)
{
    for(size_t i = 0; i < docs.size(); i++) {
        if(doc2Vec.find(docs[i]) != doc2Vec.end())
            continue;
        doc2Vec[docs[i]] = vecs[i];
    }
}


void Group::updateNegWordEmbeddings(const std::vector<std::string> &docs, const WordEmbeddings &vecs, 
                                    std::unordered_map<std::string, std::vector<std::vector<double>>> &doc2Vec)
{
    for(size_t i = 0; i < docs.size(); i++) {
        if(doc2Vec.find(docs[i]) != doc2Vec.end())
            continue;
        doc2Vec[docs[i]] = vecs[i];
    }
}


double Group::calculateCosineSim(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    assert(lhs.size() == rhs.size());
    size_t size = lhs.size();

    double dot = 0.0;
    double lNorm = 0.0;
    double rNorm = 0.0;

    for(size_t idx = 0; idx < size; idx++) {
        dot += lhs[idx] * rhs[idx];
        lNorm += lhs[idx] * lhs[idx];
        rNorm += rhs[idx] * rhs[idx];
    }

    lNorm = sqrt(lNorm);
    rNorm = sqrt(rNorm);

    return dot / (lNorm * rNorm);
}


double Group::calculateCoherentFactor(const std::vector<std::vector<double>> &lhs, const std::vector<std::vector<double>> &rhs)
{
    std::vector<std::vector<double>> unionVecs;
    __gnu_parallel::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), 
                              std::back_inserter(unionVecs));

    size_t X = unionVecs.size();
    double FX = 0.0;

    for(size_t i = 0; i < X; i++)
        for(size_t j = i + 1; j < X; j++)
            FX += calculateCosineSim(unionVecs[i], unionVecs[j]);

    FX /= (X * 1.0);

    return FX;
}


Table Group::slimTab(const Table &tab, ui workIdCol, ui queryIdCol, ui workCol, ui queryCol, 
                     const std::unordered_map<std::string, std::vector<double>> &doc2Vec)
{
    Table slimTab;
    std::unordered_map<int, std::vector<std::pair<int, int>>> buck;
    std::unordered_map<int, std::string> workId2Value, queryId2Value;

    for(int i = 0; i < tab.row_no; i++) {
        auto workId = std::stoi(tab.rows[i][workIdCol]);
        auto queryId = std::stoi(tab.rows[i][queryIdCol]);
        workId2Value[workId] = tab.rows[i][workCol];
        queryId2Value[queryId] = tab.rows[i][queryCol];
        buck[workId].emplace_back(queryId, i);
    }

    slimTab.copySchema(tab);

    for(const auto &p : buck) {
        const auto &workId = p.first;
        const auto &workVec = doc2Vec.at(workId2Value.at(workId));
        const auto &queryList = p.second;

        if(queryList.size() == 1) {
            slimTab.insertOneRow(tab.rows[queryList[0].second]);
            continue;
        }

        double maxSim = -19260817.0;
        int maxIdx = -1;

        for(const auto &q : queryList) {
            const auto &queryId = q.first;
            const auto &queryVec = doc2Vec.at(queryId2Value.at(queryId));
            double sim = calculateCosineSim(workVec, queryVec);
            if(sim > maxSim) {
                maxSim = sim;
                maxIdx = q.second;
            }
        }

        slimTab.insertOneRow(tab.rows[maxIdx]);
    }

    slimTab.Profile();

    return slimTab;
}


Table Group::slimTab(const Table &tab, ui workIdCol, ui queryIdCol, ui workCol, ui queryCol, 
                     const std::unordered_map<std::string, std::vector<std::vector<double>>> &doc2Vec)
{
    Table slimTab;
    std::unordered_map<int, std::vector<std::pair<int, int>>> buck;
    std::unordered_map<int, std::string> workId2Value, queryId2Value;

    for(int i = 0; i < tab.row_no; i++) {
        auto workId = std::stoi(tab.rows[i][workIdCol]);
        auto queryId = std::stoi(tab.rows[i][queryIdCol]);
        workId2Value[workId] = tab.rows[i][workCol];
        queryId2Value[queryId] = tab.rows[i][queryCol];
        buck[workId].emplace_back(queryId, i);
    }

    slimTab.copySchema(tab);

    for(const auto &p : buck) {
        const auto &workId = p.first;
        // if(doc2Vec.find(workId2Value.at(workId)) == doc2Vec.end()) {
        //     std::cerr << "error: " << workId2Value.at(workId) << std::endl;
        //     exit(1);
        // }
        const auto &workVec = doc2Vec.at(workId2Value.at(workId));
        const auto &queryList = p.second;

        if(queryList.size() == 1) {
            slimTab.insertOneRow(tab.rows[queryList[0].second]);
            continue;
        }

        double maxSim = -19260817.0;
        int maxIdx = -1;

        for(const auto &q : queryList) {
            const auto &queryId = q.first;
            // if(doc2Vec.find(queryId2Value.at(queryId)) == doc2Vec.end()) {
            //     std::cerr << "error: " << queryId2Value.at(queryId) << std::endl;
            //     exit(1);
            // }
            const auto &queryVec = doc2Vec.at(queryId2Value.at(queryId));
            double sim = calculateCoherentFactor(workVec, queryVec);
            if(sim > maxSim) {
                maxSim = sim;
                maxIdx = q.second;
            }
        }

        slimTab.insertOneRow(tab.rows[maxIdx]);
    }

    slimTab.Profile();

    return slimTab;
}


Table Group::slimTab(const Table &tab, ui workCol, ui queryCol, ui K)
{
    Table slimTab;
    std::vector<std::pair<int, double>> buck;

    for(int i = 0; i < tab.row_no; i++) {
        auto workAttr = tab.rows[i][workCol];   
        auto queryAttr = tab.rows[i][queryCol];
        std::vector<std::string> workTokens, queryTokens, tmp;
        Tokenizer::string2TokensDlm(workAttr, workTokens, " \"\',\\\t\r\n");
        Tokenizer::string2TokensDlm(queryAttr, queryTokens, " \"\',\\\t\r\n");
        std::sort(workTokens.begin(), workTokens.end());    
        std::sort(queryTokens.begin(), queryTokens.end());
        std::set_intersection(workTokens.begin(), workTokens.end(), queryTokens.begin(), queryTokens.end(), 
                              std::back_inserter(tmp));
        double sim = tmp.size() * 1.0 / (workTokens.size() + queryTokens.size() - tmp.size());
        buck.emplace_back(i, sim);
    }

    slimTab.copySchema(tab);

    std::sort(buck.begin(), buck.end(), [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) {
        return lhs.second > rhs.second;
    });

    ui selectedSize = std::min(K, (ui)buck.size());

    for(ui i = 0; i < selectedSize; i++)
        slimTab.insertOneRow(tab.rows[buck[i].first]);

    slimTab.Profile();

    return slimTab;
}


std::unordered_map<int, std::string> Group::getOriginalValue(const std::string &pathTab, const std::string &attr)
{
    CSVReader reader;
    reader.reading_one_table(pathTab, false);

    Table tab = reader.tables[0];
    tab.Profile();

    ui pos = tab.inverted_schema.at(attr);
    ui idPos = tab.inverted_schema.at("id");

    std::unordered_map<int, std::string> id2Value;
    for(int i = 0; i < tab.row_no; i++)
        id2Value[std::stoi(tab.rows[i][idPos])] = tab.rows[i][pos];

    return id2Value;
}


Table Group::restoreTab(const std::string &pathMatchTab, const std::unordered_map<int, std::string> &id2ValueA, 
                        const std::unordered_map<int, std::string> &id2ValueB, const std::string &attr)
{
    CSVReader reader;
    reader.reading_one_table(pathMatchTab, false);

    Table tab = reader.tables[0];
    tab.Profile();

    ui lAttrPos = tab.inverted_schema.at("ltable_" + attr);
    ui rAttrPos = tab.inverted_schema.at("rtable_" + attr);
    ui lIdPos = tab.inverted_schema.at("ltable_id");
    ui rIdPos = tab.inverted_schema.at("rtable_id");

    for(int i = 0; i < tab.row_no; i++) {
        tab.rows[i][lAttrPos] = id2ValueA.at(std::stoi(tab.rows[i][lIdPos]));
        tab.rows[i][rAttrPos] = id2ValueB.at(std::stoi(tab.rows[i][rIdPos]));
    }

    return tab;
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
    directory = defaultMatchResDir == "" ? directory + "../../output/match_res/" 
                                         : (defaultMatchResDir.back() == '/' ? defaultMatchResDir : defaultMatchResDir + "/");

    return directory;
}


std::string Group::getBufferDir(const std::string &defaultBufferDir)
{
    std::string fullPath = __FILE__;
    size_t lastSlash = fullPath.find_last_of("/\\");
    std::string directory = fullPath.substr(0, lastSlash + 1);
    directory = defaultBufferDir == "" ? directory + "../../output/buffer/" 
                                       : (defaultBufferDir.back() == '/' ? defaultBufferDir : defaultBufferDir + "/");

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
    ui lIdPos = matchRes.inverted_schema.at("ltable_id");
    ui rIdPos = matchRes.inverted_schema.at("rtable_id");

    // matchRes = slimTab(matchRes, lIdPos, rIdPos, lpos, rpos, doc2Vec);
    // matchRes = slimTab(matchRes, rIdPos, lIdPos, rpos, lpos, doc2Vec);

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
    ui lIdPos = matchRes.inverted_schema.at("ltable_id");
    ui rIdPos = matchRes.inverted_schema.at("rtable_id");

    // matchRes = slimTab(matchRes, lIdPos, rIdPos, lpos, rpos, doc2Vec);
    // matchRes = slimTab(matchRes, rIdPos, lIdPos, rpos, lpos, doc2Vec);

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


void Group::slimMatchResDoc(const std::string &pathMatchTab, const std::string &groupAttribute, const std::string &defaultICVDir, 
                            const std::string &defaultBufferDir, int isNeg)
{
    std::vector<std::string> docs;
    DocEmbeddings vecs;
    std::string vecFile = isNeg > 0 ? "vec_interchangeable_neg.txt" : "vec_interchangeable.txt";
    Group::readDocsAndVecs(docs, vecs, defaultICVDir, vecFile);

    const std::string bufferDir = getBufferDir(defaultBufferDir);
    auto id2ValueA = getOriginalValue(bufferDir + "clean_A.csv", groupAttribute);
    auto id2ValueB = getOriginalValue(bufferDir + "clean_B.csv", groupAttribute);

    auto matchRes = restoreTab(pathMatchTab, id2ValueA, id2ValueB, groupAttribute);

    ui lPos = matchRes.inverted_schema.at("ltable_" + groupAttribute);
    ui rPos = matchRes.inverted_schema.at("rtable_" + groupAttribute);
    ui lIdPos = matchRes.inverted_schema.at("ltable_id");
    ui rIdPos = matchRes.inverted_schema.at("rtable_id");

    auto embeddings = Group::saveNegEmbeddings(docs, vecs);
    if(isNeg > 1) {
        std::vector<std::string> docs2;
        DocEmbeddings vecs2;
        Group::readDocsAndVecs(docs2, vecs2, defaultICVDir, "vec_interchangeable.txt");
        updateNegEmbeddings(docs2, vecs2, embeddings);
    }

    matchRes = slimTab(matchRes, lIdPos, rIdPos, lPos, rPos, embeddings);
    matchRes = slimTab(matchRes, rIdPos, lIdPos, rPos, lPos, embeddings);

    MultiWriter::writeOneTable(matchRes, pathMatchTab);
}


void Group::slimMatchResWord(const std::string &pathMatchTab, const std::string &groupAttribute, const std::string &defaultICVDir, 
                             const std::string &defaultBufferDir, int isNeg)
{
    std::vector<std::string> docs;
    WordEmbeddings vecs;
    std::string vecFile = isNeg > 0 ? "vec_interchangeable_neg.txt" : "vec_interchangeable.txt";
    Group::readWordEmbeddingDocsAndVecs(docs, vecs, defaultICVDir, vecFile);

    // const std::string bufferDir = getBufferDir(defaultBufferDir);
    // auto id2ValueA = getOriginalValue(bufferDir + "clean_A.csv", groupAttribute);
    // auto id2ValueB = getOriginalValue(bufferDir + "clean_B.csv", groupAttribute);

    // auto matchRes = restoreTab(pathMatchTab, id2ValueA, id2ValueB, groupAttribute);

    CSVReader reader;
    reader.reading_one_table(pathMatchTab, false);
    auto matchRes = reader.tables[0];

    ui lPos = matchRes.inverted_schema.at("ltable_" + groupAttribute);
    ui rPos = matchRes.inverted_schema.at("rtable_" + groupAttribute);
    ui lIdPos = matchRes.inverted_schema.at("ltable_id");
    ui rIdPos = matchRes.inverted_schema.at("rtable_id");

    auto embeddings = Group::saveNegWordEmbeddings(docs, vecs);
    if(isNeg > 1) {
        std::vector<std::string> docs2;
        WordEmbeddings vecs2;
        Group::readWordEmbeddingDocsAndVecs(docs2, vecs2, defaultICVDir, "vec_interchangeable.txt");
        updateNegWordEmbeddings(docs2, vecs2, embeddings);
    }

    matchRes = slimTab(matchRes, lIdPos, rIdPos, lPos, rPos, embeddings);
    matchRes = slimTab(matchRes, rIdPos, lIdPos, rPos, lPos, embeddings);

    MultiWriter::writeOneTable(matchRes, pathMatchTab);
}


void Group::slimMatchResSynatic(const std::string &pathMatchTab, const std::string &groupAttribute, ui K)
{
    CSVReader reader;
    reader.reading_one_table(pathMatchTab, false);
    auto matchRes = reader.tables[0];

    ui lPos = matchRes.inverted_schema.at("ltable_" + groupAttribute);
    ui rPos = matchRes.inverted_schema.at("rtable_" + groupAttribute);

    matchRes = slimTab(matchRes, lPos, rPos, K);

    MultiWriter::writeOneTable(matchRes, pathMatchTab);
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

    void slim_refactored_match_res_by_graph(const char *path_match_tab, const char *group_attribute, 
                                            const char *default_icv_dir, const char *default_buffer_dir, 
                                            int if_neg) {
        Group::slimMatchResWord(path_match_tab, group_attribute, default_icv_dir, default_buffer_dir, if_neg);
    }

    void slim_refactored_match_res_by_synatic(const char *path_match_tab, const char *group_attribute, unsigned int K) {
        Group::slimMatchResSynatic(path_match_tab, group_attribute, K);
    }

    void group_interchangeable_values_by_cluster() {
        std::cerr << "not established" << std::endl;
        exit(1);
    }
}


/*
 * check if vectors are sorted before set_intersection
 */