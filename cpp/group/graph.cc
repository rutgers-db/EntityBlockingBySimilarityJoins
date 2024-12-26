/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "group/graph.h"


bool Graph::isDocContained(const std::string &doc) const
{
    return doc2Id.find(doc) != doc2Id.end();
}


double Graph::calculateCosineSim(const std::vector<double> &lhs, const std::vector<double> &rhs) const
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


double Graph::calculateCoherentFactor(const std::vector<std::vector<double>> &lhs, 
                                      const std::vector<std::vector<double>> &rhs) const
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


void Graph::readVertex(std::string info, int &id, std::string &doc)
{
    std::istringstream iss(info);
    std::string token;
    // char delim = ' ';

    // char
    getline(iss, token, ' ');
    // id
    getline(iss, token, ' ');
    id = std::stoi(token);
    // doc
    getline(iss, token);
    doc = token;
};


void Graph::readEdge(std::string info, int &from, int &to)
{
    std::istringstream iss(info);
    std::string token;
    // char delim = ' ';

    // char
    getline(iss, token, ' ');
    // from
    getline(iss, token, ' ');
    from = std::stoi(token);
    // to
    getline(iss, token);
    to = std::stoi(token);
}


void Graph::buildSemanticGraph(const std::vector<std::string> &_docs, const DocEmbedding &_vecs, 
                               const std::vector<std::pair<std::string, std::string>> &candidates)
{
    // nodes
    docs = _docs;
    vecs = _vecs;

    int count = 0;
    for(const auto &doc : docs)
        doc2Id[doc] = count ++;

    // stat
    numVertex = count;

    // edges
    graLists.resize(count, std::vector<int> ());

    for(const auto &p : candidates) {
        int lId = doc2Id.at(p.first);
        int rId = doc2Id.at(p.second); 

        if(lId == rId || checkEdgeExistence(lId, rId))
            continue;

        double cos = calculateCosineSim(vecs[lId], vecs[rId]);
        // std::cout << cos << std::endl;
        if(cos >= tau) {
            graLists[lId].emplace_back(rId);
            graLists[rId].emplace_back(lId);
            ++ numEdge;
        }
    }

    // transitive closure
    if(isTransitiveClosure) {
        std::cout << "adding edges" << std::endl;
        for(int i = 0; i < numVertex; i++) {
            // lazy modify
            std::vector<int> lazyNeighb;

            // check two-hop neighbors of i in order
            // a ~ b, b ~ c, thus a ~ c if sim(a, c) > tau
            for(const auto &v : graLists[i]) {
                // if(v < i)
                //     continue;
                for(const auto &u : graLists[v]) {
                    if(u <= i || checkEdgeExistence(i, u))
                        continue;
                    double cos = calculateCosineSim(vecs[i], vecs[u]);
                    if(cos >= tau) {
                        lazyNeighb.emplace_back(u);
                        // ++ numEdge;
                    }
                }
            }

            // add
            std::sort(lazyNeighb.begin(), lazyNeighb.end());
            auto iter = std::unique(lazyNeighb.begin(), lazyNeighb.end());
            lazyNeighb.resize(std::distance(lazyNeighb.begin(), iter));
            for(const auto &u : lazyNeighb) {
                graLists[i].emplace_back(u);
                graLists[u].emplace_back(i);
                ++ numEdge;
            }
        }
    }

    int tmpc = 0;
    for(auto &edges : graLists) {
        std::sort(edges.begin(), edges.end());
        auto iter = std::unique(edges.begin(), edges.end());
        if(iter != edges.end()) {
            std::cerr << "duplicate edge in semantic graph : " << tmpc << " " << docs[tmpc] << std::endl;
            exit(1);
        }
        if(std::count(edges.begin(), edges.end(), tmpc) > 0) {
            std::cerr << "self circle in semantic graph : " << tmpc << " " << docs[tmpc] << std::endl;
            exit(1);
        }
        ++ tmpc;
    }
}


void Graph::buildSemanticGraph(const std::vector<std::string> &_docs, const WordEmbedding &_vecs, 
                               const std::vector<std::pair<std::string, std::string>> &candidates)
{
    // nodes
    docs = _docs;
    wordVecs = _vecs;

    int count = 0;
    for(const auto &doc : docs)
        doc2Id[doc] = count ++;

    // stat
    numVertex = count;

    // edges
    graLists.resize(count, std::vector<int> ());

    for(const auto &p : candidates) {
        int lId = doc2Id.at(p.first);
        int rId = doc2Id.at(p.second); 

        if(lId == rId || checkEdgeExistence(lId, rId))
            continue;

        // coherent group
        const auto &lEmbedding = wordVecs[lId];
        const auto &rEmbedding = wordVecs[rId];
        double FX = calculateCoherentFactor(lEmbedding, rEmbedding);

        if(FX >= tau) {
            graLists[lId].emplace_back(rId);
            graLists[rId].emplace_back(lId);
            ++ numEdge;
        }
    }

    // transitive closure
    if(isTransitiveClosure) {
        std::cout << "adding edges" << std::endl;
        for(int i = 0; i < numVertex; i++) {
            // lazy modify
            std::vector<int> lazyNeighb;

            // check two-hop neighbors of i in order
            // a ~ b, b ~ c, thus a ~ c if sim(a, c) > tau
            for(const auto &v : graLists[i]) {
                for(const auto &u : graLists[v]) {
                    if(u <= i || checkEdgeExistence(i, u))
                        continue;

                    const auto &lEmbedding = wordVecs[i];
                    const auto &rEmbedding = wordVecs[u];
                    double FX = calculateCoherentFactor(lEmbedding, rEmbedding);
                    
                    if(FX >= tau)
                        lazyNeighb.emplace_back(u);
                }
            }

            // add
            std::sort(lazyNeighb.begin(), lazyNeighb.end());
            auto iter = std::unique(lazyNeighb.begin(), lazyNeighb.end());
            lazyNeighb.resize(std::distance(lazyNeighb.begin(), iter));
            for(const auto &u : lazyNeighb) {
                graLists[i].emplace_back(u);
                graLists[u].emplace_back(i);
                ++ numEdge;
            }
        }
    }

    int tmpc = 0;
    for(auto &edges : graLists) {
        std::sort(edges.begin(), edges.end());
        auto iter = std::unique(edges.begin(), edges.end());
        if(iter != edges.end()) {
            std::cerr << "duplicate edge in semantic graph : " << tmpc << " " << docs[tmpc] << std::endl;
            exit(1);
        }
        if(std::count(edges.begin(), edges.end(), tmpc) > 0) {
            std::cerr << "self circle in semantic graph : " << tmpc << " " << docs[tmpc] << std::endl;
            exit(1);
        }
        ++ tmpc;
    }
}


void Graph::buildSemanticGraph(const std::string &pathGraph)
{
    std::ifstream streamGraph(pathGraph.c_str(), std::ios::in);

    std::string header;
    getline(streamGraph, header);
    std::istringstream iss(header);
    std::string token;
    // |V|
    getline(iss, token, ' ');
    // std::cout << token << std::endl << std::flush;
    numVertex = std::stoi(token);
    // |E|
    getline(iss, token, ' ');
    // std::cout << token << std::endl << std::flush;
    numEdge = std::stoi(token);

    for(int i = 0; i < numVertex; i++) {
        std::string info;
        getline(streamGraph, info);
        // parse
        int id = 0;
        std::string doc;
        readVertex(info, id, doc);
        docs.emplace_back(doc);
        doc2Id[doc] = id;
    }

    graLists.resize(numVertex, std::vector<int> ());
    for(int i = 0; i < numEdge; i++) {
        std::string info;
        getline(streamGraph, info);
        // parse
        int from = 0, to = 0;
        readEdge(info, from, to);
        graLists[from].emplace_back(to);
        graLists[to].emplace_back(from);
    }

    for(auto &edge : graLists)
        std::sort(edge.begin(), edge.end());
}


void Graph::updateTokenizedDocs(const std::vector<std::vector<std::string>> &dlm, 
                                const std::vector<std::vector<std::string>> &qgm)
{
    docsDlm = dlm;
    docsQgm = qgm;
}


bool Graph::checkEdgeExistence(int u, int v) const
{
    assert(u < numVertex && v < numVertex);
    // if(u >= numVertex || v >= numVertex) {
    //     std::cerr << "error in : " << u << " " << v << " " << numVertex << std::endl;
    //     exit(1);
    // }
    return std::count(graLists[u].begin(), graLists[u].end(), v) > 0;
}


bool Graph::checkEdgeExistence(const std::string &u, const std::string &v) const
{
    if(!isDocContained(u) || !isDocContained(v))
        return false;

    int uId = doc2Id.at(u);
    int vId = doc2Id.at(v);
    return std::count(graLists[uId].begin(), graLists[uId].end(), vId) > 0;
}


bool Graph::isVertexIsolated(const std::string &str) const
{
    if(!isDocContained(str))
        return true;

    int id = doc2Id.at(str);
    return graLists[id].size() == 0;
}


void Graph::printMetaData() const 
{
    std::cout << "|V| : " << numVertex << " |E| : " << numEdge << std::endl;
    std::cout << " is transitive colsure : " << isTransitiveClosure << " tau : " << tau << std::endl; 
}


void Graph::writeSemanticGraph(const std::string &pathGraph)
{
    std::ofstream graphStream(pathGraph.c_str(), std::ios::out);
    graphStream << numVertex << " " << numEdge << std::endl;

    // vertices
    for(int i = 0; i < numVertex; i++) {
        graphStream << "v " << i << " " << docs[i] << std::endl;
        // for(const auto &v : vecs[i])
        //     graphStream << v << " ";
        // graphStream << std::endl;
    }

    // edges
    int tmpc = 0;
    for(int i = 0; i < numVertex; i++) {
        for(const auto &to : graLists[i]) {
            if(to < i)
                continue;
            graphStream << "e " << i << " " << to << std::endl; 
        }
    }

    graphStream << std::flush;
}


void Graph::retrieveNeighbors(const std::string &doc, std::vector<std::string> &neighbors) const
{
    if(!isDocContained(doc)) {
        std::cerr << "no such key : " << doc << std::endl;
        exit(1);
    }

    int id = doc2Id.at(doc);
    for(const auto &to : graLists[id]) 
        neighbors.emplace_back(docs[to]);
}


void Graph::retrieveTokenizedNeighbors(const std::string &doc, const std::string &type,
                                       std::vector<std::vector<std::string>> &neighbors) const
{
    if(!isDocContained(doc)) {
        std::cerr << "no such key : " << doc << std::endl;
        exit(1);
    }

    if(type != "dlm" && type != "qgm") {
        std::cerr << "no such tokenizer type : " << type << std::endl;
        exit(1);
    }

    int id = doc2Id.at(doc);
    for(const auto &to : graLists[id]) {
        if(type == "dlm")
            neighbors.emplace_back(docsDlm[to]);
        else
            neighbors.emplace_back(docsQgm[to]);
    }
}


std::string Graph::retrieveMostSimilarNeighborsDoc(const std::string &workDoc, const std::vector<double> &queryVec) const 
{
    if(!isDocContained(workDoc)) {
        std::cerr << "no such key : " << workDoc << std::endl;
        exit(1);
    }

    DocEmbedding neighbors_;
    int id = doc2Id.at(workDoc);
    for(const auto &to : graLists[id])
        neighbors_.emplace_back(vecs[to]);

    double maxSim = 0.0;
    std::string maxDoc = "";
    for(size_t i = 0; i < neighbors_.size(); i++) {
        double sim = calculateCosineSim(neighbors_[i], queryVec);
        if(sim > maxSim) {
            maxSim = sim;
            maxDoc = docs[graLists[id][i]];
        }
    }

    return maxDoc;
}


std::pair<std::string, std::string> Graph::retrieveMostSimilarNeighborsDoc(const std::string &lhsDoc, const std::string &rhsDoc) const
{
    if(!isDocContained(lhsDoc) || !isDocContained(rhsDoc)) {
        std::cerr << "no such key : " << lhsDoc << " " << rhsDoc << std::endl;
        exit(1);
    }

    int lId = doc2Id.at(lhsDoc);
    int rId = doc2Id.at(rhsDoc);

    DocEmbedding lNeighbors_;
    for(const auto &to : graLists[lId])
        lNeighbors_.emplace_back(vecs[to]);

    DocEmbedding rNeighbors_;
    for(const auto &to : graLists[rId])
        rNeighbors_.emplace_back(vecs[to]);

    double maxSim = 0.0;
    std::string maxLDoc = "";
    std::string maxRDoc = "";
    for(size_t i = 0; i < lNeighbors_.size(); i++) {
        for(size_t j = 0; j < rNeighbors_.size(); j++) {
            double sim = calculateCosineSim(lNeighbors_[i], rNeighbors_[j]);
            if(sim > maxSim) {
                maxSim = sim;
                maxLDoc = docs[graLists[lId][i]];
                maxRDoc = docs[graLists[rId][j]];
            }
        }
    }

    return std::make_pair(maxLDoc, maxRDoc);
}


std::string Graph::retrieveMostSimilarNeighborsWord(const std::string &workDoc, const std::vector<std::vector<double>> &queryVec) const
{
    if(!isDocContained(workDoc)) {
        std::cerr << "no such key : " << workDoc << std::endl;
        exit(1);
    }

    WordEmbedding neighbors_;
    int id = doc2Id.at(workDoc);
    for(const auto &to : graLists[id])
        neighbors_.emplace_back(wordVecs[to]);

    double maxSim = 0.0;
    std::string maxDoc = "";
    for(size_t i = 0; i < neighbors_.size(); i++) {
        double sim = calculateCoherentFactor(neighbors_[i], queryVec);
        if(sim > maxSim) {
            maxSim = sim;
            maxDoc = docs[graLists[id][i]];
        }
    }

    return maxDoc;
}


std::pair<std::string, std::string> Graph::retrieveMostSimilarNeighborsWord(const std::string &lhsDoc, const std::string &rhsDoc) const
{
    if(!isDocContained(lhsDoc) || !isDocContained(rhsDoc)) {
        std::cerr << "no such key : " << lhsDoc << " " << rhsDoc << std::endl;
        exit(1);
    }

    int lId = doc2Id.at(lhsDoc);
    int rId = doc2Id.at(rhsDoc);

    WordEmbedding lNeighbors_;
    for(const auto &to : graLists[lId])
        lNeighbors_.emplace_back(wordVecs[to]);

    WordEmbedding rNeighbors_;
    for(const auto &to : graLists[rId])
        rNeighbors_.emplace_back(wordVecs[to]);

    double maxSim = 0.0;
    std::string maxLDoc = "";
    std::string maxRDoc = "";
    for(size_t i = 0; i < lNeighbors_.size(); i++) {
        for(size_t j = 0; j < rNeighbors_.size(); j++) {
            double sim = calculateCoherentFactor(lNeighbors_[i], rNeighbors_[j]);
            if(sim > maxSim) {
                maxSim = sim;
                maxLDoc = docs[graLists[lId][i]];
                maxRDoc = docs[graLists[rId][j]];
            }
        }
    }

    return std::make_pair(maxLDoc, maxRDoc);
}
