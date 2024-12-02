/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "group/graph.h"


double Graph::calculateCosineSim(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    std::vector<double> tmp;
    std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(tmp));

    return tmp.size() * 1.0 / std::sqrt(lhs.size() * rhs.size() * 1.0);
}


void Graph::readVertex(std::string info, int &id, std::string &doc)
{
    std::istringstream iss(info);
    std::string token;
    char delim = ' ';

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
    char delim = ' ';

    // char
    getline(iss, token, ' ');
    // from
    getline(iss, token, ' ');
    from = std::stoi(token);
    // to
    getline(iss, token);
    to = std::stoi(token);
}


void Graph::buildSemanticGraph(const std::vector<std::string> &_docs, const std::vector<std::vector<double>> &_vecs, 
                               const std::vector<std::pair<std::string, std::string>> &candidates)
{
    // nodes
    docs = _docs;
    vecs = _vecs;

    int count = 0;
    for(const auto &doc : docs)
        doc2Id[doc] = count ++;

    // edges
    graLists.resize(count, std::vector<int> ());
    for(const auto &p : candidates) {
        int lId = doc2Id[p.first];
        int rId = doc2Id[p.second]; 
        double cos = calculateCosineSim(vecs[lId], vecs[rId]);
        if(cos >= tau) {
            graLists[lId].emplace_back(rId);
            graLists[rId].emplace_back(lId);
            ++ numEdge;
        }
    }

    // stat
    numVertex = count;

    // transitive closure
    if(isTransitiveClosure) {
        std::cout << "adding edges" << std::endl;
        for(int i = 0; i < numVertex; i++) {
            // lazy modify
            std::vector<int> lazyNeighb;

            // check two-hop neighbors of i in order
            // a ~ b, b ~ c, thus a ~ c if sim(a, c) > tau
            for(const auto &v : graLists[i]) {
                if(v < i)
                    continue;
                for(const auto &u : graLists[v]) {
                    if(u <= i)
                        continue;
                    double cos = calculateCosineSim(vecs[i], vecs[u]);
                    if(cos >= tau) {
                        lazyNeighb.emplace_back(u);
                        ++ numEdge;
                    }
                }
            }

            // add
            for(const auto &u : lazyNeighb) {
                graLists[i].emplace_back(u);
                graLists[u].emplace_back(i);
            }
        }
    }

    int tmpc = 0;
    for(const auto &edges : graLists) {
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

    streamGraph >> numVertex >> numEdge;

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

    for(int i = 0; i < numEdge; i++) {
        std::string info;
        getline(streamGraph, info);
        // parse
        int from = 0, to = 0;
        readEdge(info, from, to);
        graLists[from].emplace_back(to);
        graLists[to].emplace_back(from);
    }

    for(const auto &edge : graLists)
        std::sort(edge.begin(), edge.end());
}


bool Graph::checkEdgeExistence(int u, int v) const
{
    return std::count(graLists[u].begin(), graLists[u].end(), v) > 0;
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
    for(int i = 0; i < numVertex; i++)
        graphStream << "v " << i << " " << docs[i] << std::endl;

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