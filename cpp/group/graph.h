/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>


/*
 * semantic similarity graph
 */
class Graph
{
public:
    bool isTransitiveClosure{false};
    double tau{0.0};

    int numVertex{0};
    int numEdge{0};

    // docs & tokens
    std::vector<std::string> docs;
    std::vector<std::vector<std::string>> docsDlm; // dlm_dc0
    std::vector<std::vector<std::string>> docsQgm; // qgm_3

    std::vector<std::vector<double>> vecs; // semantic similarity
    std::unordered_map<std::string, int> doc2Id;
    std::vector<std::vector<int>> graLists;          

public:
    Graph() = default;
    explicit Graph(bool _isTransitiveClosure, double _tau) 
    : isTransitiveClosure(_isTransitiveClosure), tau(_tau) { }
    ~Graph() = default;

    // for allocate vectors
    // Graph(const Graph &other) = delete;
    // Graph(Graph &&other) = delete;

private:
    bool isDocContained(const std::string &doc) const;

    double calculateCosineSim(const std::vector<double> &lhs, const std::vector<double> &rhs);

    void readVertex(std::string info, int &id, std::string &doc);
    void readEdge(std::string info, int &from, int &to);

public:
    // currently we only check the two-hop neighbors
    // that is if we deduce x ~ z according to the transitive closure
    // we do not perform further check for z's neighbors for x
    void buildSemanticGraph(const std::vector<std::string> &_docs, const std::vector<std::vector<double>> &_vecs, 
                            const std::vector<std::pair<std::string, std::string>> &candidates);

    void buildSemanticGraph(const std::string &pathGraph);
    void updateTokenizedDocs(const std::vector<std::vector<std::string>> &dlm, 
                             const std::vector<std::vector<std::string>> &qgm);

    // as the graph is sparse, we do not perform binary search
    bool checkEdgeExistence(int u, int v) const;
    bool checkEdgeExistence(const std::string &u, const std::string &v) const;

    bool isVertexIsolated(const std::string &str) const;

    void printMetaData() const;

    void writeSemanticGraph(const std::string &pathGraph);

    // retrieve one-hop neighbors
    void retrieveNeighbors(const std::string &doc, std::vector<std::string> &neighbors) const;
    void retrieveTokenizedNeighbors(const std::string &doc, const std::string &type,
                                    std::vector<std::vector<std::string>> &neighbors) const;
};

#endif // _GRAPH_H_