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

    std::vector<std::string> docs;
    std::vector<std::vector<double>> vecs;
    std::unordered_map<std::string, int> doc2Id;
    std::vector<std::vector<int>> graLists;                 

public:
    Graph() = default;
    explicit Graph(bool _isTransitiveClosure, double _tau) 
    : isTransitiveClosure(_isTransitiveClosure), tau(_tau) { }
    ~Graph() = default;
    Graph(const Graph &other) = delete;
    Graph(Graph &&other) = delete;

private:
    double calculateCosineSim(const std::vector<double> &lhs, const std::vector<double> &rhs);

public:
    // currently we only check the two-hop neighbors
    // that is if we deduce x ~ z according to the transitive closure
    // we do not perform further check for z's neighbors for x
    void buildSemanticGraph(const std::vector<std::string> &_docs, const std::vector<std::vector<double>> &_vecs, 
                            const std::vector<std::pair<std::string, std::string>> &candidates);

    // as the graph is sparse, we do not perform binary search
    bool checkEdgeExistence(int u, int v) const;

    void printMetaData() const;

    void writeSemanticGraph(const std::string &pathGraph);
};

#endif // _GRAPH_H_