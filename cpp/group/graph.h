/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>
#include <string>


/*
 * semantic similarity graph
 */
class Graph
{
public:
    std::vector<std::string> docs;
    std::vector<std::vector<double>> vecs;
    std::vector<std::vector<int>> graLists;

public:
    Graph() = default;
    ~Graph() = default;
    Graph(const Graph &other) = delete;
    Graph(Graph &&other) = delete;


};

#endif // _GRAPH_H_