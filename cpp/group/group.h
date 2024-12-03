/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _GROUP_H_
#define _GROUP_H_

#include "group/graph.h"
#include <vector>
#include <string>
#include <fstream>


/*
 * externally group interchangeable values
 */
class Group
{
public:
    Group() = default;
    ~Group() = default;
    Group(const Group &other) = delete;
    Group(Group &&other) = delete;

private:
    // interchangeable values directory
    static std::string getICVDir(const std::string &defaultICVDir);

public:
    // io
    static void readDocsAndVecs(std::vector<std::string> &docs, std::vector<std::vector<double>> &vecs, 
                                const std::string &defaultICVDir = "");
    static void readDocCandidatePairs(std::vector<std::pair<std::string, std::string>> &candidates, 
                                      const std::string &defaultICVDir = "");

    // apis
    static void groupInterchangeableValuesByGraph(const std::string &groupAttribute, const std::string &groupStrategy, 
                                                  double groupTau, bool isTransitiveClosure, 
                                                  const std::string &defaultICVDir = "");

    static void groupInterchangeableValuesByCluster();
};

#endif // _GROUP_H_