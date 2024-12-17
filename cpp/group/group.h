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
    using DocEmbeddings = std::vector<std::vector<double>>;
    using WordEmbeddings = std::vector<std::vector<std::vector<double>>>;

public:
    Group() = default;
    ~Group() = default;
    Group(const Group &other) = delete;
    Group(Group &&other) = delete;

public:
    // interchangeable values directory
    static std::string getICVDir(const std::string &defaultICVDir);

public:
    // io
    // doc embedding: doc2vec
    static void readDocsAndVecs(std::vector<std::string> &docs, DocEmbeddings &vecs, 
                                const std::string &defaultICVDir = "");
    // word embedding: word2vec / fasttext
    static void readWordEmbeddingDocsAndVecs(std::vector<std::string> &docs, WordEmbeddings &vecs, 
                                             const std::string &defaultICVDir = "");

    static void readDocCandidatePairs(std::vector<std::pair<std::string, std::string>> &candidates, 
                                      const std::string &defaultICVDir = "");

    // apis
    // default: doc embedding graph
    static void groupInterchangeableValuesByGraph(const std::string &groupAttribute, const std::string &groupStrategy, 
                                                  double groupTau, bool isTransitiveClosure, 
                                                  const std::string &defaultICVDir = "");

    // word embedding graph
    static void groupInterchangeableValuesByWordGraph(const std::string &groupAttribute, const std::string &groupStrategy, 
                                                      double groupTau, bool isTransitiveClosure, 
                                                      const std::string &defaultICVDir = "");

    static void groupInterchangeableValuesByCluster();
};

#endif // _GROUP_H_