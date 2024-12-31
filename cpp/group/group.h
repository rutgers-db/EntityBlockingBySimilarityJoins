/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _GROUP_H_
#define _GROUP_H_

#include "common/io.h"
#include "group/graph.h"
#include <vector>
#include <string>
#include <fstream>
#include <omp.h>


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

private:
    static std::unordered_map<std::string, std::vector<double>> 
    saveNegEmbeddings(const std::vector<std::string> &docs, const DocEmbeddings &vecs);
    static std::unordered_map<std::string, std::vector<std::vector<double>>>
    saveNegWordEmbeddings(const std::vector<std::string> &docs, const WordEmbeddings &vecs);

    static void updateNegEmbeddings(const std::vector<std::string> &docs, const DocEmbeddings &vecs, 
                                    std::unordered_map<std::string, std::vector<double>> &doc2Vec);
    static void updateNegWordEmbeddings(const std::vector<std::string> &docs, const WordEmbeddings &vecs, 
                                        std::unordered_map<std::string, std::vector<std::vector<double>>> &doc2Vec);

    static double calculateCosineSim(const std::vector<double> &lhs, const std::vector<double> &rhs);

    static double calculateCoherentFactor(const std::vector<std::vector<double>> &lhs, 
                                          const std::vector<std::vector<double>> &rhs);

    static Table slimTab(const Table &tab, ui workIdCol, ui queryIdCol, ui workCol, ui queryCol, 
                         const std::unordered_map<std::string, std::vector<double>> &doc2Vec);
    static Table slimTab(const Table &tab, ui workIdCol, ui queryIdCol, ui workCol, ui queryCol, 
                         const std::unordered_map<std::string, std::vector<std::vector<double>>> &doc2Vec);

    static std::unordered_map<int, std::string> getOriginalValue(const std::string &pathTab, const std::string &attr);
    static Table restoreTab(const std::string &pathMatchTab, const std::unordered_map<int, std::string> &id2ValueA, 
                            const std::unordered_map<int, std::string> &id2ValueB, const std::string &attr);

public:
    // interchangeable values directory
    static std::string getICVDir(const std::string &defaultICVDir);
    static std::string getNegMatchDir(const std::string &defaultMatchResDir);
    static std::string getBufferDir(const std::string &defaultBufferDir);

public:
    // io
    // doc embedding: doc2vec
    static void readDocsAndVecs(std::vector<std::string> &docs, DocEmbeddings &vecs, 
                                const std::string &defaultICVDir = "", 
                                const std::string &defaultTabName = "");
    // word embedding: word2vec / fasttext
    static void readWordEmbeddingDocsAndVecs(std::vector<std::string> &docs, WordEmbeddings &vecs, 
                                             const std::string &defaultICVDir = "", 
                                             const std::string &defaultTabName = "");

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

    // reformat match result table
    static void reformatMatchResTableDoc(const std::string &pathMatchTab, const std::string &groupAttribute, const Graph &semanticGraph,
                                         const std::unordered_map<std::string, std::vector<double>> &doc2Vec);

    static void reformatMatchResTableWord(const std::string &pathMatchTab, const std::string &groupAttribute, const Graph &semanticGraph,
                                          const std::unordered_map<std::string, std::vector<std::vector<double>>> &doc2Vec);

    static void reformatTableByInterchangeableValuesByGraph(const std::string &groupAttribute, double groupTau, bool isTransitiveClosure, 
                                                            const std::string &defaultICVDir = "", const std::string &defaultMatchResDir = "");

    static void reformatTableByInterchangeableValuesByWordGraph(const std::string &groupAttribute, double groupTau, bool isTransitiveClosure, 
                                                                const std::string &defaultICVDir = "", const std::string &defaultMatchResDir = "");

    // slim the matching results after the second matching on refactored tables
    static void slimMatchResDoc(const std::string &pathMatchTab, const std::string &groupAttribute, const std::string &defaultICVDir = "", 
                                const std::string &defaultBufferDir = "", int ifNeg = 0);

    static void slimMatchResWord(const std::string &pathMatchTab, const std::string &groupAttribute, const std::string &defaultICVDir = "", 
                                 const std::string &defaultBufferDir = "", int isNeg = 0);
};

#endif // _GROUP_H_