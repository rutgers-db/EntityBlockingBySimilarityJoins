/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#include "group/group.h"
#include "group/graph.h"


struct GroupArgument
{
    std::string groupAttribute;
    std::string groupStrategy;
    std::string defaultICVDir;
    double groupTau;
    bool isTransitiveClosure;

    void loadArguments(char *argv[]) {
        groupAttribute = argv[1];
        groupStrategy = argv[2];
        defaultICVDir = argv[3];
        groupTau = std::stod(argv[4]);
        isTransitiveClosure = atoi(argv[5]) == 0 ? false : true;
    }

    void checkArguments() const {
        if(groupStrategy != "graph" && groupStrategy != "cluster") {
            std::cerr << "error in groupStrategy : " << groupStrategy << std::endl;
            exit(1);
        }
        else if(groupTau > 1.0 || groupTau < 0.0) {
            std::cerr << "error in groupTau : " << groupTau << std::endl;
            exit(1);
        }
    }

    void printArguments() const {
        std::cerr << "group interchangeable values on attribute : " << groupAttribute 
        << "\tstrategy : " << groupStrategy 
        << "\tdefault directory for output : " << defaultICVDir
        << "\tthreshold : " << groupTau
        << "\tis transitive closure enabled : " << isTransitiveClosure << std::endl;
    }
};


int main(int argc, char *argv[])
{
    GroupArgument arguments;
    arguments.loadArguments(argv);
    arguments.checkArguments();
    arguments.printArguments();

    // io
    std::vector<std::string> docs;
    std::vector<std::vector<double>> vecs;
    std::vector<std::pair<std::string, std::string>> candidates;

    Group::readDocsAndVecs(docs, vecs, arguments.defaultICVDir);
    Group::readDocCandidatePairs(candidates, arguments.defaultICVDir);

    if(arguments.groupStrategy == "graph") {
        // build
        Graph senmaticGraph(arguments.isTransitiveClosure, arguments.groupTau);
        senmaticGraph.buildSemanticGraph(docs, vecs, candidates);

        // write
        std::string pathGraph = arguments.defaultICVDir + "interchangeable_graph_" 
                                + arguments.groupAttribute + ".txt";
        senmaticGraph.writeSemanticGraph(pathGraph);
    }
    else if(arguments.groupStrategy == "cluster") {

    }
    else {
        std::cout << "no such strategy : " << arguments.groupStrategy << std::endl;
        exit(1);
    }

    return 0;
}