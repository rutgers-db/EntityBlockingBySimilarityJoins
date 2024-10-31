/*
 * Test unit on string join
 * RS serial & parallel
 */
#include "common/dataframe.h"
#include "common/io.h"
#include "common/stringjoin.h"
#include "common/stringjoin_parallel.h"
#include "common/simfunc.h"
#include <vector>

#define MAX_SCHEMA_SIZE 10

int verify(const std::vector<std::string> &recordsA, const std::vector<std::string> &recordsB, 
           int tau)
{
    int hit = 0;

    ui sizeA = recordsA.size();
    ui sizeB = recordsB.size();
    for(ui i = 0; i < sizeA; i++)
        for(ui j = 0; j < sizeB; j++)
            if(SimFuncs::levDist(recordsA[i], recordsB[j]) <= tau && !recordsA[i].empty() && !recordsB[j].empty())
                ++ hit;

    return hit;
}

int verify(const std::vector<std::string> &records, int tau)
{
    int hit = 0;

    ui size = records.size();
    for(ui i = 0; i < size; i++)
        for(ui j = i + 1; j < size; j++)
            if(SimFuncs::levDist(records[i], records[j]) <= tau && !records[i].empty() && !records[j].empty())
                ++ hit;

    return hit;
}

int main(int argc, char *argv[])
{
    std::string dirName = argv[1];
    std::string pathDir = "/home/ylilo/projects/datasets/tables/megallen/" + dirName + "/";
    std::string pathTableA = pathDir + "table_a.csv";
    std::string pathTableB = pathDir + "table_b.csv";
    
    CSVReader reader;
    bool normalize = false;

    reader.reading_one_table(pathTableA, normalize);
    reader.reading_one_table(pathTableB, normalize);

    Table tableA = reader.tables[0];
    Table tableB = reader.tables[1];
    tableA.Profile();
    tableB.Profile();

    std::vector<std::pair<int, int>> serialPairs, parallelPairs;

    StringJoin *RSjoin = nullptr, *selfJoin = nullptr;
    StringJoinParallel *RSjoinParallel = nullptr, *selfJoinParallel = nullptr;

    int selfDet = 3;
    int RSDet = 3;
    int hit = 0;

    ui schemaSize = tableA.schema.size();
    for(ui i = 0; i < schemaSize; i++) {
        std::string curAttr = tableA.schema[i];
        std::cout << "--- working on: " << curAttr << " ---" << std::endl;

        if(tableA.schema[i] == "name" || tableA.schema[i] == "title") {
            selfDet = 11;
            RSDet = 11;
        }

        const auto &colA = tableA.cols[i];
        const auto &colB = tableB.cols[i];

        // RS join
        RSjoin = new StringJoin(colB, colA, RSDet);
        RSjoinParallel = new StringJoinParallel(colB, colA, RSDet);

        RSjoin->RSJoin(serialPairs);
        RSjoinParallel->RSJoin(parallelPairs);

        hit = 0;

        std::cout << "###                       --- test report on RS join ---" << std::endl;
        hit = verify(colB, colA, RSDet);
        std::cout << "###                       --- verify result lev: " << hit << " serial: " << serialPairs.size() 
                  << " parallel: " << parallelPairs.size() << " ---" << std::endl;

        // release
        delete RSjoin;
        delete RSjoinParallel;
        serialPairs.clear();
        parallelPairs.clear();

        // self join
        selfJoin = new StringJoin(colB, selfDet);
        selfJoinParallel = new StringJoinParallel(colB, selfDet);

        selfJoin->selfJoin(serialPairs);
        selfJoinParallel->selfJoin(parallelPairs);

        hit = 0;

        std::cout << "###                       --- test report on self join ---" << std::endl;
        hit = verify(colB, selfDet);
        std::cout << "###                       --- verify result lev: " << hit << " serial: " << serialPairs.size() 
                  << " parallel: " << parallelPairs.size() << " ---" << std::endl;

        // release
        delete selfJoin;
        delete selfJoinParallel;
        serialPairs.clear();
        parallelPairs.clear();
    }
}