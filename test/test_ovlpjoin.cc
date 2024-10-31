/*
 * Test unit on overlap join
 * RS serial & parallel join
 */
#include "common/dataframe.h"
#include "common/io.h"
#include "common/tokenizer.h"
#include "common/ovlpjoin.h"
#include "common/ovlpjoin_parallel.h"
#include "common/simfunc.h"
#include <vector>

#define MAX_SCHEMA_SIZE 10
#define VERIFY_RS_JOIN
#define VERIFY_SELF_JOIN


int verify(const std::vector<std::vector<ui>> &recordsA, const std::vector<std::vector<ui>> &recordsB, int tau)
{
    int hit = 0;

    ui sizeA = recordsA.size();
    ui sizeB = recordsB.size();
    for(ui i = 0; i < sizeA; i++)
        for(ui j = 0; j < sizeB; j++)
            if(SimFuncs::overlap(recordsA[i], recordsB[j]) >= tau && !recordsA[i].empty() && !recordsB[j].empty())
                ++ hit;

    return hit;
}

int verify(const std::vector<std::vector<ui>> &records, int tau)
{
    int hit = 0;

    ui size = records.size();
    for(ui i = 0; i < size; i++) 
        for(ui j = i + 1; j < size; j++) 
            if(SimFuncs::overlap(records[i], records[j]) >= tau && !records[i].empty() && !records[j].empty())
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

    std::vector<std::vector<ui>> recordsA[MAX_SCHEMA_SIZE], recordsB[MAX_SCHEMA_SIZE];
    std::vector<ui> idMapA[MAX_SCHEMA_SIZE], idMapB[MAX_SCHEMA_SIZE];
    std::vector<double> weightsA[MAX_SCHEMA_SIZE], weightsB[MAX_SCHEMA_SIZE], wordwt[MAX_SCHEMA_SIZE];

    std::vector<std::pair<int, int>> serialPairs, parallelPairs;

    OvlpRSJoin *RSjoin = nullptr;
    OvlpRSJoinParallel *RSjoinParallel = nullptr;
    OvlpSelfJoin *selfJoin = nullptr;
    OvlpSelfJoinParallel *selfJoinParallel = nullptr;

    int detRSJoin = 1;
    int detSelfJoin = 1;

    ui schemaSize = tableA.schema.size();
    for(ui i = 0; i < schemaSize; i++) {
        int hit = 0;
        std::string curAttr = tableA.schema[i];
        std::cout << "--- working on: " << curAttr << " ---" << std::endl;

        // tokenize
        ui numWord = 0;
        TokenizerType tokType = TokenizerType::QGram;
        if(curAttr == "name" || curAttr == "title" || curAttr == "authors") {
            detRSJoin = 3;
            detSelfJoin = 3;
            tokType = TokenizerType::Dlm;
        }
        Tokenizer::RStableAttr2IntVector(tableA, tableB, recordsA[i], recordsB[i], 
                                         weightsA[i], weightsB[i], wordwt[i], 
                                         idMapA[i], idMapB[i], i, i, 
                                         tokType, numWord, 3);
        // RS join
        RSjoin = new OvlpRSJoin(recordsA[i], recordsB[i], weightsA[i], weightsB[i], wordwt[i]);
        RSjoinParallel = new OvlpRSJoinParallel(recordsA[i], recordsB[i], weightsA[i], weightsB[i], wordwt[i]);

        RSjoin->overlapjoin(detRSJoin, serialPairs);
        RSjoinParallel->overlapjoin(detRSJoin, parallelPairs);

        hit = 0;

        std::cout << "###                       --- test report on RS join ---" << std::endl;
        hit = verify(recordsA[i], recordsB[i], detRSJoin);
        std::cout << "###                       --- verify result overlap: " << hit << " serial: " << serialPairs.size() 
                  << " parallel: " << parallelPairs.size() << " ---" << std::endl;

        // release
        delete RSjoin;
        delete RSjoinParallel;
        serialPairs.clear();
        parallelPairs.clear();

        // self join
        selfJoin = new OvlpSelfJoin(recordsB[i], weightsB[i], wordwt[i]);
        selfJoinParallel = new OvlpSelfJoinParallel(recordsB[i], weightsB[i], wordwt[i]);

        selfJoin->overlapjoin(detSelfJoin, serialPairs);
        selfJoinParallel->overlapjoin(detSelfJoin, parallelPairs);

        hit = 0;

        std::cout << "###                       --- test report on self join ---" << std::endl;
        hit = verify(recordsB[i], detSelfJoin);
        std::cout << "###                       --- verify result overlap: " << hit << " serial: " << serialPairs.size() 
                  << " parallel: " << parallelPairs.size() << " ---" << std::endl;

        // release
        delete selfJoin;
        delete selfJoinParallel;
        serialPairs.clear();
        parallelPairs.clear();
    }
}