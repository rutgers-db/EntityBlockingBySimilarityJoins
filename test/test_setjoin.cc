/*
 * Test unit on set join
 * RS serial & parallel join
 */
#include "common/dataframe.h"
#include "common/io.h"
#include "common/tokenizer.h"
#include "common/setjoin.h"
#include "common/setjoin_parallel.h"
#include "common/simfunc.h"
#include <vector>

#define MAX_SCHEMA_SIZE 10
// #define VERIFY_RS_JOIN
// #define VERIFY_SELF_JOIN
#define VERIFY_PARAMETER

typedef double (*SetJoinFunc)(const std::vector<ui> &, const std::vector<ui> &);

int verify(SimFuncType sft, const std::vector<std::vector<ui>> &recordsA, 
            const std::vector<std::vector<ui>> &recordsB, double tau)
{
    int hit = 0;

    SetJoinFunc sjfp = nullptr;
    switch(sft) {
        case SimFuncType::JACCARD : sjfp = &SimFuncs::jaccard; break;
        case SimFuncType::COSINE : sjfp = &SimFuncs::cosine; break;
        case SimFuncType::DICE : sjfp = &SimFuncs::dice; break;
    }

    ui sizeA = recordsA.size();
    ui sizeB = recordsB.size();
    for(ui i = 0; i < sizeA; i++)
        for(ui j = 0; j < sizeB; j++)
            if(sjfp(recordsA[i], recordsB[j]) >= tau && !recordsA[i].empty() && !recordsB[j].empty())
                ++ hit;

    return hit;
}

int verify(SimFuncType sft, const std::vector<std::vector<ui>> &records, double tau)
{
    int hit = 0;

    SetJoinFunc sjfp = nullptr;
    switch(sft) {
        case SimFuncType::JACCARD : sjfp = &SimFuncs::jaccard; break;
        case SimFuncType::COSINE : sjfp = &SimFuncs::cosine; break;
        case SimFuncType::DICE : sjfp = &SimFuncs::dice; break;
    }

    ui size = records.size();
    for(ui i = 0; i < size; i++) 
        for(ui j = i + 1; j < size; j++) 
            if(sjfp(records[i], records[j]) >= tau && !records[i].empty() && !records[j].empty())
                ++ hit;

    return hit;
}

int main(int argc, char *argv[])
{
    std::string dirName = argv[1];
    std::string pathDir = "/home/ylilo/projects/datasets/tables/secret/" + dirName + "/";
    std::string pathTableA = pathDir + "table_a.csv";
    std::string pathTableB = pathDir + "table_b.csv";
    std::string pathSimPairs = "dummy";
    
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

    std::vector<std::pair<int, int>> jaccardPairs, jaccardParallelPairs;
    std::vector<std::pair<int, int>> cosinePairs, cosineParallelPairs;
    std::vector<std::pair<int, int>> dicePairs, diceParallelPairs;

    SetJoin *jaccardJoin = nullptr; SetJoinParallel *jaccardParallelJoin = nullptr;
    SetJoin *cosineJoin = nullptr; SetJoinParallel *cosineParallelJoin = nullptr;
    SetJoin *diceJoin = nullptr; SetJoinParallel *diceParallelJoin = nullptr;

    double detRSJoin = 0.41;
    double detSelfJoin = 0.41;

    ui schemaSize = tableA.schema.size();
    for(ui i = 1; i < schemaSize; i++) {
        int hit = 0;
        std::string curAttr = tableA.schema[i];
        std::cout << "--- working on: " << curAttr << " ---" << std::endl;

        // tokenize
        ui numWord = 0;
        TokenizerType tokType = TokenizerType::QGram;
        if(curAttr == "name" || curAttr == "title")
            tokType = TokenizerType::Dlm;
#ifndef VERIFY_PARAMETER
        Tokenizer::RStableAttr2IntVector(tableA, tableB, recordsA[i], recordsB[i], 
                                         weightsA[i], weightsB[i], wordwt[i], 
                                         idMapA[i], idMapB[i], i, i, 
                                         tokType, numWord, 3);
#endif
#ifdef VERIFY_PARAMETER
        Tokenizer::SelftableAttr2IntVector(tableA, recordsA[i], weightsA[i], 
                                           wordwt[i], idMapA[i], i, tokType, 
                                           numWord, 3);
#endif
#ifdef VERIFY_RS_JOIN
        // RS join
        jaccardJoin = new SetJoin(recordsB[i], recordsA[i], weightsB[i], weightsA[i], wordwt[i], pathSimPairs, detRSJoin);
        cosineJoin = new SetJoin(recordsB[i], recordsA[i], weightsB[i], weightsA[i], wordwt[i], pathSimPairs, detRSJoin);
        diceJoin = new SetJoin(recordsB[i], recordsA[i], weightsB[i], weightsA[i], wordwt[i], pathSimPairs, detRSJoin);
        jaccardJoin->simFType = SimFuncType::JACCARD;
        cosineJoin->simFType = SimFuncType::COSINE;
        diceJoin->simFType = SimFuncType::DICE;

        jaccardParallelJoin = new SetJoinParallel(recordsB[i], recordsA[i], weightsB[i], weightsA[i], wordwt[i], detRSJoin);
        cosineParallelJoin = new SetJoinParallel(recordsB[i], recordsA[i], weightsB[i], weightsA[i], wordwt[i], detRSJoin);
        diceParallelJoin = new SetJoinParallel(recordsB[i], recordsA[i], weightsB[i], weightsA[i], wordwt[i], detRSJoin);
        jaccardParallelJoin->simFType = SimFuncType::JACCARD;
        cosineParallelJoin->simFType = SimFuncType::COSINE;
        diceParallelJoin->simFType = SimFuncType::DICE;

        jaccardJoin->setRSJoin(detRSJoin, jaccardPairs);
        cosineJoin->setRSJoin(detRSJoin, cosinePairs);
        diceJoin->setRSJoin(detRSJoin, dicePairs);

        jaccardParallelJoin->index(detRSJoin);
        jaccardParallelJoin->findSimPairsRS();
        jaccardParallelJoin->mergeResults(jaccardParallelPairs);

        cosineParallelJoin->index(detRSJoin);
        cosineParallelJoin->findSimPairsRS();
        cosineParallelJoin->mergeResults(cosineParallelPairs);

        diceParallelJoin->index(detRSJoin);
        diceParallelJoin->findSimPairsRS();
        diceParallelJoin->mergeResults(diceParallelPairs);

        std::cout << "--- test report on RS join ---" << std::endl;
        hit = verify(SimFuncType::JACCARD, recordsA[i], recordsB[i], detRSJoin);
        std::cout << "--- verify result jaccard: " << hit << " serial: " << jaccardPairs.size() 
                  << " parallel: " << jaccardParallelPairs.size() << " ---" << std::endl;
        hit = verify(SimFuncType::COSINE, recordsA[i], recordsB[i], detRSJoin);
        std::cout << "--- verify result cosine: " << hit << " serial: " << cosinePairs.size() 
                  << " parallel: " << cosineParallelPairs.size() << " ---" << std::endl;
        hit = verify(SimFuncType::DICE, recordsA[i], recordsB[i], detRSJoin);
        std::cout << "--- verify result dice: " << hit << " serial: " << dicePairs.size() 
                  << " parallel: " << diceParallelPairs.size() << " ---" << std::endl;

        // release
        delete jaccardJoin;
        delete jaccardParallelJoin;
        delete cosineJoin;
        delete cosineParallelJoin;
        delete diceJoin;
        delete diceParallelJoin;
        jaccardPairs.clear();
        jaccardParallelPairs.clear();
        cosinePairs.clear();
        cosineParallelPairs.clear();
        dicePairs.clear();
        diceParallelPairs.clear();
#endif
#ifdef VERIFY_SELF_JOIN
        // self join
        jaccardJoin = new SetJoin(recordsB[i], weightsB[i], wordwt[i], pathSimPairs, detSelfJoin);
        cosineJoin = new SetJoin(recordsB[i], weightsB[i], wordwt[i], pathSimPairs, detSelfJoin);
        diceJoin = new SetJoin(recordsB[i], weightsB[i], wordwt[i], pathSimPairs, detSelfJoin);
        jaccardJoin->simFType = SimFuncType::JACCARD;
        cosineJoin->simFType = SimFuncType::COSINE;
        diceJoin->simFType = SimFuncType::DICE;

        jaccardParallelJoin = new SetJoinParallel(recordsB[i], weightsB[i], wordwt[i], detSelfJoin);
        cosineParallelJoin = new SetJoinParallel(recordsB[i], weightsB[i], wordwt[i], detSelfJoin);
        diceParallelJoin = new SetJoinParallel(recordsB[i], weightsB[i], wordwt[i], detSelfJoin);
        jaccardParallelJoin->simFType = SimFuncType::JACCARD;
        cosineParallelJoin->simFType = SimFuncType::COSINE;
        diceParallelJoin->simFType = SimFuncType::DICE;

        jaccardJoin->setSelfJoin(detSelfJoin, jaccardPairs);
        cosineJoin->setSelfJoin(detSelfJoin, cosinePairs);
        diceJoin->setSelfJoin(detSelfJoin, dicePairs);

        jaccardParallelJoin->index(detSelfJoin);
        jaccardParallelJoin->findSimPairsSelf();
        jaccardParallelJoin->mergeResults(jaccardParallelPairs);

        cosineParallelJoin->index(detSelfJoin);
        cosineParallelJoin->findSimPairsSelf();
        cosineParallelJoin->mergeResults(cosineParallelPairs);

        diceParallelJoin->index(detSelfJoin);
        diceParallelJoin->findSimPairsSelf();
        diceParallelJoin->mergeResults(diceParallelPairs);

        std::cout << "--- test report on self join ---" << std::endl;
        hit = verify(SimFuncType::JACCARD, recordsB[i], detSelfJoin);
        std::cout << "--- verify result jaccard: " << hit << " serial: " << jaccardPairs.size() 
                  << " parallel: " << jaccardParallelPairs.size() << " ---" << std::endl;
        hit = verify(SimFuncType::COSINE, recordsB[i], detSelfJoin);
        std::cout << "--- verify result cosine: " << hit << " serial: " << cosinePairs.size() 
                  << " parallel: " << cosineParallelPairs.size() << " ---" << std::endl;
        hit = verify(SimFuncType::DICE, recordsB[i], detSelfJoin);
        std::cout << "--- verify result dice: " << hit << " serial: " << dicePairs.size() 
                  << " parallel: " << diceParallelPairs.size() << " ---" << std::endl;

        // release
        delete jaccardJoin;
        delete jaccardParallelJoin;
        delete cosineJoin;
        delete cosineParallelJoin;
        delete diceJoin;
        delete diceParallelJoin;
        jaccardPairs.clear();
        jaccardParallelPairs.clear();
        cosinePairs.clear();
        cosineParallelPairs.clear();
        dicePairs.clear();
        diceParallelPairs.clear();
#endif
#ifdef VERIFY_PARAMETER
        jaccardParallelJoin = new SetJoinParallel(recordsA[i], weightsA[i], wordwt[i], 0.97);
        cosineParallelJoin = new SetJoinParallel(recordsA[i], weightsA[i], wordwt[i], 0.97);
        diceParallelJoin = new SetJoinParallel(recordsA[i], weightsA[i], wordwt[i], 0.97);
        jaccardParallelJoin->simFType = SimFuncType::JACCARD;
        cosineParallelJoin->simFType = SimFuncType::COSINE;
        diceParallelJoin->simFType = SimFuncType::DICE;

        jaccardParallelJoin->index(0.97);
        jaccardParallelJoin->findSimPairsSelf();
        jaccardParallelJoin->mergeResults(jaccardParallelPairs);

        cosineParallelJoin->index(0.97);
        cosineParallelJoin->findSimPairsSelf();
        cosineParallelJoin->mergeResults(cosineParallelPairs);

        diceParallelJoin->index(0.97);
        diceParallelJoin->findSimPairsSelf();
        diceParallelJoin->mergeResults(diceParallelPairs);

        break;
#endif
    }
}