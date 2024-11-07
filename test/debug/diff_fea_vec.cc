#include "common/io.h"
#include <string>


int main()
{
    std::string pathStatFile = "output/blk_res/stat.txt";
    FILE *fp = fopen(pathStatFile.c_str(), "r");

    int totalTable, totalEntity;
    fscanf(fp, "%d %d\n", &totalTable, &totalEntity);
    printf("%d\n", totalTable);

    fclose(fp);

    CSVReader reader;
    auto diffOriFeaVec = Table();
    auto diffProFeaVec = Table();

    for(int i = 0; i < totalTable; i++) {
        std::string pathOriFeaVec = "output/blk_res/feature_vec" + std::to_string(i) + "_py.csv";
        std::string pathProFeaVec = "output/blk_res/feature_vec" + std::to_string(i) + ".csv";

        reader.reading_one_table(pathOriFeaVec, false);
        reader.reading_one_table(pathProFeaVec, false);

        Table oriFeaVec = reader.tables[i * 2];
        Table proFeaVec = reader.tables[i * 2 + 1];

        if(i == 0) {
            diffOriFeaVec.schema = oriFeaVec.schema;
            diffProFeaVec.schema = proFeaVec.schema;
            diffOriFeaVec.inverted_schema = oriFeaVec.inverted_schema;
            diffProFeaVec.inverted_schema = proFeaVec.inverted_schema;
            diffOriFeaVec.cols.resize(oriFeaVec.col_no);
            diffProFeaVec.cols.resize(proFeaVec.col_no);
        }

        oriFeaVec.Profile();
        proFeaVec.Profile();
        int numRow = std::min(oriFeaVec.row_no, proFeaVec.row_no);
        int numColumn = oriFeaVec.col_no;

        for(int row = 0; row < numRow; row++) {
            for(int col = 0; col < numColumn; col++) {
                if(oriFeaVec.rows[row][col].empty() ^ proFeaVec.rows[row][col].empty() || 
                   oriFeaVec.rows[row][col] != proFeaVec.rows[row][col]) {
                    diffOriFeaVec.insertOneRow(oriFeaVec.rows[row]);
                    diffProFeaVec.insertOneRow(proFeaVec.rows[row]);
                    break;
                }
            }
        }
    }

    diffOriFeaVec.Profile();
    diffProFeaVec.Profile();

    MultiWriter::writeOneTable(diffOriFeaVec, "test/debug/ori_fea_vec.csv");
    MultiWriter::writeOneTable(diffProFeaVec, "test/debug/pro_fea_vec.csv");

    return 0;
}