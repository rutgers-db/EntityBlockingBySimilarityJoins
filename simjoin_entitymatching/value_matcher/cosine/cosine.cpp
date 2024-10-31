#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <assert.h>

/*
 * calculate cosine similarity based on word embedding vectors
 * since the implementation in numpy is quite inefficient
 */
int main(int argc, char *argv[])
{
    std::string filenameInput = argv[1];
    std::string filenameOutput = argv[2];
    double tau = std::stod(argv[3]);
    std::cerr << tau << std::endl;
    
    std::ifstream vecfile(filenameInput.c_str(), std::ios::in);
    std::vector<std::vector<double>> lvecs, rvecs;

    int tot = 0;
    vecfile >> tot;
    lvecs.reserve(tot);
    rvecs.reserve(tot);
    for(int i = 0; i < tot; i++) {
        std::vector<double> lvec, rvec;
        int lsize = 0;
        vecfile >> lsize;
        lvec.reserve(lsize);
        for(int j = 0; j < lsize; j++) {
            double val = 0.0;
            vecfile >> val;
            lvec.emplace_back(val);
        }
        int rsize = 0;
        vecfile >> rsize;
        rvec.reserve(rsize);
        for(int j = 0; j < rsize; j++) {
            double val = 0.0;
            vecfile >> val;
            rvec.emplace_back(val);
        }
        assert(lsize == rsize);
        if(lsize != rsize) {
            std::cerr << tot << " " << i << std::endl;
            std::cerr << lvec.front() << std::endl;
            std::cerr << lsize << " " << rsize << std::endl;
            exit(1);
        }
        lvecs.emplace_back(std::move(lvec));
        rvecs.emplace_back(std::move(rvec));
    }

    FILE *labelfile = fopen(filenameOutput.c_str(), "w");
    if(labelfile == nullptr) {
        fprintf(stderr, "Can not open: %s\n", filenameOutput.c_str());
        exit(1);
    }
    int poscnt = 0;
    for(int i = 0; i < tot; i++) {
        int size = (int)lvecs[i].size();
        double dotp = 0.0, norml = 0.0, normr = 0.0;
        for(int j = 0; j < size; j++) {
            double lval = lvecs[i][j];
            double rval = rvecs[i][j];
            dotp += (lval * rval);
            norml += (lval * lval);
            normr += (rval * rval);
        }
        norml = sqrt(norml);
        normr = sqrt(normr);
        double sim = dotp / (norml * normr);
        if(sim >= tau) {
            ++ poscnt;
            fprintf(labelfile, "1\n");
        }
        else
            fprintf(labelfile, "0\n");
    }

    printf("calculation done: %d\n", poscnt);
    fclose(labelfile);
    return 0;
}