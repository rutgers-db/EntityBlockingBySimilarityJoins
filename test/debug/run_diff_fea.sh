#! /bin/bash

g++ -std=c++11 -O3 -g test/debug/diff_fea_vec.cc cpp/common/dataframe.cc cpp/common/io.cc -Icpp -o test/debug/diff_fea_vec -fopenmp

./test/debug/diff_fea_vec