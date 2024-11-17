#! /bin/bash

cmake -S . -B build/test/debug -DCMAKE_BUILD_TYPE=Debug

cmake --build build/test/debug --clean-first -- -j $(nproc) > build/debug_compile.log 2>&1

if [ $? -eq 0 ]; then
    echo "compile done."
else
    echo "fail to compile."
fi