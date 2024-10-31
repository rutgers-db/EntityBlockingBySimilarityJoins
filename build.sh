#! /bin/bash

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build --clean-first -- -j $(nproc) > build/compile.log 2>&1

if [ $? -eq 0 ]; then
    echo "done compile."
else
    echo "fail to compile."
fi