#! /bin/bash

RED='\033[0;31m'

# compile_file="build/Release/${1}_compile.log"
# output_file="build/Release/${1}_output.log"
compile_file="test/build/Debug/${1}_compile.log"
output_file="test/build/Debug/${1}_output.log"

# rm -rf build
my_strings=("$@")

# cmake -B test/build/Release -DEXECUTABLE_NAME=${1} -DCMAKE_BUILD_TYPE=Release
cmake -B test/build/Debug -DEXECUTABLE_NAME=${1} -DCMAKE_BUILD_TYPE=Debug

# nohup cmake --build test/build/Release --clean-first -- -j $(nproc) > "${compile_file}">&1 &
nohup cmake --build test/build/Debug --clean-first -- -j $(nproc) > "${compile_file}">&1 &

oldpid=$!
wait $oldpid
oldpid=$?
echo "cmake exit with $oldpid"

if [ $oldpid -eq 0 ]
then
    # nohup ./test/build/Release/${1} $2 > "${output_file}" 2>&1 &
    gdb --silent ./test/build/Debug/${1}

    echo $!
    wait $!
    status=$?
    echo ${status} >> ${output_file}
    if [ $status -eq 0 ]
    then
        echo "benchmark ${1} done"
    else
        exit 1
    fi
else
    echo -e "${RED}Fail to compile, check ${compile_file}"
    exit 1
fi