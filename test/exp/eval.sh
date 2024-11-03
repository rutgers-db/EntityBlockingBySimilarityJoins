#! /bin/bash

script="test/exp/${1}.py"
output="output/block_${1}_${2}.txt"

if [[ "$1" == "secret" ]]
then
nohup python3 ${script} --turn $3 --number $2 >> "${output}">&1 &
else
nohup python3 ${script} --turn $3 --dtype $2 >> "${output}">&1 &
fi

echo $!
wait $!
echo $?