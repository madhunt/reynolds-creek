#!/bin/bash

declare -a day_list=("2023-10-05" "2023-10-06" "2023-10-07" "2023-10-08")
declare -a arr_list=("JDNA" "JDNB" "JDSA" "JDSB")

for day in "${day_list[@]}"
do
    for array in "${arr_list[@]}"
    do
        command="sbatch -J $array$day sbatch_parallel.sh $array $day"
        echo "Running command: $command"
        $command
    done
done
