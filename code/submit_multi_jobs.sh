#!/bin/bash

declare -a arr=("2023-10-05" "2023-10-06" "2023-10-07" "2023-10-08")

for day in "${arr[@]}"
do
    command="sbatch -J TOP_$day sbatch_top_parallel.sh $day"
    echo "Running command: $command"
    $command
done