#!/bin/bash
#SBATCH -J python           # job name
#SBATCH -o log_slurm.o%j    # output and error file
#SBATCH -n 1                # num tasks requested
#SBATCH -N 1                # num nodes
#SBATCH -c 48               # one node worth of cores
#SBATCH -p bsudfq           # queue/partition
#SBATCH -t 00-00:05:00      # expected runtime (DD-HH:MM:SS)


echo "Date                  = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

. ~/.bashrc
conda activate array
python ./test.py
