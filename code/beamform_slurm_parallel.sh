#!/bin/bash
#SBATCH -J PARALLEL_BEAM    # job name
#SBATCH -o log_slurm.o%j    # output and error file
#SBATCH -n 1                # num tasks requested
#SBATCH -N 1                # num nodes
#SBATCH -c 48               # one node worth of cores
#SBATCH -p bsudfq           # queue/partition
#SBATCH -t 48:00:00         # expected run time (DD-HH:MM:SS)

# NOTE you can run this to update the time limit: 
    #$ scontrol update job JOBID timelimit=NEWTIMELIMIT

echo "Date                  = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

. ~/.bashrc
mamba activate array
python ./beamform.py -p -b -F 0.5 40 -x TOP07

echo "Date 2                  = $(date)"

