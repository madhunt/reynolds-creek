#!/bin/bash
#SBATCH -J BEAMFORM_RCEW    # job name
#SBATCH -o log_slurm.o%j    # output and error file
#SBATCH -n 1                # num tasks requested
#SBATCH -N 1                # num nodes
#SBATCH -c 48               # one node worth of cores
#SBATCH -p bsudfq           # queue/partition
#SBATCH -t 06:00:00          # expected run time (DD-HH:MM:SS)

# NOTE you can run this to update the time limit: 
    #$ scontrol update job JOBID timelimit=NEWTIMELIMIT

echo "Date                  = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

. ~/.bashrc
mamba activate array
python ./beamform.py -p -b -f 0.5 10 -x TOP07
