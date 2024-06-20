#!/bin/bash

#SBATCH -o ./code/log/log_slurm.o%j     # output and error file
#SBATCH -n 1                            # num tasks requested
#SBATCH -N 1                            # num nodes
#SBATCH -c 48                           # one node worth of cores
#SBATCH -p bsudfq                       # queue/partition
#SBATCH -t 15:00:00                     # expected run time (DD-HH:MM:SS)

# NOTE run to update the time limit mid-run:
    #$ scontrol update job JOBID timelimit=NEWTIMELIMIT

echo "Job Name              = $SBATCH_JOB_NAME"
echo "Time Submitted        = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

. ~/.bashrc
mamba activate array
python -u code/beamform.py -p -a TOP -F 0.5 32 -x TOP07 -t1 $1

echo "Time Completed        = $(date)"
