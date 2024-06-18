#!/bin/bash
#SBATCH -J 10-06               # job name
#SBATCH -o ./code/log/log_slurm.o%j     # output and error file
#SBATCH -n 1                            # num tasks requested
#SBATCH -N 1                            # num nodes
#SBATCH -c 48                           # one node worth of cores
#SBATCH -p bsudfq                       # queue/partition
#SBATCH -t 10:00:00                     # expected run time (DD-HH:MM:SS)

# NOTE run to update the time limit mid-run:
    #$ scontrol update job JOBID timelimit=NEWTIMELIMIT

echo "Time Submitted        = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

. ~/.bashrc
mamba activate array
python -u code/beamform.py -p -b -f 0.5 10 -x TOP07 -t1 2023-10-06

echo "Time Completed        = $(date)"
