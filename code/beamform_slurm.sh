#!/bin/bash
#SBATCH -J BEAMFORM_RCEW    # job name
#SBATCH -o log_slurm.o%j    # output and error file
#SBATCH --ntasks=1          # num tasks requested
#SBATCH --nodes=1           # num nodes
#SBATCH --cpus-per-task=48  # one node worth of cores
#SBATCH -p bsudfq           # queue/partition
#SBATCH -t 01-00:00:00      # expected run time (DD-HH:MM:SS)

# NOTE you can run this to update the time limit: scontrol update job JOBID timelimit=NEWTIMELIMIT

echo "Date                  = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

module load borah-base openmpi/4.1.3/gcc/12.1.0
mpirun -n 48 ./main.py
