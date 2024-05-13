#!/bin/bash
#SBATCH -J TEST             # job name
#SBATCH -o log_slurm.o%j    # output and error file
#SBATCH --ntasks=1          # num tasks requested
#SBATCH --nodes=1           # num nodes
#SBATCH --cpus-per-task=48  # one node worth of cores
#SBATCH -p bsudfq           # queue/partition
#SBATCH -t 00-00:05:00      # expected runtime (DD-HH:MM:SS)


echo "Date                  = $(date)"
echo "Hostname              = $(hostname -s)"
echo "PWD                   = $(pwd)"
echo "Num Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated   = $SLURM_NTASKS"

module purge
#module load borah-base python/3.9.7 #openmpi/4.1.3/gcc/12.1.0

#mpirun -np 48 python ./test.py
#mpirun -n 48 ./test.py

. ~/.bashrc
conda activate #ENVIRONEMTN NAME HERE
python ./test.py
