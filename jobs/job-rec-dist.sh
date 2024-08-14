#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J main-rec
#SBATCH -p nodes           # Ensure this is the correct partition
#SBATCH -o slurm-rec-dist.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00

# Optionally set OMP_NUM_THREADS if using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

date
echo "This code is running on "
hostname
echo "Starting running on host $HOSTNAME"

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"

conda activate genome
python3 main-recursive.py --worker 38 --save-file "rec-38-1.txt"
echo "Finished running - goodbye from $HOSTNAME"
