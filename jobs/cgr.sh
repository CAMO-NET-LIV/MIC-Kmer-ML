#!/bin/bash -l
# Use the current working directory, which is the default setting.
#SBATCH -D ./
# Use the current environment for this job, which is the default setting.
#SBATCH --export=ALL
#SBATCH -p lowpriority,nodes
# Job name:
#SBATCH -J cgr
# Define an output file - will contain error messages too
#SBATCH -o slurm-cgr.out
# Number of nodes
#SBATCH -N 1
# Number of tasks
#SBATCH -n 40

date
echo "This code is running on "
hostname
echo "Starting running on host $HOSTNAME"

conda activate genome

python3 main-cgr.py \
    --label-file "../volatile/cgr_labels/cgr_label.csv" \
    --data-dir "../volatile/cgr/" \
    --antibiotic "mic_AMC" \
    --epochs 2 \
    --workers 38 \

echo "Finished running - goodbye from $HOSTNAME"
