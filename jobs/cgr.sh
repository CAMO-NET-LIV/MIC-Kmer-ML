#!/bin/bash -l
# Use the current working directory, which is the default setting.
#SBATCH -D ./
# Use the current environment for this job, which is the default setting.
#SBATCH --export=ALL
#SBATCH -p lowpriority,nodes
# Job name:
#SBATCH -J cgr
# Define an output file - will contain error messages too
#SBATCH -o slurm-cgr_%a.out
# Number of nodes
#SBATCH -N 1
# Number of tasks
#SBATCH -n 40
# Array job
#SBATCH --array=0-9

# set evirontment variables
export RAY_ADDRESS=""

# Array of antibiotics
antibiotics=("mic_AMC" "mic_AMK" "mic_AMX" "mic_CAZ" "mic_CHL" "mic_CIP" "mic_FEP" "mic_GEN" "mic_MEM" "mic_TGC")

# Get the antibiotic for this job
antibiotic=${antibiotics[$SLURM_ARRAY_TASK_ID]}

date
echo "This code is running on "
hostname
echo "Starting running on host $HOSTNAME with antibiotic $antibiotic"

conda activate genome

ray start --head --num-cpus 38 --block &

python3 main-cgr.py \
    --label-file "../volatile/cgr_labels/cgr_label.csv" \
    --data-dir "../volatile/cgr/" \
    --antibiotic "$antibiotic" \
    --epochs 100 \
    --workers 38 \

echo "Finished running - goodbye from $HOSTNAME with antibiotic $antibiotic"
