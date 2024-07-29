#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J main-nn
#SBATCH -o slurm-12mer-dist-cpu.out
#SBATCH -p lowpriority,nodes            # Use the CPU partition
#SBATCH -N 4                  # Number of nodes
#SBATCH --ntasks-per-node=1   # Number of tasks per node
#SBATCH --cpus-per-task=40    # Number of CPU cores per task (adjust based on your requirements)
#SBATCH --time=1-00:00:00       # Increase time limit

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

# Load required modules and activate virtual environment
module load apps/python3/3.8.5/gcc-5.5.0
source .venv/bin/activate

# Get the node list
NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "NODELIST: $NODELIST"
MASTER_NODE=$(echo "$NODELIST" | head -n 1)
echo "MASTER_NODE: $MASTER_NODE"
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
MASTER_PORT=15625  # Choose a port number that's not in use

# Debugging: Print master address and port
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Get the number of nodes
NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD_SIZE="$NUM_NODES"

echo "NUM_NODES: $NUM_NODES"
echo "WORLD_SIZE: $WORLD_SIZE"

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE

# Run a simple srun command to verify it's working
srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" echo "srun test on MASTER_NODE: $MASTER_NODE"
if [ $? -ne 0 ]; then
  echo "srun test failed"
  exit 1
fi

# Run the PyTorch distributed training script
echo "Running Python script"
srun python3 main-nn.py --kmer 10 \
--nodes "$NUM_NODES" \
--master-addr "$MASTER_ADDR" \
--master-port "$MASTER_PORT" \
--world-size "$WORLD_SIZE" \
--model "cnn" \
--in-mem 1 \
--lr 0.0004

if [ $? -ne 0 ]; then
  echo "srun Python script failed"
  exit 1
fi

echo "Finished running - goodbye from $HOSTNAME"

