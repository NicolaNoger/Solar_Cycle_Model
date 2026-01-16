#!/bin/bash
#SBATCH --job-name=hp_search_parallel
#SBATCH --time=01:00:00
#SBATCH --partition=earth-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=hp_search_%j.out
#SBATCH --error=hp_search_%j.err



# Load modules
module load USS/2022
module load gcc/9.4.0-pe5.34
module load python/3.9.12-pe5.34

# Activate virtual environment
source /cfs/earth/scratch/nogernic/HPC_Project/hpc_project/bin/activate

# Change to script directory
cd /cfs/earth/scratch/nogernic/HPC_Project


# PYTHONUNBUFFERED=1 ensures real-time output (no buffering)
PYTHONUNBUFFERED=1 python hyperparameter_search_parallel.py \
    --cores $SLURM_CPUS_PER_TASK \
    --iterations 20000 \
    --repeats 2


