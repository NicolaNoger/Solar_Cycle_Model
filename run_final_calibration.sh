#!/bin/bash
#SBATCH --job-name=final_calib
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_calibration_%j.out
#SBATCH --error=slurm_calibration_%j.err



cd /cfs/earth/scratch/nogernic/HPC_Project

# Load required modules (for Python 3.9)
module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load python/3.9.12-pe5.34

# Activate virtual environment (new one with all packages)
source hpc_project_new/bin/activate



python main_final_calibration.py


echo "Job finished at: $(date)"

