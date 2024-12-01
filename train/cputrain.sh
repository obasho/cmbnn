#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=96:00:00
#SBATCH --job-name=cmb 
#SBATCH --error=out/job.%J.e
#SBATCH --output=out/job.%J.o
#SBATCH --partition=standard
#SBATCH --exclude=gpu005   # Exclude problematic node

## Command(s) to run:


# Set environment variables
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Load necessary modules

module load DL-Conda_3.7

# Initialize conda
conda init bash

# Source bashrc to apply conda changes
source ~/.bashrc

# Activate the conda environment
conda activate myenv39

# Display GPU information

# Set CUDA_VISIBLE_DEVICES to restrict the GPUs used by the script

# Check if CUDA is available in the environment

# Run the training for different k values on different GPUs in parallel
python3 maintrain.py
