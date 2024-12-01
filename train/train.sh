#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=cmbg 
#SBATCH --error=out/job.%J.e
#SBATCH --output=out/job.%J.o
#SBATCH --partition=gpu
#SBATCH --exclude=gpu005   # Exclude problematic node

## Command(s) to run:

# Initialize conda
conda init bash

# Source bashrc to apply conda changes
source ~/.bashrc

# Set environment variables
export PYSM_LOCAL_DATA=/pysm-data
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Load necessary modules

module load DL-Conda_3.7

# Activate the conda environment
conda activate myenv39

# Display GPU information
nvidia-smi

# Set CUDA_VISIBLE_DEVICES to restrict the GPUs used by the script
export CUDA_VISIBLE_DEVICES=0,1

# Check if CUDA is available in the environment
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run the training for different k values on different GPUs in parallel
python3 maintrain.py --k 0 --gpu 0 &  # Run on GPU 0
python3 maintrain.py --k 11 --gpu 1 &  # Run on GPU 1
# Wait for all background processes to finish
wait
