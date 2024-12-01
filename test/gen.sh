#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00 
#SBATCH --job-name=cmb 
#SBATCH --error=out/job.%J.e
#SBATCH --output=out/job.%J.o
#SBATCH --partition=standard
## Command(s) to run (example):
export PYSM_LOCAL_DATA=/pysm-data
module load DL-Conda_3.7
module load openmpi/4.1.4
export OMPI_MCA_btl=^openib

module load  spack/0.20.0
source /home/apps/spack/share/spack/setup-env.sh
spack load /keplgji
spack load /2bfym2p
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/obashom.sps.iitmandi/.conda/envs/myenv/lib
conda init bash
source activate myenv39
python3 -u prediict.py
