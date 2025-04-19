#!/bin/bash
#SBATCH --job-name=RMT_VAT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12gb
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupart_24hour
#SBATCH --time=20:00:00
#SBATCH --output=logs/rmt_%j.out

module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
nvidia-smi
echo "Active Conda environment: $CONDA_DEFAULT_ENV"


./train_50tcsm.sh
