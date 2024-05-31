#!/bin/bash

#SBATCH -A m2616 -q regular
#SBATCH -C gpu&hbm80g
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J Acorn-train
#SBATCH --gpu-bind=none
#SBATCH --comment=96:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
eval "$(conda shell.bash hook)"

# module load python/3.9-anaconda-2021.11
conda activate acorn

export SLURM_CPU_BIND="cores"
export WANDB__SERVICE_WAIT=300
echo -e "\nStarting training\n"

# Single GPU training
srun acorn train $@