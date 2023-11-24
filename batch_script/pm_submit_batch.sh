#!/bin/bash

#SBATCH -A m2616_g
#SBATCH -C "gpu&hbm80g"
#SBATCH -q regular

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --signal=SIGUSR1@240
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/pm-slurm-%j-%x.out
#SBATCH --error slurm_logs/pm-slurm-%j-%x.err

export SLURM_CPU_BIND="cores"
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting Training with arguments $@\n"

mkdir -p slurm_logs

srun g4i-train $@

wait
