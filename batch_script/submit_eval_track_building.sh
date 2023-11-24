#!/bin/bash

#SBATCH -A m2616
#SBATCH -C "cpu"
#SBATCH -q regular

#SBATCH --nodes=1

#SBATCH --time=03:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue

#SBATCH -o slurm_logs/%j-%x.out
#SBATCH --error slurm_logs/%j-%x.err

mkdir -p slurm_logs
export SLURM_CPU_BIND="cores"
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting Infer\n"

eval_config=$1

srun g4i-eval $1 

srun g4i-eval $2

wait
