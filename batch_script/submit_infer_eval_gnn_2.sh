#!/bin/bash

#SBATCH -A m2616_g
#SBATCH -C "gpu&hbm80g"
#SBATCH -q regular

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/%j-%x.out
#SBATCH --error slurm_logs/%j-%x.err

mkdir -p slurm_logs
export SLURM_CPU_BIND="cores"
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting Infer\n"

infer_config=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/homo/ignn2/infer_ignn2_pyg.yaml
checkpoint='/global/cfs/cdirs/m2616/pmtuan/GNN4ITK/CommonFrameworkExamples/2023_ttbar_uncorrelated/gnn/artifacts/best-15694548-val_loss=0.001350-epoch=21.ckpt'

eval_config=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/homo/ignn2/eval_ignn2_pyg.yaml

# srun g4i-infer $infer_config --checkpoint $checkpoint

srun --exact -u -n 1 --gpus-per-task 1 g4i-eval $eval_config --checkpoint $checkpoint 

wait
