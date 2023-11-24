#!/bin/bash

#SBATCH -A m2616_g
#SBATCH -C "gpu&hbm80g"
#SBATCH -q regular

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/%j-%x.out
#SBATCH --error slurm_logs/%j-%x.err

mkdir -p slurm_logs
export SLURM_CPU_BIND="cores"
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting Eval\n"

infer_config=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/fgnn/infer.yaml
checkpoint=/global/cfs/cdirs/m3443/data/GNN4ITK/CommonFrameworkExamples/2023_ttbar_uncorrelated/fgnn/artifacts/best-13810901-auc=0.985425-epoch=360.ckpt

eval_config=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/fgnn/eval.yaml

srun g4i-eval /global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/fgnn/eval_trainset.yaml --checkpoint $checkpoint 

mv /pscratch/sd/p/pmtuan/2023_ttbar_uncorrelated/gfilter_13810901_epoch360/testset /pscratch/sd/p/pmtuan/2023_ttbar_uncorrelated/gfilter_13810901_epoch360/testset_
mv /pscratch/sd/p/pmtuan/2023_ttbar_uncorrelated/gfilter_13810901_epoch360/trainset /pscratch/sd/p/pmtuan/2023_ttbar_uncorrelated/gfilter_13810901_epoch360/testset
mv /pscratch/sd/p/pmtuan/2023_ttbar_uncorrelated/gfilter_13810901_epoch360/testset_ /pscratch/sd/p/pmtuan/2023_ttbar_uncorrelated/gfilter_13810901_epoch360/trainset

srun g4i-eval $eval_config --checkpoint $checkpoint



wait
