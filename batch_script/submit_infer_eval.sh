#!/bin/bash

#SBATCH -A m2616_g
#SBATCH -C "gpu&hbm40g"
#SBATCH -q regular

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
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

infer_config=/global/cfs/cdirs/m3443/usr/pmtuan/staged_ctf/trackml/examples/trackml/metric_learning_infer.yaml
checkpoint_ml='/global/cfs/cdirs/m2616/pmtuan/trackml/metric_learning/artifacts/best-17139231-signal_eff=0.996630-epoch=12.ckpt'

# checkpoint_gf='/global/cfs/cdirs/m2616/pmtuan/GNN4ITK/CommonFrameworkExamples/2023_ttbar_uncorrelated/fgnn/artifacts/best-13810901-auc=0.985425-epoch=360.ckpt'

eval_config=/global/cfs/cdirs/m3443/usr/pmtuan/staged_ctf/trackml/examples/trackml/metric_learning_eval.yaml

# eval_config_with_radius_split_edges='/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/metric_learning/metric_learning_eval_with_radius_splitedge.yaml'

srun g4i-infer $infer_config 

srun --exact -u -n 1 --gpus-per-task 1 g4i-eval $eval_config # --checkpoint $checkpoint_ml

# srun --exact -u -n 1 --gpus-per-task 1 g4i-eval $eval_config_with_radius_split_edges --checkpoint $checkpoint &

wait
