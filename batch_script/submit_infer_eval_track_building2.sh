#!/bin/bash

#SBATCH -A m2616
#SBATCH -C "cpu"
#SBATCH -q regular

#SBATCH --nodes=1

#SBATCH --time=08:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue

#SBATCH -o slurm_logs/%j-%x.out
#SBATCH --error slurm_logs/%j-%x.err

mkdir -p slurm_logs
export SLURM_CPU_BIND="cores"
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
echo -e "\nStarting Infer\n"

infer_config=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/track_building/track_building_infer_cc_and_walk.yaml
# checkpoint='/global/cfs/cdirs/m3443/data/GNN4ITK/CommonFrameworkExamples/2023_ttbar_uncorrelated/gnn/artifacts/best-14999421-val_loss=0.001299-epoch=5.ckpt'

eval_config=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/track_building/track_building_eval.yaml

srun g4i-infer $infer_config 

srun g4i-eval $eval_config

wait
