#!/bin/bash
gpu_requirement="gpu&hbm80g"
# gpu_requirement="gpu"
salloc -A m2616_g -C $gpu_requirement -q interactive --nodes 1 --ntasks-per-node 4 --gpus-per-task 1 --cpus-per-task 32 --mem-per-gpu 32G --time 02:00:00 --gpu-bind=none --signal=SIGUSR1@180 #--image=tuanpham1503/torch_conda:0.4
