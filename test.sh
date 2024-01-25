#!/bin/bash
#SBATCH --account=bbhj-delta-gpu
#SBATCH --job-name=acorn_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --time=0:15:00
#SBATCH -o /u/jareddb2/GNN4ITk/acorn/job.out
#SBATCH -e /u/jareddb2/GNN4ITk/acorn/job.err

source /u/jareddb2/GNN4ITk/acorn/begin_env.sh
python /u/jareddb2/GNN4ITk/acorn/check_acorn.py

## Data Reader
#start=`date +%s`
#g4i-infer data_reader/data_reader_$CASE_NUM.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "data_reader.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job-$SLURM_ARRAY_TASK_ID.out

## Module Map Inference
#start=`date +%s`
#g4i-infer module_map_infer.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "module_map_infer.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_mm_v3.out

## Module Map Eval
#start=`date +%s`
#g4i-eval module_map_eval.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "module_map_eval.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_mm_CTD2023.out

## GNN Train
#start=`date +%s`
#srun g4i-train gnn_train_noCut_IGNN2.yaml --checkpoint=/scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/gnn_noCut_1000events_h64_IGNN2_t2/artifacts/last-v1.ckpt
##end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "gnn_train.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_IGNN2_gnn_train_h64_t2.out

## GNN Infer
#start=`date +%s`
#g4i-infer gnn_infer.yaml --checkpoint=/scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/gnn_layer_CTD2023/artifacts/best--val_loss=0.000492-epoch=0.ckpt
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "gnn_infer.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_gnn_layer_CTD2023.out

## GNN Eval
#start=`date +%s`
#g4i-eval gnn_eval.yaml --checkpoint=/scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/gnn_layer_CTD2023/artifacts/best--val_loss=0.000492-epoch=0.ckpt
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "gnn_eval.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_gnn_layer_CTD2023.out

## Track Building Infer
#start=`date +%s`
#g4i-infer track_building_infer.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "track_building_infer.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_track_CTD2023.out

## Track Building Eval
#start=`date +%s`
#g4i-eval track_building_eval.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "track_building_eval.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_track_CTD2023.out
