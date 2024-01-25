#!/bin/bash
#SBATCH --account=bbhj-delta-gpu
#SBATCH --job-name=commonframework_uncorrelated2023
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH -o /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_gnn_batch_v4_CTD2023.out
#SBATCH -e /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_gnn_batch_v4_CTD2023.err

#CASE_NUM=`printf %02d $SLURM_ARRAY_TASK_ID`

source /u/jareddb2/GNN4ITk/commonframework/begin_env.sh

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
#echo "module_map_eval.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job.out

## GNN Train
#start=`date +%s`
#srun g4i-train gnn_train_noCut_IGNN2.yaml --checkpoint=/scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/gnn_noCut_1000events_h64_IGNN2_t2/artifacts/last-v1.ckpt
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "gnn_train.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_IGNN2_gnn_train_h64_t2.out

## GNN Infer
start=`date +%s`
g4i-infer gnn_infer_batch.yaml --checkpoint=/scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/gnn_1GeVpT_batchnorm_v4_CTD2023/artifacts/best--val_loss=0.003657-epoch=18.ckpt
end=`date +%s`
time_diff=$(($end-$start-18*3600))
echo "gnn_infer.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_gnn_batch_v4_CTD2023.out

## GNN Eval
start=`date +%s`
g4i-eval gnn_eval_batch.yaml
end=`date +%s`
time_diff=$(($end-$start-18*3600))
echo "gnn_eval.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_gnn_batch_v4_CTD2023.out

## Track Building Infer
#start=`date +%s`
#g4i-infer track_building_infer.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "track_building_infer.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_IGNN2_h64_t2.out

## Track Building Eval
#start=`date +%s`
#g4i-eval track_building_eval.yaml
#end=`date +%s`
#time_diff=$(($end-$start-18*3600))
#echo "track_building_eval.yaml Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023/job_logs/job_IGNN2_h64_t2.out
