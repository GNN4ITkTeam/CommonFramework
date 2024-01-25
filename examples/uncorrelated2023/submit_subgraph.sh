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
#SBATCH -o /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023_split1x2_o1/job_subgraph_val.out
#SBATCH -e /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023_split1x2_o1/job_subgraph_val.err

source /u/jareddb2/GNN4ITk/commonframework/begin_env.sh
#pip install torch-geometric

## Subgraph Construction
start=`date +%s`
python /u/jareddb2/GNN4ITk/commonframework/gnn4itk_cf/stages/graph_construction/subgraph_construction_stage_v2.py
end=`date +%s`
time_diff=$(($end-$start-18*3600))
echo "subgraph_construction_stage.py Elapsed Time: `date -d @$time_diff +%H:%M:%S`" >> /scratch/bbhj/jareddb2/GNN4ITk/commonframework_data_output/uncorrelated2023_split1x2_o1/job_subgraph_val.out
