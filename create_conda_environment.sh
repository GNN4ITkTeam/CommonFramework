#!/usr/bin/env bash

# Input env name, if empty, use default acorn.
# Pass --cuda-ext or set ACORN_BUILD_CUDA_EXT=1 to compile optional CUDA/C++ extensions.
name=acorn
build_cuda_ext=${ACORN_BUILD_CUDA_EXT:-0}

for arg in "$@"; do
    case "$arg" in
        --cuda-ext)
            build_cuda_ext=1
            ;;
        *)
            name=$arg
            ;;
    esac
done

# This script is written for a machine with GPU run on CUDA-12. One might need to find out the specific cuda version 
# on their GPU and install the appropriate torch build

conda create --yes --name "$name" python=3.10 
conda activate "$name"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 
if [ "$build_cuda_ext" = "1" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: --cuda-ext requires nvcc on PATH. Load a CUDA toolkit module or set CUDA_HOME, then rerun."
        return 1 2>/dev/null || exit 1
    fi
    ACORN_BUILD_CUDA_EXT=1 pip install -e . --no-build-isolation
else
    pip install -e .
fi
python check_acorn.py
