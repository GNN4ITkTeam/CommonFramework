
# input env name, if empty, use default acorn
name=acorn
if [ ! -z "$1" ]; then
    name=$1
fi

# this cript is written for a machine with GPU run on CUDA-12. One might need to find out the specific cuda version 
# on their GPU and install the appropriate torch build

conda create --yes --name $name python=3.10 
conda activate $name
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -r requirements-cpu.txt  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install -e .
python check_acorn.py