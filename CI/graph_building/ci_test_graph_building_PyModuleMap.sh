#!/bin/bash
set -e

ref_dir=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/ci_ref/unscored_graphs/ttbar_pu0
test_dir=graphs_ci_PyModuleMap

# Remove directory if it exists
if [ -d "$test_dir" ]; then
    echo "Removing existing $test_dir directory to proceed to new test"
    rm -rf $test_dir
fi

nvidia-smi

echo "---------------------------------------------"
echo "Running graph building stage with PyModuleMap"
echo "---------------------------------------------"
acorn infer ci_graph_building_PyModuleMap.yml

ls -la $test_dir/valset

echo "----------------------"
echo "Comparing to reference"
echo "----------------------"
python ../diff.py $ref_dir $test_dir graph_building
