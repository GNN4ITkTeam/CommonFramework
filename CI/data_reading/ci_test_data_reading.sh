#!/usr/bin/bash
set -e

skip_csv_comparison=${1:-false}

if [ "$skip_csv_comparison" = true ]; then
    config_file=ci_data_reader_skip_csv.yml
else
    config_file=ci_data_reader.yml
fi

ref_dir=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/acorn_data_reading_output/ci_ref/feature_store_ci
test_dir=feature_store_ci

# Remove directory if it exists
if [ -d "$test_dir" ]; then
    echo "Removing existing $test_dir directory to proceed to new test"
    rm -rf $test_dir
fi

echo "--------------------------"
echo "Running data reading stage"
echo "--------------------------"
acorn infer $config_file

ls -la $test_dir/trainset
ls -la $test_dir/valset
ls -la $test_dir/testset

echo "----------------------"
echo "Comparing to reference"
echo "----------------------"
if [ "$skip_csv_comparison" = true ]; then
    echo "Skipping .csv comparison as requested."
    python diff.py $ref_dir $test_dir --skip_csv_comparison
else
    python diff.py $ref_dir $test_dir
fi
