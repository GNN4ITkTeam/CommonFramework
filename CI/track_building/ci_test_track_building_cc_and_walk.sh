#!/usr/bin/bash
set -e

ref_dir=/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/ATLAS-P2-RUN4-03-00-00_Rel.24/ci_ref/tracks/cc_and_walk
test_dir=tracks_ci_cc_and_walk

# Remove directory if it exists
if [ -d "$test_dir" ]; then
    echo "Removing existing $test_dir directory to proceed to new test"
    rm -rf $test_dir
fi

echo "----------------------------"
echo "Running track building stage"
echo "----------------------------"
acorn infer ci_track_building_cc_and_walk.yml

ls -la $test_dir/valset_tracks

echo "----------------------"
echo "Comparing to reference"
echo "----------------------"
python diff.py $ref_dir $test_dir/valset_tracks