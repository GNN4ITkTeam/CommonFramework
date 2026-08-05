import torch
import os
import sys
from argparse import ArgumentParser

import pandas as pd

from acorn.utils.loading_utils import load_pyg
    

def dataframes_equal_ignoring_row_order(df1: pd.DataFrame, df2: pd.DataFrame):
    """Compares two dataframes for equality, ignoring the order of rows."""
    df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
    
    try:
        pd.testing.assert_frame_equal(df1_sorted, df2_sorted, check_exact=True)
        return True
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")
        return False


def compare_csv(ref_dir,test_dir):
    """
    Compare the reference and test directories for differences in .csv files.
    """
    # Get list of .csv files in both directories
    ref_files = {f for f in os.listdir(ref_dir) if f.endswith('.csv')}
    test_files = {f for f in os.listdir(test_dir) if f.endswith('.csv')}

    if not ref_files:
        print(f"No .csv files found in reference directory: {ref_dir}")
        return False

    if not test_files:
        print(f"No .csv files found in test directory: {test_dir}")
        return False

    # Check for missing files
    missing_in_test = ref_files - test_files
    missing_in_ref = test_files - ref_files

    if missing_in_test:
        print(f"Missing files in test directory: {missing_in_test}")
        return False
    if missing_in_ref:
        print(f"Missing files in reference directory: {missing_in_ref}")
        return False

    print(f"Comparing {len(test_files)} .csv files")

    # Compare contents of each file
    for file_name in ref_files:
        ref_file_path = os.path.join(ref_dir, file_name)
        test_file_path = os.path.join(test_dir, file_name)

        # Load the CSV files into DataFrames with max 100 hits per track
        # this is needed because the first line is not necessarily the longest track
        track_length_max = 100
        col_names = [f"hit_{i}" for i in range(track_length_max)]
        test_df = pd.read_csv(test_file_path, names=col_names, header=None)
        ref_df = pd.read_csv(ref_file_path, names=col_names, header=None)

        if not dataframes_equal_ignoring_row_order(test_df, ref_df):
            print(f"Difference found in file: {file_name}")
            return False

    print("All .csv files match between reference and test directories.")
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare .csv files in reference and test directories.")
    parser.add_argument("ref_dir", help="Path to the reference directory.")
    parser.add_argument("test_dir", help="Path to the test directory.")
    args = parser.parse_args()

    if not compare_csv(args.ref_dir, args.test_dir):
        print(f"Test failed: Differences in .csv files found.")
        sys.exit(1)

    print(f"Test passed.")
    sys.exit(0)