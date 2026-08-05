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

        test_df = pd.read_csv(test_file_path)
        ref_df = pd.read_csv(ref_file_path)

        if not dataframes_equal_ignoring_row_order(test_df, ref_df):
            print(f"Difference found in file: {file_name}")
            return False

    print("All .csv files match between reference and test directories.")
    return True


def compare_pyg(ref_dir, test_dir):
    """
    Compare the reference and test directories for differences in .pyg files.
    """
    # Get list of .pyg files in both directories
    ref_files = {f for f in os.listdir(ref_dir) if f.endswith('.pyg') or f.endswith('.pyg.gz')}
    test_files = {f for f in os.listdir(test_dir) if f.endswith('.pyg') or f.endswith('.pyg.gz')}

    if not ref_files:
        print(f"No .pyg files found in reference directory: {ref_dir}")
        return False

    if not test_files:
        print(f"No .pyg files found in test directory: {test_dir}")
        return False

    # Check for missing files
    tmp_ref_files = {f[:-3] if f.endswith('.gz') else f for f in ref_files}
    tmp_test_files = {f[:-3] if f.endswith('.gz') else f for f in test_files}
    missing_in_test = tmp_ref_files - tmp_test_files
    missing_in_ref = tmp_test_files - tmp_ref_files

    if missing_in_test:
        print(f"Missing files in test directory: {missing_in_test}")
        return False
    if missing_in_ref:
        print(f"Missing files in reference directory: {missing_in_ref}")
        return False

    print(f"Comparing {len(test_files)} .pyg files")

    # Compare contents of each file
    for file_name in ref_files:
        ref_file_path = os.path.join(ref_dir, file_name)
        test_file_path = os.path.join(test_dir, file_name)

        ref_tensor = load_pyg(ref_file_path)
        test_tensor = load_pyg(test_file_path)

        ref_keys = set(ref_tensor.keys())
        test_keys = set(test_tensor.keys())
        if ref_keys != test_keys:
            print(
                f"Keys mismatch for file {file_name}: "
                f"ref-only={sorted(ref_keys - test_keys)}, "
                f"test-only={sorted(test_keys - ref_keys)}"
            )
            return False

        for key in ref_tensor.keys():

            if key in [ 'config', 'event_id' ]:
                continue  # Skip config key as it may contain non-tensor data
                
            if not torch.allclose(ref_tensor[key], test_tensor[key], equal_nan=True):
                print(f"!!! Data mismatch for file {file_name}, key {key}")
                print(f"Ref data: {ref_tensor[key]}")
                print(f"Test data: {test_tensor[key]}")
                if ref_tensor[key].shape != test_tensor[key].shape:
                    print(
                        f"Shape mismatch: ref {ref_tensor[key].shape} "
                        f"vs test {test_tensor[key].shape}"
                    )
                else:
                    n_diff = int(torch.count_nonzero(ref_tensor[key] != test_tensor[key]))
                    print(f"{n_diff} differing elements (of {ref_tensor[key].numel()})")
                print(f"Difference found in file: {file_name}")
                return False


    print("All .pyg files match between reference and test directories.")
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare .csv and .pyg files in reference and test directories.")
    parser.add_argument("ref_dir", help="Path to the reference directory.")
    parser.add_argument("test_dir", help="Path to the test directory.")
    parser.add_argument("--skip_csv_comparison", action="store_true", help="Skip .csv files comparison, do only .pyg files.")
    args = parser.parse_args()


    for folder in ["trainset", "valset", "testset"]:
        ref_subdir = os.path.join(args.ref_dir, folder)
        test_subdir = os.path.join(args.test_dir, folder)

        if not args.skip_csv_comparison:
            if not compare_csv(ref_subdir, test_subdir):
                print(f"Test failed for {folder}: Differences in .csv files found.")
                sys.exit(1)

        if not compare_pyg(ref_subdir, test_subdir):
            print(f"Test failed for {folder}: Differences in .pyg files found.")
            sys.exit(1)

    print(f"Test passed for all folders: No differences in .csv and .pyg files found.")
    sys.exit(0)