import os
import sys
from argparse import ArgumentParser

from csv_comparison import compare_csv
from graph_comparison import compare_pyg


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compare .pyg files in reference and test directories."
    )
    parser.add_argument("ref_dir", help="Path to the reference directory.")
    parser.add_argument("test_dir", help="Path to the test directory.")
    parser.add_argument(
        "stage",
        choices=["data_reading", "graph_building", "track_building"],
        help="Stage of the pipeline to compare: 'data_reading' or 'graph_building'.",
    )
    parser.add_argument(
        "--skip_csv_comparison",
        action="store_true",
        help="Skip .csv files comparison, do only .pyg files.",
    )
    args = parser.parse_args()

    skip_csv_comparison = args.skip_csv_comparison
    track_length_max = -1  # Default value, no limit on track length

    if args.stage == "data_reading":
        graph_is_built = False
        folders_to_compare = ["trainset", "valset", "testset"]
    elif args.stage == "graph_building":
        graph_is_built = True
        folders_to_compare = ["valset"]
        skip_csv_comparison = (
            True  # Always skip CSV comparison for graph_building stage
        )
    elif args.stage == "track_building":
        folders_to_compare = ["valset_tracks"]
        track_length_max = 100  # Limit to 100 hits per track comparison

    for folder in folders_to_compare:
        ref_subdir = os.path.join(args.ref_dir, folder)
        test_subdir = os.path.join(args.test_dir, folder)

        if not skip_csv_comparison and not compare_csv(
            ref_subdir, test_subdir, track_length_max
        ):
            print(f"Test failed for {folder}: Differences in .csv files found.")
            sys.exit(1)

        if args.stage != "track_building" and not compare_pyg(
            ref_subdir, test_subdir, graph_is_built
        ):
            print("Test failed for : differences in .pyg files found.")
            sys.exit(1)

    print("Test passed: No differences in .pyg files found.")
    sys.exit(0)
