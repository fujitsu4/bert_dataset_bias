"""
merge.py
Author : Zakaria JOUILIL

Description: Rename and merge features_patterns for each dataset
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge pattern files from three datasets."
    )

    parser.add_argument("--snli", required=True,
                        help="Path to the SNLI patterns CSV file")
    parser.add_argument("--arxiv", required=True,
                        help="Path to the arXiv patterns CSV file")
    parser.add_argument("--agnews", required=True,
                        help="Path to the AG News patterns CSV file")
    parser.add_argument("--output", required=True,
                        help="Path to the merged output CSV file")

    args = parser.parse_args()

    datasets = {
        "snli": args.snli,
        "arxiv": args.arxiv,
        "agnews": args.agnews
    }

    dataframes = []

    for dataset, filepath in datasets.items():

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"File not found for {dataset}: {filepath}"
            )

        df = pd.read_csv(filepath, sep=";")

        expected_columns = {"pattern", "count", "layers"}
        if not expected_columns.issubset(df.columns):
            raise ValueError(
                f"Invalid columns in {filepath}. "
                f"Expected: {expected_columns}"
            )

        df.insert(0, "dataset", dataset)
        dataframes.append(df)

        print(f"{dataset}: {len(df)} patterns loaded.")

    merged_df = pd.concat(dataframes, ignore_index=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    merged_df.to_csv(args.output, sep=";", index=False)

    print("\nMerge completed successfully.")
    print(f"Total patterns: {len(merged_df)}")
    print(f"Output file: {args.output}")
    print("\nPatterns per dataset:")
    print(merged_df.groupby("dataset").size())


if __name__ == "__main__":
    main()
