"""
aggregate_features_by_depth.py
Author: Zakaria JOUILIL

Description :
    Aggregate decision tree feature usage by depth.

    Works for:
    - pretrained BERT (single CSV)
    - untrained BERT (directory of CSVs, one per seed)

Inputs and parameters :
    --mode : choose the pretrained or untrained processing mode
    --pretrained_csv : if mode is pretained, the input is a csv file containing the summary of the decision tree
    --untrained_dir : if mode is untrained, the input is the directoty containing 30 summaries of 30 decision trees (for each seed)
     
Outputs:
    - Aggregated CSV grouped by depth and feature (one file for each mode)

Usage for pretrained mode :
    python -m src.dt_analysis.aggregate_features_by_depth \
    --mode pretrained \
    --pretrained_csv outputs/dt_analysis/dt_features_depth_pretrained.csv \
    --output_csv outputs/dt_analysis/features_by_depth_pretrained.csv

Usage for untrained :
    python -m src.dt_analysis.aggregate_features_by_depth \
    --mode untrained \
    --untrained_dir outputs/dt_analysis/untrained \
    --output_csv outputs/dt_analysis/features_by_depth_untrained.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import glob
import ast
import os

# ---------------------------------------------------------
# PRETRAINED
# ---------------------------------------------------------
def process_pretrained(pretrained_csv, output_csv):
    df = pd.read_csv(pretrained_csv, sep=";")

    agg = (
        df.groupby(["depth", "feature"])
        .agg(
            Occurrence=("layer", "nunique"),
            Layers=("layer", lambda x: sorted(set(x))),
        )
        .reset_index()
        .sort_values(by=["depth", "Occurrence"], ascending=[True, False])
    )

    agg.rename(
        columns={
            "depth": "Depth",
            "feature": "Feature",
        },
        inplace=True,
    )

    agg["Layers"] = agg["Layers"].apply(lambda x: str(x))

    agg.to_csv(output_csv, sep=";", index=False)
    print(f"[OK] Saved pretrained aggregation: {output_csv}")


# ---------------------------------------------------------
# UNTRAINED
# ---------------------------------------------------------
def process_untrained(untrained_dir, output_csv, dataset):

    csv_files = sorted(
        untrained_dir.glob(
            f"dt_features_depth_{dataset}_untrained_seed_*.csv"
        )
    )

    if not csv_files:
        raise ValueError(
            f"No untrained CSV files found for dataset '{dataset}' "
            f"in {untrained_dir}"
        )

    used_seeds = set()

    for path in csv_files:
        fname = path.name
        seed = int(fname.split("_seed_")[1].replace(".csv", ""))
        used_seeds.add(seed)

    print(
        f"[INFO] Dataset: {dataset} | "
        f"Untrained aggregation using {len(used_seeds)} seeds"
    )

    if len(used_seeds) < 30:
        print("[WARN] Not all expected seeds were used")

    all_dfs = []

    for path in csv_files:
        df = pd.read_csv(path, sep=";")
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    agg = (
        df_all.groupby(["depth", "feature"])["seed"]
        .nunique()
        .reset_index(name="Seeds_Count")
        .sort_values(
            by=["depth", "Seeds_Count"],
            ascending=[True, False]
        )
    )

    agg.rename(
        columns={
            "depth": "Depth",
            "feature": "Feature",
        },
        inplace=True,
    )

    agg.to_csv(output_csv, sep=";", index=False)
    print(f"[OK] Saved untrained aggregation: {output_csv}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "mnli", "agnews", "arxiv"],
        help="Dataset being processed"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["pretrained", "untrained"],
        help="Processing mode"
    )

    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode

    # -----------------------------------------------------
    # INPUT DIRECTORY
    # -----------------------------------------------------
    dataset_dir = Path(f"outputs/dt_analysis/{dataset}")

    # -----------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------
    output_dir = dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = (
        output_dir /
        f"features_by_depth_{dataset}_{mode}.csv"
    )

    # -----------------------------------------------------
    # PRETRAINED
    # -----------------------------------------------------
    if mode == "pretrained":

        input_csv = (
            dataset_dir /
            "pretrained" /
            f"dt_features_depth_{dataset}_pretrained.csv"
        )

        if not input_csv.exists():
            raise FileNotFoundError(
                f"Pretrained input CSV not found: {input_csv}"
            )

        print(f"[INFO] Dataset: {dataset}")
        print("[INFO] Processing PRETRAINED model")

        process_pretrained(input_csv, output_csv)

    # -----------------------------------------------------
    # UNTRAINED
    # -----------------------------------------------------
    elif mode == "untrained":

        input_dir = dataset_dir / "untrained"

        if not input_dir.exists():
            raise FileNotFoundError(
                f"Untrained input directory not found: {input_dir}"
            )

        print(f"[INFO] Dataset: {dataset}")
        print("[INFO] Processing UNTRAINED model")

        process_untrained(
            untrained_dir=input_dir,
            output_csv=output_csv,
            dataset=dataset
        )


if __name__ == "__main__":
    main()