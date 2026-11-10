"""
analyze_patterns.py
Author : Zakaria JOUILIL

Description: Analyze features_patterns for each dataset
"""

import argparse
import ast
import itertools
from collections import Counter
from pathlib import Path

import pandas as pd


EXPECTED_DATASETS = ["snli", "arxiv", "agnews"]


def parse_layers(value):
    """Convert a CSV layers value into a list of integers."""
    if isinstance(value, list):
        return [int(layer) for layer in value]

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple)):
            return [int(layer) for layer in parsed]
        return [int(parsed)]
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(f"Invalid layers value: {value}")


def load_data(input_path):
    """Load and validate the merged pretrained patterns file."""
    df = pd.read_csv(input_path, sep=";")

    required_columns = {"dataset", "pattern", "count", "layers"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["pattern"] = df["pattern"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="raise")
    df["layers"] = df["layers"].apply(parse_layers)

    if (df["count"] < 0).any():
        raise ValueError("Pattern counts cannot be negative.")

    found_datasets = set(df["dataset"])
    missing_datasets = set(EXPECTED_DATASETS) - found_datasets

    if missing_datasets:
        raise ValueError(
            f"Missing datasets in input file: {sorted(missing_datasets)}"
        )

    unexpected = found_datasets - set(EXPECTED_DATASETS)
    if unexpected:
        raise ValueError(
            f"Unexpected dataset labels: {sorted(unexpected)}"
        )

    # Consolidate duplicated dataset-pattern rows if any.
    df = (
        df.groupby(["dataset", "pattern"], as_index=False)
        .agg(
            count=("count", "sum"),
            layers=("layers", lambda values: sorted(
                set(layer for group in values for layer in group)
            ))
        )
    )

    return df


def save_csv(df, output_dir, filename):
    """Save a DataFrame using the project's semicolon CSV convention."""
    path = output_dir / filename
    df.to_csv(path, sep=";", index=False)
    print(f"Saved: {path} ({len(df)} rows)")


# ---------------------------------------------------------
# 1. COMMON PATTERNS
# ---------------------------------------------------------

def analyze_common_patterns(df, output_dir):
    pattern_sets = {
        dataset: set(
            df.loc[df["dataset"] == dataset, "pattern"]
        )
        for dataset in EXPECTED_DATASETS
    }

    common = set.intersection(*pattern_sets.values())

    rows = []
    for pattern in sorted(common):
        row = {"pattern": pattern}

        for dataset in EXPECTED_DATASETS:
            match = df[
                (df["dataset"] == dataset)
                & (df["pattern"] == pattern)
            ]
            row[f"{dataset}_count"] = int(match["count"].sum())
            row[f"{dataset}_layers"] = (
                match.iloc[0]["layers"] if not match.empty else []
            )

        rows.append(row)

    result = pd.DataFrame(rows)
    save_csv(result, output_dir, "01_patterns_common.csv")


# ---------------------------------------------------------
# 2. DATASET-SPECIFIC PATTERNS
# ---------------------------------------------------------

def analyze_specific_patterns(df, output_dir):
    rows = []

    for _, row in df.iterrows():
        pattern = row["pattern"]
        dataset = row["dataset"]

        other_datasets = [
            other for other in EXPECTED_DATASETS
            if other != dataset
        ]

        appears_elsewhere = any(
            (
                (df["dataset"] == other)
                & (df["pattern"] == pattern)
            ).any()
            for other in other_datasets
        )

        if not appears_elsewhere:
            rows.append({
                "dataset": dataset,
                "pattern": pattern,
                "count": int(row["count"]),
                "layers": row["layers"]
            })

    result = pd.DataFrame(rows)
    save_csv(result, output_dir, "02_patterns_specific.csv")


# ---------------------------------------------------------
# 3. PATTERN FREQUENCY
# ---------------------------------------------------------

def analyze_pattern_frequency(df, output_dir):
    result = df.copy()

    totals = (
        result.groupby("dataset")["count"]
        .sum()
        .to_dict()
    )

    result["dataset_total_occurrences"] = (
        result["dataset"].map(totals)
    )

    result["relative_frequency"] = (
        result["count"] / result["dataset_total_occurrences"]
    )

    result = result.rename(columns={
        "count": "occurrences"
    })

    result = result.sort_values(
        ["dataset", "occurrences", "pattern"],
        ascending=[True, False, True]
    )

    save_csv(result, output_dir, "03_patterns_frequency.csv")


# ---------------------------------------------------------
# 4. CROSS-DOMAIN CONSISTENCY: JACCARD SIMILARITY
# ---------------------------------------------------------

def analyze_jaccard_similarity(df, output_dir):
    pattern_sets = {
        dataset: set(
            df.loc[df["dataset"] == dataset, "pattern"]
        )
        for dataset in EXPECTED_DATASETS
    }

    rows = []

    for dataset_a, dataset_b in itertools.combinations(
        EXPECTED_DATASETS, 2
    ):
        intersection = pattern_sets[dataset_a] & pattern_sets[dataset_b]
        union = pattern_sets[dataset_a] | pattern_sets[dataset_b]

        jaccard = len(intersection) / len(union) if union else 0.0

        rows.append({
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "common_patterns": len(intersection),
            "union_patterns": len(union),
            "jaccard_similarity": jaccard
        })

    result = pd.DataFrame(rows)
    save_csv(result, output_dir, "04_dataset_jaccard_similarity.csv")


# ---------------------------------------------------------
# 5. STRUCTURAL ANALYSIS OF DECISION PATHS
# ---------------------------------------------------------

def analyze_path_structure(df, output_dir):
    length_rows = []
    root_counter = Counter()
    transition_counter = Counter()
    subpath_counter = Counter()

    for _, row in df.iterrows():
        dataset = row["dataset"]
        pattern = row["pattern"]
        occurrences = int(row["count"])

        features = [
            feature.strip()
            for feature in pattern.split("->")
        ]

        if not features or any(not feature for feature in features):
            raise ValueError(f"Invalid pattern: {pattern}")

        path_length = len(features)

        length_rows.append({
            "dataset": dataset,
            "pattern": pattern,
            "path_length": path_length,
            "occurrences": occurrences
        })

        # Root feature
        root_counter[(dataset, features[0])] += occurrences

        # Consecutive feature transitions
        for source, target in zip(features, features[1:]):
            transition_counter[
                (dataset, source, target)
            ] += occurrences

        # All contiguous subpaths of length 1 to 4
        for n in range(1, min(4, path_length) + 1):
            for start in range(path_length - n + 1):
                subpath = tuple(features[start:start + n])
                subpath_counter[(dataset, subpath)] += occurrences

    # Path length summary, weighted by pattern occurrences
    lengths = pd.DataFrame(length_rows)

    summary_rows = []
    for dataset, group in lengths.groupby("dataset"):
        weights = group["occurrences"]
        total_weight = weights.sum()

        weighted_mean = (
            (group["path_length"] * weights).sum() / total_weight
            if total_weight else 0
        )

        summary_rows.append({
            "dataset": dataset,
            "unique_patterns": len(group),
            "total_occurrences": int(total_weight),
            "mean_path_length_weighted": weighted_mean,
            "min_path_length": int(group["path_length"].min()),
            "max_path_length": int(group["path_length"].max())
        })

    save_csv(
        pd.DataFrame(summary_rows),
        output_dir,
        "05a_path_length_summary.csv"
    )

    # Root features
    roots = pd.DataFrame([
        {
            "dataset": dataset,
            "root_feature": feature,
            "occurrences": count
        }
        for (dataset, feature), count in root_counter.items()
    ])

    if not roots.empty:
        roots = roots.sort_values(
            ["dataset", "occurrences"],
            ascending=[True, False]
        )

    save_csv(roots, output_dir, "05b_root_features.csv")

    # Consecutive transitions
    transitions = pd.DataFrame([
        {
            "dataset": dataset,
            "source_feature": source,
            "target_feature": target,
            "transition": f"{source} -> {target}",
            "occurrences": count
        }
        for (dataset, source, target), count
        in transition_counter.items()
    ])

    if not transitions.empty:
        transitions = transitions.sort_values(
            ["dataset", "occurrences"],
            ascending=[True, False]
        )

    save_csv(
        transitions,
        output_dir,
        "05c_feature_transitions.csv"
    )

    # Contiguous subpaths
    subpaths = pd.DataFrame([
        {
            "dataset": dataset,
            "subpath": " -> ".join(subpath),
            "subpath_length": len(subpath),
            "occurrences": count
        }
        for (dataset, subpath), count in subpath_counter.items()
    ])

    if not subpaths.empty:
        subpaths = subpaths.sort_values(
            ["dataset", "subpath_length", "occurrences"],
            ascending=[True, True, False]
        )

    save_csv(
        subpaths,
        output_dir,
        "05d_contiguous_subpaths.csv"
    )

    # Subpaths present in all three datasets
    subpath_sets = {
        dataset: set(
            subpaths.loc[
                subpaths["dataset"] == dataset, "subpath"
            ]
        )
        for dataset in EXPECTED_DATASETS
    }

    common_subpaths = (
        set.intersection(*subpath_sets.values())
        if all(subpath_sets.values())
        else set()
    )

    common_rows = []
    for subpath in sorted(common_subpaths):
        row = {"subpath": subpath}

        for dataset in EXPECTED_DATASETS:
            match = subpaths[
                (subpaths["dataset"] == dataset)
                & (subpaths["subpath"] == subpath)
            ]
            row[f"{dataset}_occurrences"] = int(
                match["occurrences"].sum()
            )

        common_rows.append(row)

    save_csv(
        pd.DataFrame(common_rows),
        output_dir,
        "05e_common_subpaths.csv"
    )


# ---------------------------------------------------------
# 6. LAYER DISTRIBUTION
# ---------------------------------------------------------

def analyze_layers(df, output_dir):
    rows = []

    for _, row in df.iterrows():
        for layer in row["layers"]:
            rows.append({
                "dataset": row["dataset"],
                "pattern": row["pattern"],
                "layer": layer,
                "pattern_occurrences": int(row["count"])
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["dataset", "layer", "pattern"]
        )

    save_csv(result, output_dir, "06_patterns_by_layer.csv")

    # Feature-level layer presence, without assigning pattern
    # occurrence counts to individual layers.
    feature_rows = []

    for _, row in df.iterrows():
        features = set(
            feature.strip()
            for feature in row["pattern"].split("->")
        )

        for layer in row["layers"]:
            for feature in features:
                feature_rows.append({
                    "dataset": row["dataset"],
                    "layer": layer,
                    "feature": feature
                })

    feature_layers = pd.DataFrame(feature_rows)

    if not feature_layers.empty:
        feature_layers = (
            feature_layers
            .groupby(["dataset", "layer", "feature"])
            .size()
            .reset_index(name="distinct_pattern_count")
        )

    save_csv(
        feature_layers,
        output_dir,
        "06b_feature_presence_by_layer.csv"
    )


# ---------------------------------------------------------
# 7. FEATURE FREQUENCY
# ---------------------------------------------------------

def analyze_feature_frequency(df, output_dir):
    counters = Counter()
    pattern_counters = Counter()
    dataset_presence = {}

    for _, row in df.iterrows():
        dataset = row["dataset"]
        pattern = row["pattern"]
        occurrences = int(row["count"])

        # Preserve repeated features within a path for total
        # feature appearances; use a set for distinct path counts.
        features = [
            feature.strip()
            for feature in pattern.split("->")
        ]

        for feature, feature_count in Counter(features).items():
            counters[(dataset, feature)] += (
                feature_count * occurrences
            )
            pattern_counters[(dataset, feature)] += 1

            dataset_presence.setdefault(feature, set()).add(dataset)

    rows = []

    for (dataset, feature), total in counters.items():
        rows.append({
            "dataset": dataset,
            "feature": feature,
            "weighted_feature_occurrences": total,
            "distinct_patterns_containing_feature": (
                pattern_counters[(dataset, feature)]
            ),
            "datasets_containing_feature": len(
                dataset_presence[feature]
            )
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["dataset", "weighted_feature_occurrences"],
            ascending=[True, False]
        )

    save_csv(result, output_dir, "07_feature_frequency.csv")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantitative analysis of pretrained BERT "
            "decision-tree patterns across three datasets."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Merged pretrained patterns CSV file."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where analysis CSV files will be saved."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained BERT pattern data...")
    df = load_data(input_path)

    print("\nInput overview:")
    print(df.groupby("dataset").agg(
        unique_patterns=("pattern", "nunique"),
        total_occurrences=("count", "sum")
    ))

    print("\nRunning the seven analyses...")

    analyze_common_patterns(df, output_dir)
    analyze_specific_patterns(df, output_dir)
    analyze_pattern_frequency(df, output_dir)
    analyze_jaccard_similarity(df, output_dir)
    analyze_path_structure(df, output_dir)
    analyze_layers(df, output_dir)
    analyze_feature_frequency(df, output_dir)

    print("\nAll analyses completed successfully.")
    print(f"Results directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
