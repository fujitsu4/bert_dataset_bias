"""
extract_dt_statistics.py
Author: Zakaria JOUILIL

Description:
    Parse simplified positive decision tree rules (rules_simp_pos)
    and extract, PER MODEL / PER SEED:
        - discriminative features per depth and per layer
        - full decision patterns (feature paths)

This script:
    - works for pretrained BERT (single directory)
    - works for untrained BERT (multiple seed_xxxxxx subdirectories)
    - generates ONE CSV PER SEED (Option A)

It intentionally does NOT rely on rules_full and therefore does NOT
extract support or pos_rate.

Inputs and parameters :
    --input_dir : directory containing 12 decision trees (one per layer)
    --output_dir : directory where CSVs will be saved

Ouputs (for pretrained model OR for a seed of untrained model):
    dt_features_depth.csv : displays the discriminative features (per layer and per depth)
    dt_patterns.csv : displays patterns of discriminative features (per layer)

Usage :
    python -m src.dt_analysis.extract_dt_statistics \
    --input_dir outputs/decision_tree/pretrained
"""

import argparse
import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Parse one simplified rules file
# ---------------------------------------------------------
def parse_rules_file(path, layer, model, seed=None):
    rows_features = []
    rows_patterns = []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return rows_features, rows_patterns

    blocks = [b for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        features_in_path = []

        for line in lines:
            depth = line.count("|")

            m = re.search(r"---\s*([a-zA-Z0-9_]+)\s*[<>]=?", line)
            if m:
                feat = m.group(1)
                features_in_path.append(feat)

                rows_features.append({
                    "dataset": DATASET,
                    "model": model,
                    "seed": seed,
                    "layer": layer,
                    "depth": depth,
                    "feature": feat,
                })

        if features_in_path:
            rows_patterns.append({
                "dataset": DATASET,
                "model": model,
                "seed": seed,
                "layer": layer,
                "pattern": " -> ".join(features_in_path),
                "pattern_length": len(features_in_path),
            })

    return rows_features, rows_patterns


# ---------------------------------------------------------
# Process ONE directory (pretrained OR one seed)
# ---------------------------------------------------------
def process_one_model(input_dir, model, seed, output_dir):
    all_features = []
    all_patterns = []

    for layer in range(1, 13):
        fname = f"layer_{layer:02d}_rules_simp_pos.txt"
        fpath = input_dir / fname

        if not fpath.exists():
            print(f"[WARN] Missing file: {fpath}")
            continue

        feats, pats = parse_rules_file(
            fpath, layer=layer, model=model, seed=seed
        )

        all_features.extend(feats)
        all_patterns.extend(pats)

    if not all_features:
        print(f"[WARN] No rules found in {input_dir}")
        return

    # ------------------------------
    # FEATURES
    # ------------------------------
    df_features = pd.DataFrame(all_features)
    df_features = (
        df_features.drop_duplicates()
        .sort_values(by=["layer", "depth", "feature"])
        .reset_index(drop=True)
    )

    # ------------------------------
    # PATTERNS
    # ------------------------------
    df_patterns = pd.DataFrame(all_patterns)

    suffix = (
        f"{DATASET}_pretrained"
        if seed is None
        else f"{DATASET}_untrained_seed_{seed:06d}"
    )

    f_feat = output_dir / f"dt_features_depth_{suffix}.csv"
    f_pat = output_dir / f"dt_patterns_{suffix}.csv"

    df_features.to_csv(f_feat, sep=";", index=False)
    df_patterns.to_csv(f_pat, sep=";", index=False)

    print(f"[OK] Saved {f_feat}")
    print(f"[OK] Saved {f_pat}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    global DATASET

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "agnews", "arxiv"],
        help="Dataset being processed"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pretrained", "untrained"],
        help="BERT model type"
    )

    args = parser.parse_args()

    DATASET = args.dataset
    MODEL = args.model

    # -----------------------------------------------------
    # INPUT DIRECTORY
    # -----------------------------------------------------
    if MODEL == "pretrained":
        input_dir = Path(
            f"outputs/decision_tree/pretrained/{DATASET}"
        )
    else:
        input_dir = Path(
            f"outputs/decision_tree/untrained/{DATASET}"
        )

    # -----------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------
    output_dir = Path(
        f"outputs/dt_analysis/{DATASET}/{MODEL}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # -----------------------------------------------------
    # PRETRAINED
    # -----------------------------------------------------
    if MODEL == "pretrained":
        print(f"[INFO] Dataset: {DATASET}")
        print("[INFO] Processing PRETRAINED model")

        process_one_model(
            input_dir=input_dir,
            model="pretrained",
            seed=None,
            output_dir=output_dir
        )
        return

    # -----------------------------------------------------
    # UNTRAINED (seed directories)
    # -----------------------------------------------------
    print(f"[INFO] Dataset: {DATASET}")
    print("[INFO] Processing UNTRAINED model")

    seed_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    )

    if not seed_dirs:
        print(f"[WARN] No seed directories found in {input_dir}")
        return

    for d in seed_dirs:
        seed = int(d.name.replace("seed_", ""))

        out_feat = output_dir / (
            f"dt_features_depth_{DATASET}_untrained_seed_{seed:06d}.csv"
        )
        out_pat = output_dir / (
            f"dt_patterns_{DATASET}_untrained_seed_{seed:06d}.csv"
        )

        if out_feat.exists() and out_pat.exists():
            print(f"[SKIP] Seed {seed} already processed")
            continue

        print(f"[RUN] Processing seed {seed}")

        process_one_model(
            input_dir=d,
            model="untrained",
            seed=seed,
            output_dir=output_dir
        )


if __name__ == "__main__":
    DATASET = None
    main()