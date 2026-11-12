"""
aggregate_features_patterns.py
Author: Zakaria JOUILIL

Description :
    Aggregate decision tree features patterns.

    Works for:
    - pretrained BERT (single directory)
    - untrained BERT (30 directories , one per seed)

Inputs and parameters :
    --input_dir : input directory for pretrained or untrained decision trees
    --output_csv : path for output csv
   
Output:
    aggregate cv for decision tree features patterns 
    
Usage for pretrained mode :
    python -m src.dt_analysis.aggregate_features_patterns \
        --input_dir outputs/decision_tree/pretrained \
        --output_csv outputs/dt_analysis/features_patterns_pretrained.csv

Usage for untrained :
    python -m src.dt_analysis.aggregate_features_patterns \
        --input_dir /content/drive/MyDrive/results/decision_tree/untrained \
        --output_csv outputs/dt_analysis/features_patterns_untrained.csv
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


# ---------------------------------------------------------
# Parse ONE rules file (BLOCK-BASED)
# ---------------------------------------------------------
def parse_rules_file(path):
    patterns = []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return patterns

    # Each block = one complete root-to-leaf path
    blocks = [b for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        features = []

        for line in block.splitlines():
            m = re.search(r"---\s*([a-zA-Z0-9_]+)\s*[<>]=?", line)
            if m:
                features.append(m.group(1))

        if features:
            patterns.append(" -> ".join(features))

    return patterns


# ---------------------------------------------------------
# PRETRAINED
# ---------------------------------------------------------
def process_pretrained(input_dir, output_csv):
    stats = defaultdict(lambda: {"count": 0, "layers": set()})

    for layer in range(1, 13):
        fname = f"layer_{layer:02d}_rules_simp_pos.txt"
        fpath = input_dir / fname

        if not fpath.exists():
            print(f"[WARN] Missing file: {fpath}")
            continue

        patterns = parse_rules_file(fpath)

        for p in patterns:
            stats[p]["count"] += 1
            stats[p]["layers"].add(layer)

    rows = []
    for pattern, data in stats.items():
        rows.append({
            "pattern": pattern,
            "count": data["count"],
            "layers": sorted(data["layers"])
        })

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.to_csv(output_csv, sep=";", index=False)
    print(f"[OK] Saved pretrained patterns → {output_csv}")


# ---------------------------------------------------------
# UNTRAINED
# ---------------------------------------------------------
def process_untrained(input_dir, output_csv):
    stats = defaultdict(lambda: {"occurrences": 0, "seeds": set()})

    for seed_dir in sorted(input_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue

        seed = int(seed_dir.name.replace("seed_", ""))

        for layer in range(1, 13):
            fname = f"layer_{layer:02d}_rules_simp_pos.txt"
            fpath = seed_dir / fname

            if not fpath.exists():
                continue

            patterns = parse_rules_file(fpath)

            for p in patterns:
                stats[p]["occurrences"] += 1
                stats[p]["seeds"].add(seed)

    rows = []
    for pattern, data in stats.items():
        rows.append({
            "pattern": pattern,
            "occurrences": data["occurrences"],
            "seed_count": len(data["seeds"])
        })

    df = pd.DataFrame(rows).sort_values(["seed_count", "occurrences"],ascending=False)
    df.to_csv(output_csv, sep=";", index=False)
    print(f"[OK] Saved untrained patterns → {output_csv}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "agnews", "arxiv"],
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

    # Input directory
    if mode == "pretrained":
        input_dir = Path(
            f"outputs/decision_tree/pretrained/{dataset}"
        )
    else:
        input_dir = Path(
            f"outputs/decision_tree/untrained/{dataset}"
        )

    # Output directory
    output_dir = Path(f"outputs/dt_analysis/{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"features_patterns_{dataset}_{mode}.csv"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if mode == "pretrained":
        print(f"[INFO] Dataset: {dataset} | PRETRAINED")
        process_pretrained(input_dir, output_csv)

    else:
        print(f"[INFO] Dataset: {dataset} | UNTRAINED")
        process_untrained(input_dir, output_csv)


if __name__ == "__main__":
    main()