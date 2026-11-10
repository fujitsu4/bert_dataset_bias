"""
generate_pretrained_dt_figures.
Author : Zakaria JOUILIL

Description: Generate three figures from pretrained BERT decision-tree TXT files.

Expected:
outputs/decision_tree/pretrained/
    snli/layer_01_rules_full.txt ... layer_12_rules_full.txt
    arxiv/layer_01_rules_full.txt ... layer_12_rules_full.txt
    agnews/layer_01_rules_full.txt ... layer_12_rules_full.txt

Only *_rules_full.txt files are read. POS-only files are deliberately excluded.

Usage:
!python -m src.dataset_analysis.generate_pretrained_dt_figures
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["snli", "arxiv", "agnews"]
LAYERS = list(range(1, 13))

FEATURE_FAMILIES = {
    "Contextual": ["next_pos_id", "prev_pos_id"],
    "Positional": ["relative_position", "position_in_sentence"],
    "Distributional": [
        "token_rank", "word_burstiness",
        "word_sentence_frequency", "token_relative_frequency",
    ],
    "Linguistic": [
        "pos_id", "dep_id", "depth_in_tree", "num_dependents",
        "is_stopword", "is_pronoun",
    ],
    "Token": ["num_subtokens", "token_length", "is_punctuation"],
}

SPLIT_RE = re.compile(
    r"^\s*\|---\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:<=|>)\s+"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)
LEAF_RE = re.compile(r"^\s*\|---\s+class:\s*[01]\s*$")


def parse_tree_file(path):
    import re

    split_re = re.compile(
        r"^\|---\s+([A-Za-z0-9_]+)\s+(<=|>)\s+([-+]?\d*\.?\d+)"
    )
    class_re = re.compile(r"^\|---\s+class:\s*([01])")

    branches = {}
    paths = []
    root = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            # Ignore everything after the tree
            if "=== Per-leaf statistics" in raw_line:
                break

            # Ignore empty lines
            if not raw_line.strip():
                continue

            # Tree depth is the number of vertical bars minus one
            depth = raw_line.count("|") - 1

            # Normalize indentation before parsing
            line = raw_line.lstrip()

            split_match = split_re.match(line)
            class_match = class_re.match(line)

            if split_match:
                feature = split_match.group(1)

                if depth == 0 and root is None:
                    root = feature

                branches[depth] = feature

                # Remove deeper branches from the previous path
                for d in list(branches.keys()):
                    if d > depth:
                        del branches[d]

            elif class_match:
                label = int(class_match.group(1))

                path_features = [
                    branches[d]
                    for d in sorted(branches.keys())
                    if d <= depth
                ]

                if path_features:
                    paths.append({
                        "features": path_features,
                        "class": label
                    })

    if root is None or not paths:
        raise ValueError(
            f"Could not parse a root and leaves from {path}"
        )

    return root, paths


def load_trees(input_dir):
    records = []
    missing = []
    for dataset in DATASETS:
        for layer in LAYERS:
            file = input_dir / dataset / f"layer_{layer:02d}_rules_full.txt"
            if not file.exists():
                missing.append(str(file))
                continue
            root, paths = parse_tree_file(file)
            counts = Counter(feature for p in paths for feature in p)
            records.append({
                "dataset": dataset,
                "layer": layer,
                "root_feature": root,
                "n_paths_leaves": len(paths),
                "feature_mentions": counts,
            })

    if missing:
        sample = "\n".join(missing[:12])
        suffix = f"\n... and {len(missing)-12} more" if len(missing) > 12 else ""
        raise FileNotFoundError(
            "Missing expected files:\n" + sample + suffix +
            "\nCheck --input_dir and the dataset folder names (snli, arxiv, agnews)."
        )
    if len(records) != 36:
        raise RuntimeError(f"Expected 36 trees, found {len(records)}.")
    return records


def plot_roots(records, out_dir):
    df = pd.DataFrame([{
        "dataset": r["dataset"], "layer": r["layer"],
        "root_feature": r["root_feature"],
    } for r in records])
    pivot = df.pivot(index="dataset", columns="layer", values="root_feature").reindex(DATASETS)
    categories = sorted(df["root_feature"].unique())
    codes = {name: i for i, name in enumerate(categories)}
    matrix = pivot.apply(lambda col: col.map(codes)).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    im = ax.imshow(
        matrix, aspect="auto",
        cmap=plt.get_cmap("tab20", max(1, len(categories))),
        vmin=-0.5, vmax=len(categories)-0.5,
    )
    ax.set_xticks(np.arange(12), [str(x) for x in LAYERS])
    ax.set_yticks(np.arange(3), ["SNLI", "arXiv", "AG News"])
    ax.set_xlabel("BERT layer")
    ax.set_title("Root feature of each decision tree")
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, dataset in enumerate(DATASETS):
        for j, layer in enumerate(LAYERS):
            ax.text(j, i, pivot.loc[dataset, layer], ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(len(categories))), pad=0.02)
    cbar.ax.set_yticklabels(categories)
    cbar.set_label("Root feature")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_1_root_features_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_1_root_features_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_dir / "audit_root_features.csv", index=False)


def plot_families(records, out_dir):
    rows = []
    for dataset in DATASETS:
        counts = Counter()
        total = 0
        for r in records:
            if r["dataset"] == dataset:
                total += sum(r["feature_mentions"].values())
                counts.update(r["feature_mentions"])
        for family, features in FEATURE_FAMILIES.items():
            n = sum(counts[f] for f in features)
            rows.append({
                "dataset": dataset, "family": family,
                "feature_mentions": n, "total_feature_mentions": total,
                "percentage": 100 * n / total if total else 0,
            })

    df = pd.DataFrame(rows)
    family_order = list(FEATURE_FAMILIES)
    display = {"snli": "SNLI", "arxiv": "arXiv", "agnews": "AG News"}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(family_order))
    height = 0.24
    for offset, dataset in zip([-height, 0, height], DATASETS):
        subset = df[df.dataset == dataset].set_index("family").loc[family_order]
        ax.barh(y + offset, subset.percentage, height=height * 0.9, label=display[dataset])
    ax.set_yticks(y, family_order)
    ax.invert_yaxis()
    ax.set_xlabel("Share of feature mentions across root-to-leaf paths (%)")
    ax.set_title("Feature-family distribution by corpus")
    ax.legend(title="Corpus")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_2_feature_families.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_2_feature_families.pdf", bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_dir / "audit_feature_families.csv", index=False)


def plot_path_counts(records, out_dir):
    df = pd.DataFrame([{
        "dataset": r["dataset"], "layer": r["layer"],
        "n_paths_leaves": r["n_paths_leaves"],
    } for r in records])
    display = {"snli": "SNLI", "arxiv": "arXiv", "agnews": "AG News"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for dataset in DATASETS:
        subset = df[df.dataset == dataset].sort_values("layer")
        ax.plot(subset.layer, subset.n_paths_leaves, marker="o", linewidth=2,
                label=display[dataset])
    ax.set_xticks(LAYERS)
    ax.set_xlabel("BERT layer")
    ax.set_ylabel("Number of root-to-leaf paths (leaves)")
    ax.set_title("Decision-tree path count across layers")
    ax.grid(alpha=0.25)
    ax.legend(title="Corpus")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_3_paths_per_layer.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_3_paths_per_layer.pdf", bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out_dir / "audit_paths_per_layer.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path,
                        default=Path("outputs/decision_tree/pretrained"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/figures/pretrained"))
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_trees(args.input_dir)
    plot_roots(records, args.output_dir)
    plot_families(records, args.output_dir)
    plot_path_counts(records, args.output_dir)

    print("[OK] 3 figures generated in PNG (300 dpi) and PDF.")
    print("[OK] 3 audit CSV files generated.")
    print(f"Input : {args.input_dir.resolve()}")
    print(f"Output: {args.output_dir.resolve()}")
    print("Family percentages count repeated feature appearances across all paths.")
    print("Path counts include leaves of both class 0 and class 1.")


if __name__ == "__main__":
    main()
