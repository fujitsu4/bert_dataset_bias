"""
analyze_untrained patterns.py
Author : Zakaria JOUILIL

Description:
Stability analysis of untrained-BERT decision trees across random seeds.

Input structure:
<dataset>/seed_<seed>/layer_XX_rules_full.txt

Primary pattern definition:
- Each pattern is an ordered root-to-leaf sequence of split features.
- Only paths ending in TARGET_CLASS are used for pattern analyses.
- Numerical thresholds and split operators are ignored when identifying patterns.
- No complementary-leaf/path simplification is performed.

Scope conventions:
- Positive-class paths: pattern stability, path-based feature frequencies,
  subpaths, cross-dataset stable-pattern comparisons and feature-rank correlations.
- Complete trees: root-feature stability and descriptive tree complexity.
- Jaccard for two empty sets is undefined (NaN), not 1.
- Bootstrap CI: resample seed units with replacement, but do not count a pair
  formed by two bootstrap copies of the SAME original seed. Distinct original
  seed pairs are weighted by their bootstrap multiplicities.

CSV delimiter: semicolon.

Usage:
    !python -m src.dataset_analysis.analyze_untrained_patterns \
  --root outputs/decision_tree/untrained \
  --output outputs/untrained_stability_analysis_corrected \
  --expected-seeds 30 \
  --expected-layers 12 \
  --thresholds 0.50 0.80 0.90 \
  --subpath-threshold 0.50 \
  --bootstrap 2000
"""

import argparse
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT = re.compile(
    r"^\s*(?P<prefix>(?:\|\s*)*)\|---\s+(?P<feature>.+?)\s+"
    r"(?:<=|>)\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)
LEAF = re.compile(
    r"^\s*(?P<prefix>(?:\|\s*)*)\|---\s+class:\s*(?P<class>-?\d+)\s*$"
)
SEED = re.compile(r"seed_(\d+)$")
LAYER = re.compile(r"layer_(\d+)_rules_full\.txt$", re.I)


def parse_tree(file_path, dataset, seed, layer):
    """Parse one sklearn export_text tree."""
    stack = {}
    split_features = []
    split_depths = []
    leaves = []
    root_feature = None

    for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SPLIT.match(line)
        if match:
            depth = match.group("prefix").count("|")
            feature = match.group("feature").strip()
            stack[depth] = feature
            stack = {d: f for d, f in stack.items() if d <= depth}
            if depth == 0:
                root_feature = feature
            split_features.append(feature)
            split_depths.append(depth)
            continue

        match = LEAF.match(line)
        if match:
            depth = match.group("prefix").count("|")
            path = tuple(stack[d] for d in range(depth) if d in stack)
            leaves.append({
                "dataset": dataset,
                "seed": int(seed),
                "layer": int(layer),
                "class": int(match.group("class")),
                "features": path,
                "pattern": " -> ".join(path),
                "path_length": len(path),
                "source_file": str(file_path),
            })

    if root_feature is None:
        raise ValueError("No root split parsed; check export_text formatting.")
    if not leaves:
        raise ValueError("No leaves parsed.")

    return {
        "dataset": dataset,
        "seed": int(seed),
        "layer": int(layer),
        "root_feature": root_feature,
        "split_nodes": len(split_features),
        "leaf_count": len(leaves),
        "max_depth": max(split_depths, default=0),
        "mean_path_length_all": float(np.mean([x["path_length"] for x in leaves])),
        "feature_mentions_all": Counter(split_features),
        "features_used_all": set(split_features),
        "source_file": str(file_path),
    }, leaves


def jaccard(a, b):
    """Jaccard similarity; empty-vs-empty is undefined."""
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def bootstrap_seed_pair_ci(seed_sets, n_boot, rng):
    """
    Seed-level bootstrap CI for mean pairwise Jaccard.

    Bootstrap draws are made with replacement. If the same original seed is
    drawn more than once, pairs between its duplicate copies are excluded.
    For each pair of distinct original seeds, its contribution is weighted
    by the product of the two bootstrap multiplicities. This prevents
    artificial self-comparisons with Jaccard=1.
    """
    seed_ids = sorted(seed_sets)
    if len(seed_ids) < 2 or n_boot <= 0:
        return np.nan, np.nan, 0

    estimates = []
    n_valid = 0
    for _ in range(n_boot):
        draws = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        multiplicity = Counter(int(x) for x in draws)
        weighted_sum = 0.0
        total_weight = 0.0

        unique_drawn = sorted(multiplicity)
        for i, s1 in enumerate(unique_drawn):
            for s2 in unique_drawn[i + 1:]:
                score = jaccard(seed_sets[s1], seed_sets[s2])
                if np.isfinite(score):
                    weight = multiplicity[s1] * multiplicity[s2]
                    weighted_sum += weight * score
                    total_weight += weight

        if total_weight > 0:
            estimates.append(weighted_sum / total_weight)
            n_valid += 1

    if not estimates:
        return np.nan, np.nan, n_valid
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), n_valid


def save_csv(rows, output_dir, filename):
    pd.DataFrame(rows).to_csv(output_dir / filename, sep=";", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stability of untrained-BERT decision trees."
    )
    parser.add_argument("--root", required=True,
                        help="Root containing dataset/seed_x/layer_xx_rules_full.txt")
    parser.add_argument("--output", default="outputs/untrained_stability_analysis_corrected")
    parser.add_argument("--datasets", nargs="+", default=["snli", "arxiv", "agnews"])
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--expected-seeds", type=int, default=30)
    parser.add_argument("--expected-layers", type=int, default=12)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.50, 0.80, 0.90])
    parser.add_argument("--subpath-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=2026)
    args = parser.parse_args()

    datasets = [d.lower() for d in args.datasets]
    root = Path(args.root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    trees = []
    leaves = []
    errors = []
    dataset_names = set(datasets)

    for file_path in sorted(root.rglob("layer_*_rules_full.txt")):
        layer_match = LAYER.search(file_path.name)
        seed_dir = next((p for p in file_path.parents if SEED.match(p.name)), None)
        dataset = next(
            (p.name.lower() for p in file_path.parents if p.name.lower() in dataset_names),
            None
        )
        if layer_match is None or seed_dir is None or dataset is None:
            continue

        seed = int(SEED.match(seed_dir.name).group(1))
        layer = int(layer_match.group(1))
        try:
            tree, tree_leaves = parse_tree(file_path, dataset, seed, layer)
            trees.append(tree)
            leaves.extend(tree_leaves)
        except Exception as exc:
            errors.append({"file": str(file_path), "error": str(exc)})

    if not trees:
        raise SystemExit(f"No full tree files parsed under: {root}")

    tdf = pd.DataFrame(trees)
    ldf = pd.DataFrame(leaves)
    if errors:
        save_csv(errors, out, "00_parse_errors.csv")

    target = ldf[ldf["class"] == args.target_class].copy()
    if target.empty:
        raise SystemExit(
            f"No class-{args.target_class} paths found. Check --target-class and parsing."
        )

    # ------------------------------------------------------------------
    # 00. Coverage and positive-leaf audit
    # ------------------------------------------------------------------
    audit_rows = []
    for ds in datasets:
        sub = tdf[tdf.dataset == ds]
        found_seeds = sorted(sub.seed.unique())
        found_layers = sorted(sub.layer.unique())
        audit_rows.append({
            "dataset": ds,
            "tree_files": len(sub),
            "seeds": len(found_seeds),
            "layers": len(found_layers),
            "expected_tree_files": args.expected_seeds * args.expected_layers,
            "seed_ids": ",".join(map(str, found_seeds)),
            "layers_found": ",".join(map(str, found_layers)),
            "trees_with_zero_target_leaves": int((sub.target_leaf_count == 0).sum())
            if "target_leaf_count" in sub.columns else np.nan,
        })

    # Add target leaf counts after merging parsed leaf information.
    target_counts = (
        target.groupby(["dataset", "seed", "layer"]).size().rename("target_leaf_count")
        .reset_index()
    )
    tdf = tdf.merge(target_counts, on=["dataset", "seed", "layer"], how="left")
    tdf["target_leaf_count"] = tdf["target_leaf_count"].fillna(0).astype(int)
    for row in audit_rows:
        ds = row["dataset"]
        row["trees_with_zero_target_leaves"] = int(
            ((tdf.dataset == ds) & (tdf.target_leaf_count == 0)).sum()
        )
    save_csv(audit_rows, out, "00_coverage_audit.csv")

    target[["dataset", "seed", "layer", "class", "pattern",
            "path_length", "source_file"]].to_csv(
        out / "01_detailed_target_paths.csv", sep=";", index=False
    )

    # ------------------------------------------------------------------
    # 02. Global pattern stability (positive-class paths)
    # ------------------------------------------------------------------
    pattern_rows = []
    for (ds, pattern), group in target.groupby(["dataset", "pattern"]):
        seed_count = int(group.seed.nunique())
        occurrences = int(len(group))
        pattern_rows.append({
            "dataset": ds,
            "pattern": pattern,
            "pattern_length": int(group.path_length.iloc[0]),
            "occurrences": occurrences,
            "seed_count": seed_count,
            "stability_rate": seed_count / args.expected_seeds,
            "mean_occurrences_per_seed": occurrences / seed_count,
        })
    patterns = pd.DataFrame(pattern_rows)
    patterns.to_csv(out / "02_global_pattern_stability.csv", sep=";", index=False)

    # ------------------------------------------------------------------
    # 03. Stable-pattern counts, survival and seed-count distribution
    # ------------------------------------------------------------------
    stable_rows, survival_rows, distribution_rows = [], [], []
    for ds in datasets:
        subset = patterns[patterns.dataset == ds]
        for threshold in sorted(args.thresholds):
            minimum = int(np.ceil(threshold * args.expected_seeds))
            stable_rows.append({
                "dataset": ds,
                "threshold": threshold,
                "minimum_seed_count": minimum,
                "pattern_count": int((subset.seed_count >= minimum).sum()),
            })
        for minimum in range(1, args.expected_seeds + 1):
            survival_rows.append({
                "dataset": ds,
                "minimum_seed_count": minimum,
                "stability_rate": minimum / args.expected_seeds,
                "pattern_count": int((subset.seed_count >= minimum).sum()),
            })
        for count, number in subset.seed_count.value_counts().sort_index().items():
            distribution_rows.append({
                "dataset": ds,
                "seed_count": int(count),
                "stability_rate": count / args.expected_seeds,
                "pattern_count": int(number),
            })
    save_csv(stable_rows, out, "03a_stable_pattern_counts.csv")
    save_csv(survival_rows, out, "03b_pattern_survival_curve.csv")
    save_csv(distribution_rows, out, "03c_seed_count_distribution.csv")

    # ------------------------------------------------------------------
    # 04. Pattern diversity
    # ------------------------------------------------------------------
    diversity_rows = []
    for ds in datasets:
        subset = patterns[patterns.dataset == ds]
        n_patterns = len(subset)
        total_occurrences = int(subset.occurrences.sum())
        singleton_count = int((subset.seed_count == 1).sum())
        diversity_rows.append({
            "dataset": ds,
            "distinct_patterns": n_patterns,
            "total_occurrences": total_occurrences,
            "distinct_over_occurrences": (
                n_patterns / total_occurrences if total_occurrences else np.nan
            ),
            "singleton_patterns": singleton_count,
            "singleton_share": singleton_count / n_patterns if n_patterns else np.nan,
        })
    save_csv(diversity_rows, out, "04_pattern_diversity.csv")

    # ------------------------------------------------------------------
    # 05. Pairwise seed Jaccard, including empty positive-path sets.
    # ------------------------------------------------------------------
    pattern_sets = {}
    feature_sets_positive = {}
    for ds in datasets:
        ds_trees = tdf[tdf.dataset == ds]
        for row in ds_trees.itertuples():
            key = (ds, int(row.seed), int(row.layer))
            pattern_sets[key] = set()
            feature_sets_positive[key] = set()

    for (ds, seed, layer), group in target.groupby(["dataset", "seed", "layer"]):
        key = (ds, int(seed), int(layer))
        pattern_sets[key] = set(group.pattern)
        feature_sets_positive[key] = set(
            feature for path in group.features for feature in path
        )

    pair_rows, summary_rows = [], []
    for ds in datasets:
        ds_seed_ids = sorted(tdf.loc[tdf.dataset == ds, "seed"].unique())
        ds_layers = sorted(tdf.loc[tdf.dataset == ds, "layer"].unique())
        for layer in ds_layers:
            for set_type, mapping in [
                ("positive_class_patterns", pattern_sets),
                ("positive_path_features", feature_sets_positive),
            ]:
                seed_sets = {
                    int(seed): mapping[(ds, int(seed), int(layer))]
                    for seed in ds_seed_ids
                    if (ds, int(seed), int(layer)) in mapping
                }
                scores = []
                undefined_pairs = 0
                for seed_1, seed_2 in itertools.combinations(sorted(seed_sets), 2):
                    score = jaccard(seed_sets[seed_1], seed_sets[seed_2])
                    if np.isfinite(score):
                        scores.append(score)
                    else:
                        undefined_pairs += 1
                    pair_rows.append({
                        "dataset": ds,
                        "layer": layer,
                        "set_type": set_type,
                        "seed_1": seed_1,
                        "seed_2": seed_2,
                        "jaccard": score,
                    })

                ci_low, ci_high, valid_boot = bootstrap_seed_pair_ci(
                    seed_sets, args.bootstrap, rng
                )
                summary_rows.append({
                    "dataset": ds,
                    "layer": layer,
                    "set_type": set_type,
                    "n_seeds": len(seed_sets),
                    "n_pairs_total": len(seed_sets) * (len(seed_sets) - 1) // 2,
                    "n_pairs_defined": len(scores),
                    "n_pairs_undefined_empty_empty": undefined_pairs,
                    "mean_jaccard": float(np.mean(scores)) if scores else np.nan,
                    "sd_jaccard": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
                    "median_jaccard": float(np.median(scores)) if scores else np.nan,
                    "seed_bootstrap_ci95_low": ci_low,
                    "seed_bootstrap_ci95_high": ci_high,
                    "valid_bootstrap_replicates": valid_boot,
                    "bootstrap_replicates_requested": args.bootstrap,
                })
    save_csv(pair_rows, out, "05a_pairwise_seed_jaccard.csv")
    save_csv(summary_rows, out, "05b_seed_jaccard_summary.csv")

    # ------------------------------------------------------------------
    # 06. Root feature stability (complete trees)
    # ------------------------------------------------------------------
    root_rows = []
    root_summary_rows = []
    for row in tdf.itertuples():
        root_rows.append({
            "dataset": row.dataset,
            "seed": int(row.seed),
            "layer": int(row.layer),
            "root_feature": row.root_feature,
        })
    roots_df = pd.DataFrame(root_rows)
    for (ds, layer), group in roots_df.groupby(["dataset", "layer"]):
        counts = group.root_feature.value_counts()
        n = len(group)
        probabilities = counts / n
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        root_summary_rows.append({
            "dataset": ds,
            "layer": layer,
            "n_seeds": n,
            "modal_root_feature": counts.index[0],
            "modal_root_count": int(counts.iloc[0]),
            "modal_root_share": float(counts.iloc[0] / n),
            "root_entropy_bits": entropy,
            "root_distribution": " | ".join(
                f"{feature}:{count}" for feature, count in counts.items()
            ),
        })
    save_csv(root_rows, out, "06a_root_feature_by_seed.csv")
    save_csv(root_summary_rows, out, "06b_root_feature_stability.csv")

    # ------------------------------------------------------------------
    # 07. Complete-tree complexity (descriptive/contextual)
    # ------------------------------------------------------------------
    complexity_rows = []
    for row in tdf.itertuples():
        complexity_rows.append({
            "dataset": row.dataset,
            "seed": int(row.seed),
            "layer": int(row.layer),
            "split_nodes": int(row.split_nodes),
            "leaf_count": int(row.leaf_count),
            "target_leaf_count": int(row.target_leaf_count),
            "max_depth": int(row.max_depth),
            "mean_path_length_all": float(row.mean_path_length_all),
        })
    complexity_df = pd.DataFrame(complexity_rows)
    complexity_df.to_csv(out / "07a_tree_complexity_by_seed_layer.csv",
                         sep=";", index=False)
    complexity_summary = complexity_df.groupby(["dataset", "layer"]).agg(
        n_seeds=("seed", "nunique"),
        split_nodes_mean=("split_nodes", "mean"),
        split_nodes_sd=("split_nodes", "std"),
        leaves_mean=("leaf_count", "mean"),
        leaves_sd=("leaf_count", "std"),
        depth_mean=("max_depth", "mean"),
        depth_sd=("max_depth", "std"),
        path_length_mean=("mean_path_length_all", "mean"),
        path_length_sd=("mean_path_length_all", "std"),
        target_leaves_mean=("target_leaf_count", "mean"),
        target_leaves_sd=("target_leaf_count", "std"),
    ).reset_index()
    complexity_summary.to_csv(out / "07b_tree_complexity_summary.csv",
                              sep=";", index=False)

    # ------------------------------------------------------------------
    # 08. Feature frequencies recalculated ONLY on positive-class paths.
    # Each path node contributes one feature mention.
    # ------------------------------------------------------------------
    feature_universe = sorted({
        feature for path in target.features for feature in path
    })
    feature_rows = []
    # Include all observed dataset x seed x layer units, even if no target path.
    for ds in datasets:
        for row in tdf[tdf.dataset == ds].itertuples():
            key_group = target[
                (target.dataset == ds) &
                (target.seed == int(row.seed)) &
                (target.layer == int(row.layer))
            ]
            mentions = Counter(
                feature for path in key_group.features for feature in path
            )
            total_mentions = sum(mentions.values())
            for feature in feature_universe:
                count = int(mentions.get(feature, 0))
                feature_rows.append({
                    "dataset": ds,
                    "seed": int(row.seed),
                    "layer": int(row.layer),
                    "feature": feature,
                    "positive_path_mentions": count,
                    "positive_path_mention_share": (
                        count / total_mentions if total_mentions else 0.0
                    ),
                    "selected_in_positive_paths": int(count > 0),
                    "has_positive_path": int(total_mentions > 0),
                })
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(out / "08a_positive_path_feature_use_by_seed_layer.csv",
                      sep=";", index=False)

    feature_summary = feature_df.groupby(
        ["dataset", "layer", "feature"]
    ).agg(
        n_seed_units=("seed", "nunique"),
        mean_mentions=("positive_path_mentions", "mean"),
        sd_mentions=("positive_path_mentions", "std"),
        mean_mention_share=("positive_path_mention_share", "mean"),
        sd_mention_share=("positive_path_mention_share", "std"),
        seed_selection_count=("selected_in_positive_paths", "sum"),
        seed_selection_rate=("selected_in_positive_paths", "mean"),
        seeds_with_positive_paths=("has_positive_path", "sum"),
    ).reset_index()
    feature_summary.to_csv(
        out / "08b_positive_path_feature_frequency_by_layer.csv",
        sep=";", index=False
    )

    # ------------------------------------------------------------------
    # 09. Contiguous subpaths, retain only those present in >= threshold seeds.
    # Seed count is unique seeds per dataset across all layers.
    # ------------------------------------------------------------------
    subpath_stats = defaultdict(lambda: {
        "occurrences": 0, "seeds": set(), "layers": set()
    })
    for row in target.itertuples():
        features = row.features
        for length in (1, 2, 3):
            for start in range(len(features) - length + 1):
                subpath = " -> ".join(features[start:start + length])
                key = (row.dataset, subpath, length)
                subpath_stats[key]["occurrences"] += 1
                subpath_stats[key]["seeds"].add(int(row.seed))
                subpath_stats[key]["layers"].add(int(row.layer))

    minimum_subpath_seeds = int(
        np.ceil(args.subpath_threshold * args.expected_seeds)
    )
    subpath_rows = []
    for (ds, subpath, length), stats in subpath_stats.items():
        seed_count = len(stats["seeds"])
        if seed_count < minimum_subpath_seeds:
            continue
        subpath_rows.append({
            "dataset": ds,
            "subpath_length": length,
            "subpath": subpath,
            "occurrences": stats["occurrences"],
            "seed_count": seed_count,
            "stability_rate": seed_count / args.expected_seeds,
            "minimum_seed_count_threshold": minimum_subpath_seeds,
            "layers_present": ",".join(map(str, sorted(stats["layers"]))),
        })
    subpath_rows.sort(
        key=lambda x: (x["dataset"], x["subpath_length"], -x["seed_count"], x["subpath"])
    )
    save_csv(subpath_rows, out, "09_common_subpaths_ge_50pct.csv")

    # ------------------------------------------------------------------
    # 10. Cross-dataset Jaccard on stable patterns at EACH threshold.
    # ------------------------------------------------------------------
    cross_rows = []
    stable_sets_by_threshold = {}
    for threshold in sorted(args.thresholds):
        minimum = int(np.ceil(threshold * args.expected_seeds))
        stable_sets = {
            ds: set(patterns[
                (patterns.dataset == ds) &
                (patterns.seed_count >= minimum)
            ].pattern)
            for ds in datasets
        }
        stable_sets_by_threshold[threshold] = stable_sets
        for ds1, ds2 in itertools.combinations(datasets, 2):
            cross_rows.append({
                "threshold": threshold,
                "minimum_seed_count": minimum,
                "dataset_1": ds1,
                "dataset_2": ds2,
                "stable_pattern_count_dataset_1": len(stable_sets[ds1]),
                "stable_pattern_count_dataset_2": len(stable_sets[ds2]),
                "stable_pattern_jaccard": jaccard(stable_sets[ds1], stable_sets[ds2]),
                "shared_stable_pattern_count": len(stable_sets[ds1] & stable_sets[ds2]),
            })
    save_csv(cross_rows, out, "10_cross_dataset_stable_pattern_jaccard.csv")

    # ------------------------------------------------------------------
    # 11. Multi-threshold shared-pattern table.
    # For every pattern present in all datasets, report per-dataset counts,
    # minimum seed count and whether it qualifies at each threshold.
    # ------------------------------------------------------------------
    pattern_sets_by_dataset = {
        ds: set(patterns.loc[patterns.dataset == ds, "pattern"])
        for ds in datasets
    }
    patterns_present_all = (
        set.intersection(*(pattern_sets_by_dataset[ds] for ds in datasets))
        if datasets else set()
    )
    shared_rows = []
    for pattern in sorted(patterns_present_all):
        row = {"pattern": pattern}
        counts = []
        for ds in datasets:
            match = patterns[
                (patterns.dataset == ds) & (patterns.pattern == pattern)
            ].iloc[0]
            count = int(match.seed_count)
            counts.append(count)
            row[f"{ds}_seed_count"] = count
            row[f"{ds}_stability_rate"] = float(match.stability_rate)
        row["minimum_seed_count_across_datasets"] = min(counts)
        row["minimum_stability_rate_across_datasets"] = min(counts) / args.expected_seeds
        for threshold in sorted(args.thresholds):
            minimum = int(np.ceil(threshold * args.expected_seeds))
            row[f"common_at_{int(threshold * 100)}pct"] = int(all(c >= minimum for c in counts))
        shared_rows.append(row)
    save_csv(shared_rows, out, "11_patterns_present_in_all_datasets_multi_threshold.csv")

    # ------------------------------------------------------------------
    # 12. Cross-dataset feature-frequency rank correlations.
    # Feature selection rates are based on positive-class paths.
    # Rate denominator: all expected seed-layer units, including units with
    # no positive paths (these contribute zero selected features).
    # ------------------------------------------------------------------
    all_features = sorted(feature_df.feature.unique())
    selection_rates = {}
    for ds in datasets:
        selection_rates[ds] = pd.Series({
            feature: float(
                feature_df.loc[
                    (feature_df.dataset == ds) &
                    (feature_df.feature == feature),
                    "selected_in_positive_paths"
                ].mean()
            )
            for feature in all_features
        })
    correlation_rows = []
    for ds1, ds2 in itertools.combinations(datasets, 2):
        correlation_rows.append({
            "dataset_1": ds1,
            "dataset_2": ds2,
            "spearman": selection_rates[ds1].corr(selection_rates[ds2], method="spearman"),
            "kendall": selection_rates[ds1].corr(selection_rates[ds2], method="kendall"),
            "feature_count": len(all_features),
            "feature_scope": "features occurring in positive-class paths",
        })
    save_csv(correlation_rows, out, "12_cross_dataset_positive_feature_rank_correlations.csv")

    # ------------------------------------------------------------------
    # 13. Manifest documenting analytical scopes and conventions.
    # ------------------------------------------------------------------
    manifest = [
        {"item": "trees_parsed", "value": len(tdf)},
        {"item": "target_class_paths", "value": len(target)},
        {"item": "target_class", "value": args.target_class},
        {"item": "pattern_definition", "value": "ordered root-to-leaf feature sequence; thresholds/operators ignored"},
        {"item": "pattern_scope", "value": "paths ending in target class; no complementary-path simplification"},
        {"item": "feature_frequency_scope", "value": "features counted along target-class paths only"},
        {"item": "root_feature_scope", "value": "complete trees"},
        {"item": "tree_complexity_scope", "value": "complete trees; descriptive/contextual"},
        {"item": "jaccard_empty_empty", "value": "undefined (NaN); excluded from mean and bootstrap"},
        {"item": "bootstrap_method", "value": "seed-level resampling with replacement; pairs of duplicate copies of the same original seed excluded; distinct seed-pair contributions weighted by multiplicity"},
        {"item": "bootstrap_replicates", "value": args.bootstrap},
        {"item": "subpath_minimum_stability", "value": args.subpath_threshold},
        {"item": "cross_dataset_pattern_comparison", "value": "patterns stable at each configured threshold"},
        {"item": "thresholds", "value": ",".join(map(str, sorted(args.thresholds)))},
        {"item": "expected_seed_count", "value": args.expected_seeds},
        {"item": "expected_layers", "value": args.expected_layers},
    ]
    save_csv(manifest, out, "99_analysis_manifest.csv")

    print(f"[DONE] {len(tdf)} trees parsed; {len(target)} target-class paths.")
    print(f"[DONE] Output directory: {out.resolve()}")
    print(f"[INFO] Trees with zero target-class leaves: {int((tdf.target_leaf_count == 0).sum())}")
    if len(tdf) != len(datasets) * args.expected_seeds * args.expected_layers:
        print("[WARNING] Parsed tree count differs from expected coverage; inspect 00_coverage_audit.csv.")
    if errors:
        print(f"[WARNING] {len(errors)} parse errors; inspect 00_parse_errors.csv.")


if __name__ == "__main__":
    main()