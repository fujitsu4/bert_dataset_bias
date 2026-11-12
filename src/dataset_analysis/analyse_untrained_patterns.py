#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stability analysis of untrained-BERT decision trees.

Reads ONLY layer_XX_rules_full.txt files from:
<dataset>/seed_<seed>/layer_XX_rules_full.txt

Patterns are ordered root-to-leaf feature sequences. Numerical thresholds and
operators are intentionally ignored. By default, pattern metrics use class-1
leaves; tree complexity and feature-use metrics use the complete tree.

!python -m src.dataset_analysis.analyze_untrained_patterns \
  --root outputs/decision_tree/untrained \
  --output outputs/untrained_stability_analysis \
  --expected-seeds 30 \
  --expected-layers 12

"""
import argparse
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT = re.compile(r"^\s*(?P<prefix>(?:\|\s*)*)\|---\s+(?P<feature>.+?)\s+(?:<=|>)\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$")
LEAF = re.compile(r"^\s*(?P<prefix>(?:\|\s*)*)\|---\s+class:\s*(?P<class>-?\d+)\s*$")
SEED = re.compile(r"seed_(\d+)$")
LAYER = re.compile(r"layer_(\d+)_rules_full\.txt$", re.I)


def parse_tree(file, dataset, seed, layer, target_class):
    stack, split_features, split_depths, leaves = {}, [], [], []
    root = None
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SPLIT.match(line)
        if m:
            depth = m.group("prefix").count("|")
            feature = m.group("feature").strip()
            stack[depth] = feature
            stack = {d: f for d, f in stack.items() if d <= depth}
            if depth == 0:
                root = feature
            split_features.append(feature)
            split_depths.append(depth)
            continue
        m = LEAF.match(line)
        if m:
            depth = m.group("prefix").count("|")
            path = tuple(stack[d] for d in range(depth) if d in stack)
            leaves.append({
                "dataset": dataset, "seed": seed, "layer": layer,
                "class": int(m.group("class")), "features": path,
                "pattern": " -> ".join(path), "path_length": len(path),
                "source_file": str(file),
            })
    if root is None:
        raise ValueError("No root split parsed")
    return {
        "dataset": dataset, "seed": seed, "layer": layer,
        "root_feature": root, "split_nodes": len(split_features),
        "leaf_count": len(leaves),
        "target_leaf_count": sum(x["class"] == target_class for x in leaves),
        "max_depth": max(split_depths, default=0),
        "mean_path_length_all": float(np.mean([x["path_length"] for x in leaves])) if leaves else 0,
        "feature_mentions": Counter(split_features),
        "features_used": set(split_features),
        "source_file": str(file),
    }, leaves


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def bootstrap_ci(seed_sets, n_boot, rng):
    seeds = list(seed_sets)
    if len(seeds) < 2:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = rng.choice(seeds, len(seeds), replace=True)
        vals = [jac(seed_sets[sample[i]], seed_sets[sample[j]])
                for i in range(len(sample)) for j in range(i + 1, len(sample))]
        if vals:
            means.append(np.mean(vals))
    return tuple(np.percentile(means, [2.5, 97.5])) if means else (np.nan, np.nan)


def write(rows, out, name):
    pd.DataFrame(rows).to_csv(out / name, sep=";", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root containing <dataset>/seed_xxx/layer_xx_rules_full.txt")
    ap.add_argument("--output", default="outputs/untrained_stability_analysis")
    ap.add_argument("--datasets", nargs="+", default=["snli", "arxiv", "agnews"])
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--expected-seeds", type=int, default=30)
    ap.add_argument("--expected-layers", type=int, default=12)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[.50, .80, .90])
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    trees, leaves, errors = [], [], []

    for file in sorted(root.rglob("layer_*_rules_full.txt")):
        lm = LAYER.search(file.name)
        sd = next((p for p in file.parents if SEED.match(p.name)), None)
        if not lm or sd is None:
            continue
        dataset = next((p.name.lower() for p in file.parents
                        if p.name.lower() in [d.lower() for d in args.datasets]), None)
        if dataset is None:
            continue
        seed, layer = int(SEED.match(sd.name).group(1)), int(lm.group(1))
        try:
            tree, leaf_list = parse_tree(file, dataset, seed, layer, args.target_class)
            trees.append(tree)
            leaves.extend(leaf_list)
        except Exception as e:
            errors.append({"file": str(file), "error": str(e)})

    if not trees:
        raise SystemExit(f"No full tree files parsed under {root}. Check --root and folder structure.")
    tdf, ldf = pd.DataFrame(trees), pd.DataFrame(leaves)
    if errors:
        write(errors, out, "00_parse_errors.csv")
    target = ldf[ldf["class"] == args.target_class].copy()
    if target.empty:
        raise SystemExit(f"No class-{args.target_class} leaf paths found.")

    # Coverage audit
    audit = []
    for ds in args.datasets:
        sub = tdf[tdf.dataset == ds]
        audit.append({
            "dataset": ds, "tree_files": len(sub),
            "seeds": sub.seed.nunique(), "layers": sub.layer.nunique(),
            "expected_tree_files": args.expected_seeds * args.expected_layers,
            "seed_ids": ",".join(map(str, sorted(sub.seed.unique()))),
            "layers_found": ",".join(map(str, sorted(sub.layer.unique()))),
        })
    write(audit, out, "00_coverage_audit.csv")

    # Preserve the individual paths: dataset x seed x layer.
    target[["dataset", "seed", "layer", "class", "pattern",
            "path_length", "source_file"]].to_csv(
        out / "01_detailed_target_paths.csv", sep=";", index=False)

    # A1, A4-A6: global pattern stability, distinct seeds and occurrences.
    prows = []
    for (ds, pat), g in target.groupby(["dataset", "pattern"]):
        nseed, occ = g.seed.nunique(), len(g)
        prows.append({
            "dataset": ds, "pattern": pat, "occurrences": occ,
            "seed_count": nseed, "stability_rate": nseed / args.expected_seeds,
            "mean_occurrences_per_seed": occ / nseed,
            "pattern_length": int(g.path_length.iloc[0]),
        })
    patterns = pd.DataFrame(prows)
    patterns.to_csv(out / "02_global_pattern_stability.csv", sep=";", index=False)

    # A2-A3-A5: stable counts, survival curve, seed_count distribution.
    stable, survival, distribution = [], [], []
    for ds in args.datasets:
        g = patterns[patterns.dataset == ds]
        for rate in args.thresholds:
            k = int(np.ceil(rate * args.expected_seeds))
            stable.append({"dataset": ds, "threshold": rate, "minimum_seed_count": k,
                           "pattern_count": int((g.seed_count >= k).sum())})
        for k in range(1, args.expected_seeds + 1):
            survival.append({"dataset": ds, "minimum_seed_count": k,
                             "stability_rate": k / args.expected_seeds,
                             "pattern_count": int((g.seed_count >= k).sum())})
        for k, n in g.seed_count.value_counts().sort_index().items():
            distribution.append({"dataset": ds, "seed_count": int(k),
                                 "stability_rate": k / args.expected_seeds,
                                 "pattern_count": int(n)})
    write(stable, out, "03a_stable_pattern_counts.csv")
    write(survival, out, "03b_pattern_survival_curve.csv")
    write(distribution, out, "03c_seed_count_distribution.csv")

    # A6: diversity and singleton patterns.
    diversity = []
    for ds in args.datasets:
        g = patterns[patterns.dataset == ds]
        n = len(g)
        diversity.append({
            "dataset": ds, "distinct_patterns": n,
            "total_occurrences": int(g.occurrences.sum()),
            "distinct_over_occurrences": n / g.occurrences.sum() if g.occurrences.sum() else np.nan,
            "singleton_patterns": int((g.seed_count == 1).sum()),
            "singleton_share": float((g.seed_count == 1).sum() / n) if n else np.nan,
        })
    write(diversity, out, "04_pattern_diversity.csv")

    # Seed x layer sets used for pairwise Jaccard.
    psets, fsets = {}, {}
    for (ds, seed, layer), g in target.groupby(["dataset", "seed", "layer"]):
        psets[(ds, int(seed), int(layer))] = set(g.pattern)
    for r in trees:
        fsets[(r["dataset"], int(r["seed"]), int(r["layer"]))] = r["features_used"]

    # B7-B8: all 435 pairwise comparisons per dataset/layer, where 30 seeds exist.
    pair_rows, pair_summary = [], []
    for ds in args.datasets:
        for layer in sorted(tdf[tdf.dataset == ds].layer.unique()):
            for label, mapping in [("patterns", psets), ("features", fsets)]:
                seed_sets = {s: mapping[(ds, int(s), int(layer))]
                             for d, s, ly in mapping if d == ds and ly == layer}
                scores = []
                for s1, s2 in itertools.combinations(sorted(seed_sets), 2):
                    score = jac(seed_sets[s1], seed_sets[s2])
                    scores.append(score)
                    pair_rows.append({"dataset": ds, "layer": layer, "set_type": label,
                                      "seed_1": s1, "seed_2": s2, "jaccard": score})
                lo, hi = bootstrap_ci(seed_sets, args.bootstrap, rng)
                pair_summary.append({
                    "dataset": ds, "layer": layer, "set_type": label,
                    "n_seeds": len(seed_sets), "n_pairs": len(scores),
                    "mean_jaccard": float(np.mean(scores)) if scores else np.nan,
                    "sd_jaccard": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
                    "median_jaccard": float(np.median(scores)) if scores else np.nan,
                    "seed_bootstrap_ci95_low": lo, "seed_bootstrap_ci95_high": hi,
                })
    write(pair_rows, out, "05a_pairwise_seed_jaccard.csv")
    write(pair_summary, out, "05b_seed_jaccard_summary.csv")

    # B9, C10, D14: root feature and its stability by layer.
    root_rows = [{"dataset": r["dataset"], "seed": r["seed"],
                  "layer": r["layer"], "root_feature": r["root_feature"]} for r in trees]
    roots = pd.DataFrame(root_rows)
    root_summary = []
    for (ds, layer), g in roots.groupby(["dataset", "layer"]):
        vc = g.root_feature.value_counts()
        n = len(g)
        entropy = -sum((v/n) * np.log2(v/n) for v in vc)
        root_summary.append({
            "dataset": ds, "layer": layer, "n_seeds": n,
            "modal_root_feature": vc.index[0], "modal_root_count": int(vc.iloc[0]),
            "modal_root_share": float(vc.iloc[0] / n),
            "root_entropy_bits": float(entropy),
            "root_distribution": " | ".join(f"{k}:{v}" for k, v in vc.items()),
        })
    write(root_rows, out, "06a_root_feature_by_seed.csv")
    write(root_summary, out, "06b_root_feature_stability.csv")

    # C11-C12: complexity of the complete tree, per seed/layer and summary.
    complexity = [{
        "dataset": r["dataset"], "seed": r["seed"], "layer": r["layer"],
        "split_nodes": r["split_nodes"], "leaf_count": r["leaf_count"],
        "target_leaf_count": r["target_leaf_count"], "max_depth": r["max_depth"],
        "mean_path_length_all": r["mean_path_length_all"],
    } for r in trees]
    cdf = pd.DataFrame(complexity)
    cdf.to_csv(out / "07a_tree_complexity_by_seed_layer.csv", sep=";", index=False)
    csum = cdf.groupby(["dataset", "layer"]).agg(
        n_seeds=("seed", "nunique"),
        split_nodes_mean=("split_nodes", "mean"), split_nodes_sd=("split_nodes", "std"),
        leaves_mean=("leaf_count", "mean"), leaves_sd=("leaf_count", "std"),
        depth_mean=("max_depth", "mean"), depth_sd=("max_depth", "std"),
        path_length_mean=("mean_path_length_all", "mean"),
        path_length_sd=("mean_path_length_all", "std"),
    ).reset_index()
    csum.to_csv(out / "07b_tree_complexity_summary.csv", sep=";", index=False)

    # D13-D14: feature mentions and selection frequency across seeds.
    universe = sorted(set().union(*(r["features_used"] for r in trees)))
    feature_rows = []
    for r in trees:
        total = sum(r["feature_mentions"].values())
        for feat in universe:
            count = r["feature_mentions"].get(feat, 0)
            feature_rows.append({
                "dataset": r["dataset"], "seed": r["seed"], "layer": r["layer"],
                "feature": feat, "mentions": count,
                "mention_share": count / total if total else 0,
                "selected": int(count > 0),
            })
    fdf = pd.DataFrame(feature_rows)
    fdf.to_csv(out / "08a_feature_use_by_seed_layer.csv", sep=";", index=False)
    fsum = fdf.groupby(["dataset", "layer", "feature"]).agg(
        mean_mentions=("mentions", "mean"), sd_mentions=("mentions", "std"),
        mean_mention_share=("mention_share", "mean"),
        sd_mention_share=("mention_share", "std"),
        seed_selection_count=("selected", "sum"),
        seed_selection_rate=("selected", "mean"),
    ).reset_index()
    fsum.to_csv(out / "08b_feature_frequency_by_layer.csv", sep=";", index=False)

    # D15: contiguous subpaths of length 1, 2, 3.
    subpaths = defaultdict(lambda: {"occurrences": 0, "seeds": set(), "layers": set()})
    for r in target.itertuples():
        feats = r.features
        for length in (1, 2, 3):
            for i in range(len(feats) - length + 1):
                key = (r.dataset, " -> ".join(feats[i:i+length]), length)
                subpaths[key]["occurrences"] += 1
                subpaths[key]["seeds"].add(r.seed)
                subpaths[key]["layers"].add(r.layer)
    subrows = []
    for (ds, path, length), v in subpaths.items():
        subrows.append({"dataset": ds, "subpath_length": length, "subpath": path,
                        "occurrences": v["occurrences"], "seed_count": len(v["seeds"]),
                        "stability_rate": len(v["seeds"]) / args.expected_seeds,
                        "layers_present": ",".join(map(str, sorted(v["layers"])))})
    write(subrows, out, "09_common_subpaths.csv")

    # E16-E17: dataset comparisons and stable patterns shared by all datasets.
    psets_ds, fsets_ds = {}, {}
    for ds in args.datasets:
        psets_ds[ds] = set(patterns[patterns.dataset == ds].pattern)
        fsets_ds[ds] = set().union(*(v for (d, s, l), v in fsets.items() if d == ds))
    cross = []
    for d1, d2 in itertools.combinations(args.datasets, 2):
        cross.append({"dataset_1": d1, "dataset_2": d2,
                      "pattern_jaccard": jac(psets_ds[d1], psets_ds[d2]),
                      "feature_jaccard": jac(fsets_ds[d1], fsets_ds[d2])})
    write(cross, out, "10_cross_dataset_jaccard.csv")

    min_stable = int(np.ceil(max(args.thresholds) * args.expected_seeds))
    stable_sets = {
        ds: set(patterns[(patterns.dataset == ds) &
                         (patterns.seed_count >= min_stable)].pattern)
        for ds in args.datasets
    }
    common = set.intersection(*(stable_sets[d] for d in args.datasets))
    common_rows = []
    for pat in sorted(common):
        row = {"pattern": pat}
        for ds in args.datasets:
            x = patterns[(patterns.dataset == ds) & (patterns.pattern == pat)].iloc[0]
            row[f"{ds}_seed_count"] = int(x.seed_count)
            row[f"{ds}_stability_rate"] = float(x.stability_rate)
        common_rows.append(row)
    write(common_rows, out, "11_stable_patterns_common_all_datasets.csv")

    # E18: rank correlation of feature-selection rates across datasets.
    all_feats = sorted(set(fdf.feature))
    rates = {}
    for ds in args.datasets:
        rates[ds] = pd.Series({
            feat: float(fdf[(fdf.dataset == ds) & (fdf.feature == feat)].selected.mean())
            for feat in all_feats
        })
    correlations = []
    for d1, d2 in itertools.combinations(args.datasets, 2):
        correlations.append({"dataset_1": d1, "dataset_2": d2,
                             "spearman": rates[d1].corr(rates[d2], method="spearman"),
                             "kendall": rates[d1].corr(rates[d2], method="kendall")})
    write(correlations, out, "12_cross_dataset_feature_rank_correlations.csv")

    # Additional metric: normalized Shannon entropy of split-feature selection.
    entropies = []
    for (ds, layer), g in fdf.groupby(["dataset", "layer"]):
        counts = g.groupby("feature").mentions.sum()
        positive = counts[counts > 0]
        total = positive.sum()
        probs = positive / total if total else positive
        h = float(-(probs * np.log2(probs)).sum()) if len(probs) else np.nan
        hmax = np.log2(len(positive)) if len(positive) > 1 else np.nan
        entropies.append({"dataset": ds, "layer": layer,
                          "feature_selection_entropy_bits": h,
                          "normalized_entropy": h / hmax if hmax and hmax > 0 else np.nan,
                          "features_used": len(positive)})
    write(entropies, out, "13_feature_selection_entropy.csv")

    manifest = [
        {"item": "trees_parsed", "value": len(trees)},
        {"item": "target_class_paths", "value": len(target)},
        {"item": "target_class", "value": args.target_class},
        {"item": "pattern_definition", "value": "ordered feature sequence; thresholds/operators ignored"},
        {"item": "pattern_scope", "value": "root-to-leaf paths ending in target class"},
        {"item": "complexity_and_feature_scope", "value": "complete tree"},
        {"item": "stable_common_threshold", "value": max(args.thresholds)},
        {"item": "expected_tree_files", "value": len(args.datasets)*args.expected_seeds*args.expected_layers},
    ]
    write(manifest, out, "99_analysis_manifest.csv")
    print(f"[DONE] {len(trees)} trees parsed; {len(target)} target-class paths.")
    print(f"[DONE] Results: {out.resolve()}")
    expected = len(args.datasets) * args.expected_seeds * args.expected_layers
    if len(trees) != expected:
        print(f"[WARNING] Expected {expected} trees, found {len(trees)}. Check 00_coverage_audit.csv.")
    if errors:
        print(f"[WARNING] {len(errors)} parse errors. Check 00_parse_errors.csv.")


if __name__ == "__main__":
    main()
