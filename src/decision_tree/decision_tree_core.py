"""
decision_tree_core.py
Author: Zakaria JOUILIL
Purpose:
    Core utilities to train a Decision Tree on the "top5" label of a single layer,
    and extract human-readable rules + per-leaf statistics (support, positive count,
    positive rate).

Design:
    - Minimal, well-tested functions that return pure Python objects (dicts / strings)
    - Lightweight sanity checks inside the core (existence/types of columns, label variety)
    - Optional saving of rules text (save_rules_to)
    - Minimal CLI for debugging: --input_csv, --layer, --tree_seed

Public functions:
    - train_decision_tree(X, y, max_depth=4, min_samples_leaf=5, random_state=42)
    - extract_rules_text(clf, feature_names)
    - compute_leaf_stats(clf, X, y)
    - format_rules_with_stats(clf, feature_names, X, y)
    - load_features_and_label_from_df(df, label_col, feature_cols=None)
    - train_and_extract_rules_from_df(df, layer, ...)

Input and parameters :
    --input_csv: the csv containg : bert tokens, features, attention scores and the
                    label top 5 (per sentence and per layer)
    --layer: layer number
    --tree_seed: Seed for Decision Tree (determinism)
    --save_full : Optional path to save the full rules text (if omitted, nothing is written)
    --save_pos : Optional path to save only the rules leading to top5 = 1 (if omitted, nothing is written)
    --save_simp_pos : Optional path to save simplified rules leading to top5 = 1 using absorption of redundant splits (if omitted, nothing is written)
                        
Outputs:
    rules_full.txt : a text file containing the complte output decision tree
    rules_pos.txt : a text file containing the output decision tree (with only rules leading to top5 = 1)
Usage:
    python -m src.decision_tree.decision_tree_core \
        --input_csv /content/drive/MyDrive/results/attention_score/attention_top5_pretrained.csv \
        --tree_seed 42 \
        --layer 1 \
        --save_full /content/bert_dataset_bias/outputs/decision_tree/rules_full_pretrained_layer_1.txt \
        --save_pos /content/bert_dataset_bias/outputs/decision_tree/rules_pos_pretrained_layer_1.txt \
        --save_simp_pos /content/bert_dataset_bias/outputs/decision_tree/rules_simp_pos_pretrained_layer_1.txt
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import textwrap
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from datetime import datetime, UTC

# ---------------------------
# Default feature column order (21 numeric features)
# ---------------------------
DEFAULT_FEATURE_COLS = [
    "is_numeric",
    "is_punctuation",
    "is_stopword",
    "is_pronoun",
    "pos_id",
    "dep_id",
    "ent_id",
    "depth_in_tree",
    "num_dependents",
    "is_special_token",
    "token_length",
    "is_subword",
    "position_in_sentence",
    "relative_position",
    "num_subtokens",
    "token_relative_frequency",
    "word_sentence_frequency",
    "token_rank",
    "word_burstiness",
    "prev_pos_id",
    "next_pos_id",
]


# ---------------------------
# Train a single Decision Tree
# ---------------------------
def train_decision_tree(
    X,
    y,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Train a DecisionTreeClassifier with fixed hyperparameters.

    Args:
        X: pandas DataFrame or numpy array (n_samples, n_features)
        y: array-like labels (binary 0/1)
        max_depth: tree depth (int)
        min_samples_leaf: minimal samples per leaf (int)
        random_state: RNG seed for determinism

    Returns:
        Trained sklearn DecisionTreeClassifier
    """
    # Convert to numpy
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = np.asarray(X)

    y_np = np.asarray(y)

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X and y must have same number of samples")

    if X_np.shape[0] == 0:
        raise ValueError("Empty training set")

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_np, y_np)
    return clf


# ---------------------------
# Extract raw text rules
# ---------------------------
def extract_rules_text(clf: DecisionTreeClassifier, feature_names: List[str]) -> str:
    """
    Return the sklearn export_text string representing the tree rules.

    Args:
        clf: trained DecisionTreeClassifier
        feature_names: list of feature names (length == n_features)

    Returns:
        Multiline string with rules.
    """
    rules = export_text(clf, feature_names=feature_names, decimals=6)
    return rules


# ---------------------------
# Compute leaf statistics
# ---------------------------
def compute_leaf_stats(clf: DecisionTreeClassifier, X, y) -> Dict[int, Dict[str, Any]]:
    """
    Compute support and positive counts per leaf.

    Args:
        clf: trained DecisionTreeClassifier
        X: input features (DataFrame or numpy)
        y: true binary labels (array-like)

    Returns:
        dict leaf_id -> {"n_samples":int, "n_pos":int, "pos_rate":float}
    """
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = np.asarray(X)
    y_np = np.asarray(y)

    # node indices for each sample (leaf node id)
    leaf_ids = clf.apply(X_np)  # shape (n_samples,)
    unique_leaf_ids = np.unique(leaf_ids)

    stats: Dict[int, Dict[str, Any]] = {}
    for leaf in unique_leaf_ids:
        mask = leaf_ids == leaf
        n_samples = int(mask.sum())
        n_pos = int(y_np[mask].sum())
        pos_rate = float(n_pos / n_samples) if n_samples > 0 else 0.0
        stats[int(leaf)] = {
            "n_samples": n_samples,
            "n_pos": n_pos,
            "pos_rate": pos_rate,
        }
    return stats


# ---------------------------
# Combined formatter: rules + stats
# ---------------------------
def format_rules_with_stats(
    clf: DecisionTreeClassifier, feature_names: List[str], X, y, top_k_lines: Optional[int] = None
) -> str:
    """
    Build a human-readable string that contains:
      - the tree rules (export_text)
      - a table of leaf stats sorted by pos_rate desc

    Args:
        clf: trained DecisionTreeClassifier
        feature_names: list of feature names
        X, y: training data used to compute leaf stats
        top_k_lines: if given, truncate the rules text to first N lines (useful for debug)

    Returns:
        Multiline string.
    """
    rules = extract_rules_text(clf, feature_names)
    if top_k_lines is not None:
        lines = rules.splitlines()
        rules_display = "\n".join(lines[:top_k_lines]) + "\n...[truncated]\n"
    else:
        rules_display = rules

    stats = compute_leaf_stats(clf, X, y)
    rows = []
    for leaf_id, s in stats.items():
        rows.append((leaf_id, s["n_samples"], s["n_pos"], s["pos_rate"]))

    rows_sorted = sorted(rows, key=lambda t: (-t[3], -t[1]))

    stats_lines = ["LeafID | n_samples | n_pos | pos_rate"]
    for leaf_id, n_s, n_pos, pos_rate in rows_sorted:
        stats_lines.append(f"{leaf_id:6d} | {n_s:9d} | {n_pos:5d} | {pos_rate:.4f}")

    stats_display = "\n".join(stats_lines)

    out = textwrap.dedent(
        f"""
        === Decision Tree Rules (sklearn export_text) ===

        {rules_display}

        === Per-leaf statistics (sorted by pos_rate desc) ===

        {stats_display}

        """
    ).strip() + "\n"
    return out


# ---------------------------
# Utility: load features + label from dataframe
# ---------------------------
def load_features_and_label_from_df(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    require_all_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    From a wide dataframe (one row per token) extract X (DataFrame) and y (Series)
    for the decision tree.

    Args:
        df: pandas DataFrame containing columns
        label_col: name of label column (e.g. "top5_l1")
        feature_cols: list of feature column names to use; if None uses DEFAULT_FEATURE_COLS
        require_all_features: if True, raise error when any feature missing

    Returns:
        X (DataFrame), y (Series)
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS.copy()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        if require_all_features:
            raise ValueError(f"Missing required feature columns: {missing}")
        else:
            feature_cols = [c for c in feature_cols if c in df.columns]

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    # Keep only numeric columns — attempt safe casting
    X = df[feature_cols].astype(float)        
    y = df[label_col].astype(int)
    unique_vals = np.unique(y)
    if not set(unique_vals).issubset({0, 1}):
        # transform: anything !=0 -> 1
        y = (y != 0).astype(int)

    return X, y

def filter_rules_for_class1(rules_text: str) -> str:
    """
    Reconstruct full decision paths ending in 'class: 1'.
    Uses indentation depth to track the current branch.
    """

    lines = rules_text.splitlines()
    kept_paths = []

    # Stack indexed by depth: stack[depth] = content of that node
    stack = {}

    for line in lines:
        stripped = line.strip()

        # ignore non-rule lines
        if not stripped.startswith("|"):
            continue

        # depth = number of "|   " blocks
        depth = stripped.count("|")

        # update stack at this depth
        stack[depth] = line

        # prune deeper levels no longer valid
        depths_to_delete = [d for d in stack.keys() if d > depth]
        for d in depths_to_delete:
            del stack[d]

        # CASE: leaf with class = 1
        if "class: 1" in stripped:
            # full path = all levels in sorted order
            path = [stack[d] for d in sorted(stack.keys())]

            # remove any line containing class: 0
            path = [p for p in path if "class: 0" not in p]

            kept_paths.append("\n".join(path))

    return "\n\n".join(kept_paths)

def simplify_rules(rules_pos_text: str) -> str:
    """
    Simplify positive-class rules by iteratively merging complementary final splits.
    The process is repeated until no more merges are possible.
    """

    def parse_condition(line: str):
        try:
            part = line.split("---", 1)[1].strip()
            tokens = part.split()
            if len(tokens) >= 3:
                feature = tokens[0]
                op = tokens[1]
                try:
                    thresh = float(tokens[2])
                except:
                    thresh = None
                return feature, op, thresh
        except:
            return None, None, None
        return None, None, None

    def merge_once(blocks):
        """Perform a single merge pass. Returns (new_blocks, did_merge)."""
        parsed_blocks = [b.split("\n") for b in blocks]
        n = len(parsed_blocks)
        used = set()
        merged_blocks = []
        did_merge = False

        for i in range(n):
            if i in used:
                continue

            block_i = parsed_blocks[i]
            prefix_i = block_i[:-1]
            leaf_i = block_i[-1]

            merged_block = block_i  # default

            for j in range(i + 1, n):
                if j in used:
                    continue

                block_j = parsed_blocks[j]
                prefix_j = block_j[:-1]
                leaf_j = block_j[-1]

                if len(prefix_i) >= 1 and len(prefix_i) == len(prefix_j) \
                   and prefix_i[:-1] == prefix_j[:-1]:

                    f1, op1, t1 = parse_condition(prefix_i[-1])
                    f2, op2, t2 = parse_condition(prefix_j[-1])
                    if f1 and f2 and f1 == f2 and t1 is not None and t2 is not None and abs(t1 - t2) < 1e-9:
                        left_ops = {"<", "<="}
                        right_ops = {">", ">="}
                        if (op1 in left_ops and op2 in right_ops) or (op2 in left_ops and op1 in right_ops):
                            # FUSION !!!
                            merged_block = prefix_i[:-1] + [leaf_i]
                            used.add(j)
                            did_merge = True

            merged_blocks.append("\n".join(merged_block))

        return merged_blocks, did_merge

    # Initial split
    blocks = [b for b in rules_pos_text.strip().split("\n\n") if b.strip()]

    # Repeat merging until stable
    while True:
        blocks, merged = merge_once(blocks)
        if not merged:
            break

    return "\n\n".join(blocks)

# ---------------------------
# Top-level function used by compute_dt.py
# ---------------------------
def train_and_extract_rules_from_df(
    df: pd.DataFrame,
    layer: int,
    feature_cols: Optional[List[str]] = None,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    dt_seed: int = 42,
    save_rules_full_to: Optional[str] = None,
    save_rules_pos_only_to: Optional[str] = None,
    save_rules_pos_simplified_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a DT on label 'top5_l{layer}' using the features and return information.

    Args:
        df: full dataframe (all tokens)
        layer: int layer number (1..12)
        feature_cols: feature names (if None uses DEFAULT_FEATURE_COLS)
        max_depth, min_samples_leaf, dt_seed: hyperparams
        save_rules_to: optional filepath to write the rules text

    Returns:
        dict containing:
            - "clf": trained classifier
            - "rules_text": string (rules + stats)
            - "leaf_stats": dict leaf->stats
            - "n_samples": int
            - "n_pos": int
    """
    if not isinstance(layer, int):
        raise ValueError("layer argument must be an integer (1..12)")

    label_col = f"top5_l{layer}"
    X, y = load_features_and_label_from_df(df, label_col, feature_cols=feature_cols)

    # Sanity: need at least one positive and one negative sample
    unique_vals = np.unique(y)
    if unique_vals.size == 1:
        raise RuntimeError(f"Label {label_col} is constant (only values {unique_vals}). Cannot train DT.")

    clf = train_decision_tree(X, y, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=dt_seed)
    rules_full = format_rules_with_stats(clf, list(X.columns), X, y)
    rules_pos_only = filter_rules_for_class1(extract_rules_text(clf, list(X.columns)))
    rules_pos_simplified = simplify_rules(rules_pos_only)
    leaf_stats = compute_leaf_stats(clf, X, y)

    result = {
        "clf": clf,
        "rules_full": rules_full,
        "rules_pos_only": rules_pos_only,
        "rules_pos_simplified": rules_pos_simplified,
        "leaf_stats": leaf_stats,
        "n_samples": int(X.shape[0]),
        "n_pos": int(y.sum()),
    }

    # If requested, write full rules (not full stats table)
    if save_rules_full_to is not None:
        with open(save_rules_full_to, "w", encoding="utf8") as fout:
            fout.write(rules_full)

    if save_rules_pos_only_to is not None:
        with open(save_rules_pos_only_to, "w", encoding="utf8") as fout:
            fout.write(rules_pos_only)

    if save_rules_pos_simplified_to is not None:
        with open(save_rules_pos_simplified_to, "w", encoding="utf8") as fout:
            fout.write(rules_pos_simplified)

    return result

# ---------------------------
# Minimal CLI for debugging the core
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a DT on a single layer top5 label (core debug).")
    parser.add_argument("--input_csv", required=True, help="CSV containing bert tokens + features + top5_l* columns")
    parser.add_argument("--layer", required=True, help="Layer number (e.g. 4) — accepts string or int")
    parser.add_argument("--tree_seed", type=int, default=42, help="Seed for Decision Tree (determinism)")
    parser.add_argument("--save_full", type=str, default=None,
                        help="Optional path to save the full rules text (if omitted, nothing is written).")
    parser.add_argument("--save_pos", type=str, default=None,
                        help="Optional path to save only the rules leading to top5 = 1 (if omitted, nothing is written).")
    parser.add_argument("--save_simp_pos", type=str, default=None,
                        help="Optional path to save simplified rules leading to top5 = 1 (if omitted, nothing is written).")
    args = parser.parse_args()

    # parse layer argument (allow "4" or 4)
    layer_arg = args.layer
    if isinstance(layer_arg, str):
        if layer_arg.lower() == "all":
            raise ValueError("Core function trains ONE layer at a time. Use compute_decision_trees.py to iterate over layers.")
        try:
            layer_int = int(layer_arg)
        except ValueError:
            raise ValueError(f"Cannot parse layer argument: {layer_arg}")
    else:
        layer_int = int(layer_arg)

    # Load the attention CSV containing the features between is_numeric .. next_pos_id)
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)
    # Basic sanity: required feature columns present
    missing = [c for c in DEFAULT_FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Input CSV missing required feature columns: {missing}")

    label_name = f"top5_l{layer_int}"
    if label_name not in df.columns:
        raise RuntimeError(f"Input CSV missing label column: {label_name}")

    # Run training
    start = datetime.now(UTC).isoformat()
    print(f"[{start}] Training DT for layer {layer_int} ...")
    out = train_and_extract_rules_from_df(
        df,
        layer=layer_int,
        feature_cols=DEFAULT_FEATURE_COLS,
        max_depth=4,
        min_samples_leaf=20,
        dt_seed=args.tree_seed,
        save_rules_full_to=args.save_full,
        save_rules_pos_only_to=args.save_pos,
        save_rules_pos_simplified_to=args.save_simp_pos,
    )

    # Print compact summary
    print(f"n_samples: {out['n_samples']}, n_pos: {out['n_pos']}")
    print("\n--- sample of pos_only rules (first 400 chars) ---\n")
    print(out["rules_pos_only"][:400])
    print("\n--- leaf stats (top 5 by pos_rate) ---")
    # pretty print first 5 leaves sorted by pos_rate
    leaf_rows = []
    for lid, s in out["leaf_stats"].items():
        leaf_rows.append((lid, s["n_samples"], s["n_pos"], s["pos_rate"]))
    leaf_rows_sorted = sorted(leaf_rows, key=lambda t: (-t[3], -t[1]))[:5]
    for lid, n_s, n_pos, pos_rate in leaf_rows_sorted:
        print(f"Leaf {lid}: n={n_s} pos={n_pos} pos_rate={pos_rate:.4f}")

    print("\nDone.")