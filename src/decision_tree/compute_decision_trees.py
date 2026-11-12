"""
    compute_decision_trees.py
    Author: Zakaria JOUILIL

    Description:
        Main pipeline for training decision trees on per-token features
        (pretrained or untrained BERT), extracting rules, and saving them
        in the required directory structure.

    Output structure:
        PRETRAINED → one folder only:
            outputs/decision_tree/pretrained/
                layer_01_rules_full.txt
                layer_01_rules_pos.txt
                layer_01_rules_simp_pos.txt
                ...

        UNTRAINED → one folder per seed:
            outputs/decision_tree/untrained/seed_XXXXXX/
                layer_01_rules_full.txt
                layer_01_rules_pos.txt
                layer_01_rules_simp_pos.txt
                ...

    Usage :
        Pretrained Bert :
            !python -m src.decision_tree.compute_decision_trees --model pretrained --dataset snli

    """

import argparse
import os
import pandas as pd
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.decision_tree.decision_tree_core import (
        train_and_extract_rules_from_df,
        DEFAULT_FEATURE_COLS
    )

    # ---------------------------------------------------------
    # Utility: ensure directory
    # ---------------------------------------------------------
def ensure_dir(path: str):
        Path(path).mkdir(parents=True, exist_ok=True)


    # ---------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
        parser.add_argument("--seed", type=int, default=None,
                            help="Seed required for untrained model.")
        parser.add_argument("--dataset", type=str, required=True,
                        choices=["snli", "agnews", "arxiv"],
                        help="Dataset being processed")
        args = parser.parse_args()

        input_csv = f"outputs/attention/attention_top5_{args.dataset}_{args.model}.csv"
        if args.model == "untrained":
            input_csv = f"outputs/attention/attention_top5_{args.dataset}_untrained_seed_{args.seed:03d}.csv"
        # ----------------------------------------------
        # MODEL type and output dir
        # ----------------------------------------------
        if args.model == "pretrained":
            out_dir = f"outputs/decision_tree/pretrained/{args.dataset}"
            ensure_dir(out_dir)
            print(f"[INFO] PRETRAINED mode | Dataset: {args.dataset} → saving to: {out_dir}")

        else:
            if args.seed is None:
                raise ValueError("Untrained mode requires --seed.")
            out_dir = f"outputs/decision_tree/untrained/{args.dataset}/seed_{args.seed:06d}"
            ensure_dir(out_dir)
            print(f"[INFO] UNTRAINED mode | Dataset: {args.dataset} | Seed: {args.seed} → saving to: {out_dir}")
        # ----------------------------------------------
        # LOAD DATAFRAME
        # ----------------------------------------------
        print(f"[INFO] Loading input CSV: {input_csv}")
        df = pd.read_csv(input_csv, sep=";", keep_default_na=False)

        print("\n=== EMPTY VALUES IN FEATURES ===")
        for col in DEFAULT_FEATURE_COLS:
            values = df[col].astype(str).str.strip()
            empty_count = (values == "").sum()

            if empty_count > 0:
                print(f"{col}: {empty_count} empty values")

        print("\n=== NON NUMERIC VALUES IN FEATURES ===")
        for col in DEFAULT_FEATURE_COLS:
            values = df[col].astype(str).str.strip()
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = numeric.isna() & (values != "")

            if invalid.any():
                print(f"\n{col}:")
                print(values[invalid].value_counts().head(10))


        # Sanity check for required columns
        required_columns = [
            "sentence_id", "bert_index", "bert_token",
            # feature columns must also be included, but we trust core to validate
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column in input CSV: {col}")

        # ----------------------------------------------
        # TRAIN + EXTRACT RULES FOR 12 LAYERS
        # ----------------------------------------------
        print("[INFO] Starting Decision Tree extraction for 12 layers ...")

        NUM_LAYERS = 12
        for layer in range(1, NUM_LAYERS + 1):
            print(f"\n[INFO] Processing layer {layer}/12 ...")

            # -------------------------------------------------------------
            # TRAIN MODEL AND EXTRACT 3 RULESETS
            # -------------------------------------------------------------
            rules = train_and_extract_rules_from_df(df,
            layer=layer,
            feature_cols=DEFAULT_FEATURE_COLS,
            max_depth=4,
            min_samples_leaf=20)

            # -------------------------------------------------------------
            # SAVE 3 OUTPUT FILES
            # -------------------------------------------------------------
            layer_tag = f"layer_{layer:02d}"  # 01 → 12

            filenames = {
                "rules_full": f"{layer_tag}_rules_full.txt",
                "rules_pos_only": f"{layer_tag}_rules_pos.txt",
                "rules_pos_simplified": f"{layer_tag}_rules_simp_pos.txt",
            }

            for key, fname in filenames.items():
                out_path = os.path.join(out_dir, fname)
                print(f"[SAVE] → {out_path}")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(rules[key])

        print("\n[OK] DONE — All 12 layers processed successfully.")


if __name__ == "__main__":
        main()