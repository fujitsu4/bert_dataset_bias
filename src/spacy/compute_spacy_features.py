"""
compute_spacy_features.py
Author: Zakaria JOUILIL

Description:
    Compute initial spacy features (before tokenization) for each dataset :
        - Token-specific features : is_numeric & entity_label
        - Grammatical features : is_punctuation, is_stopword, is_pronoun & pos_id
        - Syntactic features : dependency_label, depth_in_tree & num_dependents

Inputs:
    - data/cleaned/snli_filtered.csv (example)

Outputs:
    - outputs/spacy/spacy_snli_features.csv
    - outputs/spacy/spacy_label_maps (dynamic dictionnaries)

Usage:
    python -m src.spacy.compute_spacy_features --input data/cleaned/snli_filtered.csv --output outputs/spacy/spacy_snli_features.csv --label spacy_snli_label_maps.json
"""

import pandas as pd
import spacy
import json
from tqdm import tqdm
import argparse
from src.utils.paths import get_project_path


# ----------------------------------------------------------
# CLI Parser
# ----------------------------------------------------------

parser = argparse.ArgumentParser(description="Compute initial speacy features for merged datasets")
parser.add_argument("--input", type=str, default="data/cleaned/merged_datasets.csv",
                        help="Input CSV path")
parser.add_argument("--output", type=str, default="outputs/spacy/spacy_features.csv",
                        help="Output CSV path")
parser.add_argument("--label", type=str, required=True,
                    help="Label used to name the spaCy label maps file")

args = parser.parse_args()

INPUT_CSV = get_project_path(*args.input.split("/"))
OUTPUT_CSV = get_project_path(*args.output.split("/"))
LABEL_MAP_PATH = get_project_path("outputs", "spacy", f"{args.label}_label_maps.json")

print("\n[INFO] Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# dynamic dictionaries
pos2id = {}
dep2id = {}
ent2id = {}

print(f"[INFO] Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, sep=";")

records = []

print("\n[INFO] Computing spaCy features...")
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    sentence_id = row["sentence_id"]
    sentence = row["sentence"]
    dataset = row["dataset"]

    doc = nlp(sentence)

    for tok_index, token in enumerate(doc):

        # pos id
        pos_label = token.pos_
        if pos_label not in pos2id:
          pos2id[pos_label] = len(pos2id)
        pos_id = pos2id[pos_label]

        # entity id
        ent_label = token.ent_type_ if token.ent_type_ else "NONE"
        if ent_label not in ent2id:
            ent2id[ent_label] = len(ent2id)
        ent_id = ent2id[ent_label]

        # dependency id
        dep_label = token.dep_
        if dep_label not in dep2id:
            dep2id[dep_label] = len(dep2id)
        dep_id = dep2id[dep_label]

        # TREE DEPTH
        depth = 0
        cur = token
        visited = set()
        while cur.head is not cur and cur not in visited:
            visited.add(cur)
            depth += 1
            cur = cur.head

        # number of dependents
        num_dep = len(list(token.children))

        records.append({
            "sentence_id": sentence_id,
            "word_index": tok_index,
            "word": token.text,

            "is_numeric": int(token.text.isdigit()),
            "is_punctuation": int(token.is_punct),
            "is_stopword": int(token.is_stop),
            "is_pronoun": int(token.pos_ == "PRON"),

            "pos": pos_label,
            "pos_id": pos_id,

            "dep_label": dep_label,
            "dep_id": dep_id,

            "ent_label": ent_label,
            "ent_id": ent_id,

            "depth_in_tree": depth,
            "num_dependents": num_dep,
            "dataset": dataset
        })

print("\n[INFO] Saving output...")
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_CSV, sep = ";", index=False)
print(f"[INFO] SpaCy features saved to: {OUTPUT_CSV}")

with open(LABEL_MAP_PATH, "w") as f:
    json.dump({
        "pos2id": pos2id,
        "dep2id": dep2id,
        "ent2id": ent2id
    }, f, indent=4)

print(f"[INFO] Saved label maps to: {LABEL_MAP_PATH}")