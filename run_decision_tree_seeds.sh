#Check ne marche pas !!!!
# run_decision_tree_seeds.sh
# Author : Zakaria JOUILIL
# Description  : The script:
#  - ignores the first line (timestamp)
#  - reads the SEEDS list on second line
#  - for each seed: if outputs/decision_tree/${dataset}/untrained/seed_${seed} exists -> skip
#    else run the Python script for that seed
#  - stops after --max-runs jobs in this invocation (default 5)
# Usage (Colab):
# !apt-get install dos2unix -y
# !dos2unix run_decision_tree_seeds.sh
# !bash run_decision_tree_seeds.sh /content/bert_dataset_bias/outputs/attention/seeds_list.txt --max-runs 6

set -uo pipefail

SEEDS_FILE="${1:-seeds_list.txt}"
DATASET="${2:-}"

if [[ -z "$DATASET" ]]; then
  echo "ERROR: dataset is required."
  echo "Usage: bash run_decision_tree_seeds.sh seeds_list.txt snli --max-runs 5"
  exit 2
fi

# Validate dataset
if [[ ! "$DATASET" =~ ^(snli|mnli|agnews|arxiv)$ ]]; then
  echo "ERROR: Invalid dataset: $DATASET"
  echo "Allowed datasets: snli, mnli, agnews, arxiv"
  exit 2
fi

MAX_RUNS=5

# Optional argument --max-runs N
if [[ "${3:-}" == "--max-runs" && -n "${4:-}" ]]; then
  MAX_RUNS="$4"
fi

if [[ ! -f "$SEEDS_FILE" ]]; then
  echo "ERROR: seeds file not found: $SEEDS_FILE"
  exit 2
fi

# Read the second line (SEEDS = [...])
SEED_LINE=$(sed -n '2p' "$SEEDS_FILE" | sed -e 's/^SEEDS *= *//')
SEED_LINE="${SEED_LINE#[}"
SEED_LINE="${SEED_LINE%]}"

IFS=',' read -ra RAW <<< "$SEED_LINE"

# Prepare output directory
OUT_DIR="outputs/decision_tree/${DATASET}/untrained/"
mkdir -p "$OUT_DIR"

runs=0

for raw in "${RAW[@]}"; do
  seed="$(echo "$raw" | xargs)"

  if [[ -z "$seed" ]]; then
    continue
  fi

  # Match Python output directory format
  printf -v seed_padded "%06d" "$seed"

  out_dir="${OUT_DIR}/seed_${seed_padded}"

  if [[ -d "$out_dir" ]]; then
    echo "[SKIP] seed ${seed} -> directory ${out_dir} already exists"
    continue
  fi

  if (( runs >= MAX_RUNS )); then
    echo "[INFO] Reached max runs (${MAX_RUNS}). Stopping."
    exit 0
  fi

  echo "[RUN] dataset=${DATASET}, seed=${seed}"

  python /content/bert_dataset_bias/src/decision_tree/compute_decision_trees.py \
    --model untrained \
    --seed "${seed}" \
    --dataset "${DATASET}"

  status=$?
  if [[ $status -ne 0 ]]; then
    echo "[ERROR] Run for seed ${seed} failed with status ${status}. Stopping."
    exit $status
  fi

  ((runs++))
done

echo "[DONE] Dataset=${DATASET} | Completed ${runs} runs."
exit 0