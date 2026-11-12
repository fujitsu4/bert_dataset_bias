# run_attention_seeds.sh
# Author : Zakaria JOUILIL
# Description  : The script:
#  - ignores the first line (timestamp)
#  - reads the SEEDS list on second line
#  - for each seed: if outputs/attention/attention_top5_untrained_seed_${seed}.csv exists -> skip
#    else run the Python script for that seed
#  - stops after --max-runs jobs in this invocation (default 5)
# Usage (Colab):
# !apt-get install dos2unix -y
# !dos2unix run_attention_seeds.sh
# !bash run_attention_seeds.sh /content/bert_dataset_bias/outputs/attention/seeds_list.txt snli --max-runs 5


set -uo pipefail

SEEDS_FILE="${1:-seeds_list.txt}"
DATASET="${2:-}"

if [[ -z "$DATASET" ]]; then
  echo "ERROR: dataset is required."
  echo "Usage: bash run_attention_seeds.sh seeds_list.txt snli --max-runs 3"
  exit 2
fi

# Validate dataset
if [[ ! "$DATASET" =~ ^(snli|agnews|arxiv)$ ]]; then
  echo "ERROR: Invalid dataset: $DATASET"
  echo "Allowed datasets: snli, agnews, arxiv"
  exit 2
fi

MAX_RUNS=5

# Optional argument --max-runs N
if [[ "${3:-}" == "--max-runs" && -n "${4:-}" ]]; then
  MAX_RUNS="$4"
fi

if [[ ! "$SEEDS_FILE" ]]; then
  echo "ERROR: seeds file not found: $SEEDS_FILE"
  exit 2
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
OUT_DIR="outputs/attention"
mkdir -p "$OUT_DIR"

runs=0

for raw in "${RAW[@]}"; do
  seed="$(echo "$raw" | xargs)"

  if [[ -z "$seed" ]]; then
    continue
  fi

  # Match Python's three-digit seed formatting
  printf -v seed_padded "%03d" "$seed"

  out_file="${OUT_DIR}/attention_top5_${DATASET}_untrained_seed_${seed_padded}.csv"

  if [[ -f "$out_file" ]]; then
    echo "[SKIP] seed ${seed} -> ${out_file} already exists"
    continue
  fi

  if (( runs >= MAX_RUNS )); then
    echo "[INFO] Reached max runs (${MAX_RUNS}). Stopping."
    exit 0
  fi

  echo "[RUN] dataset=${DATASET}, seed=${seed}"

  python /content/bert_dataset_bias/src/attention/compute_attention_top5.py \
    --model untrained \
    --seed "${seed}" \
    --dataset "${DATASET}" \
    --input_csv "/content/bert_dataset_bias/outputs/bert/bert_${DATASET}_final_features.csv"

  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[ERROR] Run for seed ${seed} failed with status ${status}. Stopping."
    exit $status
  fi

  ((runs++))
done

echo "[DONE] Dataset=${DATASET} | Completed ${runs} runs."
exit 0