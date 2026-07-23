#!/bin/bash
# Full Mutagenicity continued SFT from the stable AIDS SFT-v3 adapter.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mut_sft_full
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

PWD_BEFORE_CD="$PWD"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] Could not determine PROJECT_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
if [[ ! -f "$PROJECT_ROOT/scripts/train_mutagenicity_continued_sft.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

resolve_project_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$PROJECT_ROOT/$value"
  fi
}

FINAL_DATA_ROOT="$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v1"
FALLBACK_DATA_ROOT="$PROJECT_ROOT/outputs/hpc/mutagenicity/sft_ppo_data_v1"
if [[ -n "${DATA_ROOT:-}" ]]; then
  DATA_ROOT="$(resolve_project_path "$DATA_ROOT")"
elif [[ -d "$FINAL_DATA_ROOT" ]]; then
  DATA_ROOT="$FINAL_DATA_ROOT"
else
  DATA_ROOT="$FALLBACK_DATA_ROOT"
fi
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/mutagenicity_sft_train.csv}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/mutagenicity_sft_val.csv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_NAME_OR_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$PROJECT_ROOT/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$BASE_MODEL_PATH}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/sft_continued_v1}"
LOG_DIR="$PROJECT_ROOT/logs"

TRAIN_CSV="$(resolve_project_path "$TRAIN_CSV")"
VAL_CSV="$(resolve_project_path "$VAL_CSV")"
BASE_MODEL_PATH="$(resolve_project_path "$BASE_MODEL_PATH")"
BASE_CHECKPOINT="$(resolve_project_path "$BASE_CHECKPOINT")"
TOKENIZER_PATH="$(resolve_project_path "$TOKENIZER_PATH")"
OUTPUT_ROOT="$(resolve_project_path "$OUTPUT_ROOT")"

MAX_STEPS="${MAX_STEPS:-500}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SEED="${SEED:-7}"
GENERATION_SAMPLES="${GENERATION_SAMPLES:-250}"

mkdir -p "$LOG_DIR"
for required in "$TRAIN_CSV" "$VAL_CSV" "$BASE_MODEL_PATH" "$BASE_CHECKPOINT" "$TOKENIZER_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Required input does not exist: $required" >&2
    exit 2
  fi
done

echo "===== MUTAGENICITY CONTINUED SFT FULL ====="
echo "hostname=$(hostname)"
echo "date=$(date)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD before cd=$PWD_BEFORE_CD"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PWD after cd=$PWD"
echo "PYTHONPATH=$PYTHONPATH"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python=$(which python)"
python --version
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
echo "DATA_ROOT=$DATA_ROOT"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "VAL_CSV=$VAL_CSV"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "BASE_CHECKPOINT=$BASE_CHECKPOINT"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MAX_STEPS=$MAX_STEPS"
echo "MAX_SEQUENCE_LENGTH=$MAX_SEQUENCE_LENGTH"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "SAVE_STEPS=$SAVE_STEPS"
echo "EVAL_STEPS=$EVAL_STEPS"
python - "$TRAIN_CSV" "$VAL_CSV" <<'PY'
import csv
import sys
for name, path in (("train", sys.argv[1]), ("val", sys.argv[2])):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        print(f"{name}_rows={sum(1 for _ in csv.DictReader(handle))}")
PY

python scripts/train_mutagenicity_continued_sft.py \
  --config configs/hpc.yaml \
  --config configs/train/mutagenicity_continued_sft.yaml \
  --mode full \
  --data-root "$DATA_ROOT" \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --base-model-path "$BASE_MODEL_PATH" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --max-train-rows 0 \
  --max-val-rows 0 \
  --max-steps "$MAX_STEPS" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --logging-steps "$LOGGING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --generation-samples "$GENERATION_SAMPLES" \
  --seed "$SEED"

test -s "$OUTPUT_ROOT/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_CONTINUED_SFT_FULL_OK]"
