#!/bin/bash
# Generate and audit the strict train-only Mutagenicity GlobalGCE pool.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=mut_globalgce
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] Could not determine PROJECT_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

: "${TRAIN_CSV:?TRAIN_CSV must be explicitly provided}"
: "${TEACHER_PATH:?TEACHER_PATH must be explicitly provided}"
: "${OFFICIAL_ROOT:?OFFICIAL_ROOT must be explicitly provided}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be explicitly provided}"

resolve_from_project_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

TRAIN_CSV="$(resolve_from_project_root "$TRAIN_CSV")"
TEACHER_PATH="$(resolve_from_project_root "$TEACHER_PATH")"
OFFICIAL_ROOT="$(resolve_from_project_root "$OFFICIAL_ROOT")"
OUTPUT_DIR="$(resolve_from_project_root "$OUTPUT_DIR")"
NATIVE_TRAIN_CSV="${NATIVE_TRAIN_CSV:-outputs/hpc/datasets/final/mutagenicity_v1_processed/train.csv}"
NATIVE_TRAIN_CSV="$(resolve_from_project_root "$NATIVE_TRAIN_CSV")"

PARENT_LIMIT="${PARENT_LIMIT:-0}"
SEED="${SEED:-13}"
EPOCHS="${EPOCHS:-100}"
TOP_K_NATIVE="${TOP_K_NATIVE:-20}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
DROPOUT="${DROPOUT:-0.5}"
RESUME="${RESUME:-true}"
EXPECTED_PARENT_COUNT="${EXPECTED_PARENT_COUNT:-1448}"
EXPECTED_INPUT_TRAIN_COUNT="${EXPECTED_INPUT_TRAIN_COUNT:-$EXPECTED_PARENT_COUNT}"
if [[ "$PARENT_LIMIT" -gt 0 ]]; then
  EXPECTED_SELECTED_PARENT_COUNT="${EXPECTED_SELECTED_PARENT_COUNT:-$PARENT_LIMIT}"
else
  EXPECTED_SELECTED_PARENT_COUNT="${EXPECTED_SELECTED_PARENT_COUNT:-$EXPECTED_INPUT_TRAIN_COUNT}"
fi
DEVICE="${DEVICE:-cuda}"
GENERATION_CHUNK_SIZE="${GENERATION_CHUNK_SIZE:-32}"
GENERATION_NUM_WORKERS="${GENERATION_NUM_WORKERS:-0}"
MEMORY_LOG_EVERY_CHUNKS="${MEMORY_LOG_EVERY_CHUNKS:-1}"

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for path in "$TRAIN_CSV" "$TEACHER_PATH" "$NATIVE_TRAIN_CSV"; do
  if [[ ! -s "$path" ]]; then
    echo "[ERROR] Required input file is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$OFFICIAL_ROOT" ]]; then
  echo "[ERROR] OFFICIAL_ROOT is not a directory: $OFFICIAL_ROOT" >&2
  exit 2
fi
if [[ -s "$OUTPUT_DIR/_RUN_COMPLETE.json" ]]; then
  echo "[ERROR] Completed OUTPUT_DIR cannot be rerun: $OUTPUT_DIR" >&2
  exit 2
fi

RESUME_NORMALIZED="$(printf '%s' "$RESUME" | tr '[:upper:]' '[:lower:]')"
case "$RESUME_NORMALIZED" in
  true|1|yes|on) RESUME_FLAG="--resume" ;;
  false|0|no|off) RESUME_FLAG="--no-resume" ;;
  *)
    echo "[ERROR] RESUME must be a boolean value" >&2
    exit 2
    ;;
esac

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  if [[ "$RESUME_FLAG" == "--no-resume" ]]; then
    echo "[ERROR] OUTPUT_DIR is non-empty and RESUME=false: $OUTPUT_DIR" >&2
    exit 2
  fi
  for required in run_manifest.json resume_checkpoint.json; do
    if [[ ! -s "$OUTPUT_DIR/$required" ]]; then
      echo "[ERROR] Resume requires $OUTPUT_DIR/$required" >&2
      exit 2
    fi
  done
fi
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"

echo "===== MUTAGENICITY GLOBALGCE TRAIN POOL ====="
echo "host=$(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "python=$(which python)"
echo "git_commit=$(git rev-parse HEAD)"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "NATIVE_TRAIN_CSV=$NATIVE_TRAIN_CSV"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "OFFICIAL_ROOT=$OFFICIAL_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PARENT_LIMIT=$PARENT_LIMIT"
echo "SEED=$SEED"
echo "EPOCHS=$EPOCHS"
echo "TOP_K_NATIVE=$TOP_K_NATIVE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "DROPOUT=$DROPOUT"
echo "DEVICE=$DEVICE"
echo "RESUME=$RESUME"
echo "EXPECTED_INPUT_TRAIN_COUNT=$EXPECTED_INPUT_TRAIN_COUNT"
echo "EXPECTED_SELECTED_PARENT_COUNT=$EXPECTED_SELECTED_PARENT_COUNT"
echo "GENERATION_CHUNK_SIZE=$GENERATION_CHUNK_SIZE"
echo "GENERATION_NUM_WORKERS=$GENERATION_NUM_WORKERS"
echo "MEMORY_LOG_EVERY_CHUNKS=$MEMORY_LOG_EVERY_CHUNKS"
echo "MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
python --version
nvidia-smi || true

python scripts/baselines/globalgce/build_mutagenicity_train_pool.py \
  --config configs/hpc.yaml \
  --train-csv "$TRAIN_CSV" \
  --native-train-csv "$NATIVE_TRAIN_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --official-root "$OFFICIAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --parent-limit "$PARENT_LIMIT" \
  --expected-parent-count "$EXPECTED_INPUT_TRAIN_COUNT" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --top-k-native "$TOP_K_NATIVE" \
  --learning-rate "$LEARNING_RATE" \
  --dropout "$DROPOUT" \
  --device "$DEVICE" \
  --generation-chunk-size "$GENERATION_CHUNK_SIZE" \
  --generation-num-workers "$GENERATION_NUM_WORKERS" \
  --memory-log-every-chunks "$MEMORY_LOG_EVERY_CHUNKS" \
  "$RESUME_FLAG" \
  --forbid-calibration-test

python scripts/baselines/globalgce/audit_mutagenicity_train_pool.py \
  --config configs/hpc.yaml \
  --run-dir "$OUTPUT_DIR" \
  --train-csv "$TRAIN_CSV" \
  --expected-parent-count "$EXPECTED_SELECTED_PARENT_COUNT" \
  --expected-input-train-count "$EXPECTED_INPUT_TRAIN_COUNT" \
  --require-target-label-zero \
  --require-unique-universe \
  --forbid-calibration-test \
  --require-complete

test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$OUTPUT_DIR/train_pool_audit.json"
echo "[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_OK]"
