#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=mut_sftppo_smoke
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
if [[ ! -f "$PROJECT_ROOT/scripts/data/build_mutagenicity_sft_ppo_data.py" ]] || \
   [[ ! -f "$PROJECT_ROOT/scripts/data/build_mutagenicity_teacher_consistent_views.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

LOG_DIR="$PROJECT_ROOT/logs"
PROCESSED_ROOT="${PROCESSED_ROOT:-$PROJECT_ROOT/outputs/hpc/datasets/final/mutagenicity_v1_processed}"
TEACHER_ROOT="${TEACHER_ROOT:-$PROJECT_ROOT/outputs/hpc/oracle/final/mutagenicity_rf_v1}"
TEACHER_CONSISTENT_ROOT="${TEACHER_CONSISTENT_ROOT:-$PROJECT_ROOT/outputs/hpc/datasets/mutagenicity_v1_teacher_consistent}"
TEACHER_PATH="${TEACHER_PATH:-$TEACHER_ROOT/mutagenicity_rf_model.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/sft_ppo_data_v1_smoke}"

resolve_project_path() {
  local path_value="$1"
  if [[ "$path_value" = /* ]]; then
    printf '%s\n' "$path_value"
  else
    printf '%s\n' "$PROJECT_ROOT/$path_value"
  fi
}

PROCESSED_ROOT="$(resolve_project_path "$PROCESSED_ROOT")"
TEACHER_ROOT="$(resolve_project_path "$TEACHER_ROOT")"
TEACHER_CONSISTENT_ROOT="$(resolve_project_path "$TEACHER_CONSISTENT_ROOT")"
TEACHER_PATH="$(resolve_project_path "$TEACHER_PATH")"
OUTPUT_ROOT="$(resolve_project_path "$OUTPUT_ROOT")"
PROCESSED_TRAIN="$PROCESSED_ROOT/train.csv"
PROCESSED_VAL="$PROCESSED_ROOT/val.csv"
PROCESSED_CALIBRATION="$PROCESSED_ROOT/calibration.csv"
PROCESSED_TEST="$PROCESSED_ROOT/test.csv"
PREDICTIONS_TRAIN="$TEACHER_ROOT/predictions_train.csv"
PREDICTIONS_VAL="$TEACHER_ROOT/predictions_val.csv"
PREDICTIONS_CALIBRATION="$TEACHER_ROOT/predictions_calibration.csv"
PREDICTIONS_TEST="$TEACHER_ROOT/predictions_test.csv"
TRAIN_INPUT="$TEACHER_CONSISTENT_ROOT/train_source_label1_teacher_correct.csv"
VAL_INPUT="$TEACHER_CONSISTENT_ROOT/val_source_label1_teacher_correct.csv"
CALIBRATION_INPUT="$TEACHER_CONSISTENT_ROOT/calibration_source_label1_teacher_correct.csv"
TEST_INPUT="$TEACHER_CONSISTENT_ROOT/test_source_label1_teacher_correct.csv"
MAX_TRAIN_PARENTS="${MAX_TRAIN_PARENTS:-100}"
MAX_VAL_PARENTS="${MAX_VAL_PARENTS:-40}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "===== MUTAGENICITY SFT/PPO DATA SMOKE ====="
echo "hostname=$(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD before cd=$PWD_BEFORE_CD"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PWD after cd=$PWD"
echo "PYTHONPATH=$PYTHONPATH"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python=$(which python)"
python --version
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "PROCESSED_ROOT=$PROCESSED_ROOT"
echo "TEACHER_ROOT=$TEACHER_ROOT"
echo "TEACHER_CONSISTENT_ROOT=$TEACHER_CONSISTENT_ROOT"
echo "TRAIN_INPUT=$TRAIN_INPUT"
echo "VAL_INPUT=$VAL_INPUT"
echo "CALIBRATION_INPUT=$CALIBRATION_INPUT"
echo "TEST_INPUT=$TEST_INPUT"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MAX_TRAIN_PARENTS=$MAX_TRAIN_PARENTS"
echo "MAX_VAL_PARENTS=$MAX_VAL_PARENTS"
echo "SOURCE_LABEL=1"
echo "TARGET_LABEL=0"

for required_file in \
  "$PROCESSED_TRAIN" \
  "$PROCESSED_VAL" \
  "$PROCESSED_CALIBRATION" \
  "$PROCESSED_TEST" \
  "$PREDICTIONS_TRAIN" \
  "$PREDICTIONS_VAL" \
  "$PREDICTIONS_CALIBRATION" \
  "$PREDICTIONS_TEST" \
  "$TEACHER_PATH"; do
  if [[ ! -s "$required_file" ]]; then
    echo "[ERROR] Required input is missing or empty: $required_file" >&2
    exit 2
  fi
done

python scripts/data/build_mutagenicity_teacher_consistent_views.py \
  --config configs/hpc.yaml \
  --processed-root "$PROCESSED_ROOT" \
  --teacher-root "$TEACHER_ROOT" \
  --output-dir "$TEACHER_CONSISTENT_ROOT"

for generated_file in \
  "$TRAIN_INPUT" \
  "$VAL_INPUT" \
  "$CALIBRATION_INPUT" \
  "$TEST_INPUT"; do
  if [[ ! -s "$generated_file" ]]; then
    echo "[ERROR] Teacher-consistent view is missing or empty: $generated_file" >&2
    exit 2
  fi
done

python scripts/data/build_mutagenicity_sft_ppo_data.py \
  --config configs/hpc.yaml \
  --config configs/data/mutagenicity_sft_ppo.yaml \
  --teacher-consistent-root "$TEACHER_CONSISTENT_ROOT" \
  --teacher-path "$TEACHER_PATH" \
  --output-dir "$OUTPUT_ROOT" \
  --max-train-parents "$MAX_TRAIN_PARENTS" \
  --max-val-parents "$MAX_VAL_PARENTS" \
  --seed "$SEED" \
  --use-teacher-ranking

echo "[MUTAGENICITY_SFT_PPO_SMOKE_OK]"
