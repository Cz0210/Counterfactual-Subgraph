#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=mut_rf_smoke
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

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

if [[ ! -f "$PROJECT_ROOT/scripts/train_mutagenicity_rf_teacher.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

resolve_project_path() {
  local path_value="$1"
  if [[ "$path_value" = /* ]]; then
    printf '%s\n' "$path_value"
  else
    printf '%s\n' "$PROJECT_ROOT/$path_value"
  fi
}

MUTAGENICITY_DATA_ROOT="${MUTAGENICITY_DATA_ROOT:-${DATA_DIR:-$PROJECT_ROOT/outputs/hpc/datasets/final/mutagenicity_v1_processed}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_DIR:-$PROJECT_ROOT/outputs/hpc/oracle/mutagenicity_rf_v1_smoke}}"
MUTAGENICITY_DATA_ROOT="$(resolve_project_path "$MUTAGENICITY_DATA_ROOT")"
OUTPUT_ROOT="$(resolve_project_path "$OUTPUT_ROOT")"
LOG_DIR="$PROJECT_ROOT/logs"

TRAIN_CSV="$MUTAGENICITY_DATA_ROOT/train.csv"
VAL_CSV="$MUTAGENICITY_DATA_ROOT/val.csv"
CALIBRATION_CSV="$MUTAGENICITY_DATA_ROOT/calibration.csv"
TEST_CSV="$MUTAGENICITY_DATA_ROOT/test.csv"

SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
RADIUS=${RADIUS:-2}
N_BITS=${N_BITS:-2048}
N_ESTIMATORS_GRID=${N_ESTIMATORS_GRID:-50,100}
MAX_DEPTH_GRID=${MAX_DEPTH_GRID:-none,20}
MIN_SAMPLES_LEAF_GRID=${MIN_SAMPLES_LEAF_GRID:-1,2}
SELECTION_METRIC=${SELECTION_METRIC:-balanced_accuracy}
RANDOM_SEED=${RANDOM_SEED:-42}
N_JOBS=${N_JOBS:-7}

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "===== MUTAGENICITY RF TEACHER SMOKE ====="
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MUTAGENICITY_DATA_ROOT=$MUTAGENICITY_DATA_ROOT"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "VAL_CSV=$VAL_CSV"
echo "CALIBRATION_CSV=$CALIBRATION_CSV"
echo "TEST_CSV=$TEST_CSV"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SMILES_COL=${SMILES_COL}"
echo "LABEL_COL=${LABEL_COL}"
echo "SOURCE_LABEL=1"
echo "TARGET_LABEL=0"
echo "RADIUS=${RADIUS}"
echo "N_BITS=${N_BITS}"
echo "N_ESTIMATORS_GRID=${N_ESTIMATORS_GRID}"
echo "MAX_DEPTH_GRID=${MAX_DEPTH_GRID}"
echo "MIN_SAMPLES_LEAF_GRID=${MIN_SAMPLES_LEAF_GRID}"
echo "SELECTION_METRIC=${SELECTION_METRIC}"

for input_csv in "$TRAIN_CSV" "$VAL_CSV" "$CALIBRATION_CSV" "$TEST_CSV"; do
  if [[ ! -s "$input_csv" ]]; then
    echo "[ERROR] Required split CSV is missing or empty: $input_csv" >&2
    exit 2
  fi
done

python scripts/train_mutagenicity_rf_teacher.py \
  --config configs/hpc.yaml \
  --data-dir "$MUTAGENICITY_DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --radius "${RADIUS}" \
  --n-bits "${N_BITS}" \
  --n-estimators-grid "${N_ESTIMATORS_GRID}" \
  --max-depth-grid "${MAX_DEPTH_GRID}" \
  --min-samples-leaf-grid "${MIN_SAMPLES_LEAF_GRID}" \
  --selection-metric "${SELECTION_METRIC}" \
  --random-seed "${RANDOM_SEED}" \
  --n-jobs "${N_JOBS}"

python scripts/evaluate_mutagenicity_rf_teacher.py \
  --config configs/hpc.yaml \
  --data-dir "$MUTAGENICITY_DATA_ROOT" \
  --model-path "$OUTPUT_ROOT/mutagenicity_rf_model.pkl" \
  --output-dir "$OUTPUT_ROOT" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}"

echo "[MUTAGENICITY_RF_TEACHER_SMOKE_OK]"
