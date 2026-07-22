#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=mut_rf_full
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_DIR=${DATA_DIR:-outputs/hpc/datasets/final/mutagenicity_v1_processed}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/oracle/mutagenicity_rf_v1}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
RADIUS=${RADIUS:-2}
N_BITS=${N_BITS:-2048}
N_ESTIMATORS_GRID=${N_ESTIMATORS_GRID:-300,600}
MAX_DEPTH_GRID=${MAX_DEPTH_GRID:-none,20,40}
MIN_SAMPLES_LEAF_GRID=${MIN_SAMPLES_LEAF_GRID:-1,2}
SELECTION_METRIC=${SELECTION_METRIC:-balanced_accuracy}
RANDOM_SEED=${RANDOM_SEED:-42}
N_JOBS=${N_JOBS:-7}

mkdir -p logs "${OUTPUT_DIR}"

echo "===== MUTAGENICITY RF TEACHER FULL ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python=$(which python)"
python --version
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
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

for split in train val calibration test; do
  test -s "${DATA_DIR}/${split}.csv"
done

python scripts/train_mutagenicity_rf_teacher.py \
  --config configs/hpc.yaml \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
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
  --data-dir "${DATA_DIR}" \
  --model-path "${OUTPUT_DIR}/mutagenicity_rf_model.pkl" \
  --output-dir "${OUTPUT_DIR}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}"

echo "[MUTAGENICITY_RF_TEACHER_FULL_OK]"

