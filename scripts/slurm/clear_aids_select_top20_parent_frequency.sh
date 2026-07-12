#!/bin/bash
#SBATCH --job-name=clear_parent_freq20
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CLEAR_CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

CANDIDATE_CSV=${CANDIDATE_CSV:-${PROJECT_ROOT}/outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv}
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hpc/baselines/clear/aids/selected_parent_frequency}
TOP_K=${TOP_K:-20}
TARGET_LABEL=${TARGET_LABEL:-1}
CANDIDATE_SMILES_COL=${CANDIDATE_SMILES_COL:-candidate_smiles}
SELECTION_MODE=${SELECTION_MODE:-parent_frequency}

echo "===== CLEAR PARENT-FREQUENCY TOP20 SELECTOR ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CANDIDATE_CSV=${CANDIDATE_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TOP_K=${TOP_K}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "CANDIDATE_SMILES_COL=${CANDIDATE_SMILES_COL}"
echo "SELECTION_MODE=${SELECTION_MODE}"
echo "CF_MODE=strict_flip"

if [ "${SELECTION_MODE}" != "parent_frequency" ]; then
  echo "[ERROR] This wrapper only supports SELECTION_MODE=parent_frequency."
  exit 2
fi
if [ ! -f "${CANDIDATE_CSV}" ]; then
  echo "[ERROR] CLEAR candidate CSV not found: ${CANDIDATE_CSV}"
  exit 2
fi
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] RF teacher not found: ${TEACHER_PATH}"
  exit 2
fi

python scripts/baselines/clear/select_clear_global_topk.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --selection-mode "${SELECTION_MODE}" \
  --candidate-csv "${CANDIDATE_CSV}" \
  --teacher-path "${TEACHER_PATH}" \
  --out-dir "${OUTPUT_DIR}" \
  --candidate-smiles-col "${CANDIDATE_SMILES_COL}" \
  --target-label "${TARGET_LABEL}" \
  --top-k "${TOP_K}"

echo "[CLEAR_PARENT_FREQUENCY_OUTPUTS]"
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort
