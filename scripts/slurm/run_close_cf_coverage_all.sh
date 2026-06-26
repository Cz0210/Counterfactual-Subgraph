#!/bin/bash
#SBATCH -J close_cf_all
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

LABEL=${LABEL:-1}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
DATASET_CSV=${DATASET_CSV:-${PROJECT_ROOT}/data/raw/AIDS/HIV.csv}
OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-${PROJECT_ROOT}/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl}
GED_THRESHOLDS=${GED_THRESHOLDS:-0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.20}
EMBEDDING_THRESHOLDS=${EMBEDDING_THRESHOLDS:-0.02,0.05,0.10,0.15,0.20,0.25,0.30}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/hpc/eval_close_cf_coverage/all_$(date +%Y%m%d_%H%M%S)}

if [ -z "${GCF_CANDIDATES_PATH:-}" ]; then
  echo "[ERROR] GCF_CANDIDATES_PATH is required for gcf baseline evaluation."
  exit 2
fi

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "python path: $(which python)"
python --version
python - <<'PY'
import importlib.util
print("rdkit available:", importlib.util.find_spec("rdkit") is not None)
print("networkx available:", importlib.util.find_spec("networkx") is not None)
print("matplotlib available:", importlib.util.find_spec("matplotlib") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
echo "DATASET_CSV=${DATASET_CSV}"
echo "OURS_SELECTED_PATH=${OURS_SELECTED_PATH}"
echo "GCF_CANDIDATES_PATH=${GCF_CANDIDATES_PATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "====================="

python scripts/run_close_cf_coverage_all.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gcf-candidates-path "${GCF_CANDIDATES_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --label "${LABEL}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --ged-thresholds "${GED_THRESHOLDS}" \
  --embedding-thresholds "${EMBEDDING_THRESHOLDS}" \
  --output-root "${OUTPUT_ROOT}" \
  --min-cf-drop 0.0

cat "${OUTPUT_ROOT}/combined/combined_report.md"
find "${OUTPUT_ROOT}/combined/figures" -type f | sort
