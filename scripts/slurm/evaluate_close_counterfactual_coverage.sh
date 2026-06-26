#!/bin/bash
#SBATCH -J close_cf_eval
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

MODE=${MODE:-ours}
DISTANCE_TYPE=${DISTANCE_TYPE:-ged}
GED_MODE=${GED_MODE:-delete}
LABEL=${LABEL:-1}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
THRESHOLDS=${THRESHOLDS:-0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.20}
EMBEDDING_THRESHOLDS=${EMBEDDING_THRESHOLDS:-0.02,0.05,0.10,0.15,0.20,0.25,0.30}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/eval_close_cf_coverage/${MODE}_${DISTANCE_TYPE}_$(date +%Y%m%d_%H%M%S)}

if [ -z "${DATASET_CSV:-}" ]; then
  DATASET_CSV=${PROJECT_ROOT}/data/raw/AIDS/HIV.csv
fi
if [ -z "${TEACHER_PATH:-}" ]; then
  TEACHER_PATH=${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl
fi
if [ "${MODE}" = "ours" ] && [ -z "${SELECTED_SUBGRAPHS_PATH:-}" ]; then
  SELECTED_SUBGRAPHS_PATH=${PROJECT_ROOT}/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20
fi
if [ "${MODE}" = "gcf" ] && [ -z "${GCF_CANDIDATES_PATH:-}" ]; then
  echo "[ERROR] GCF_CANDIDATES_PATH is required when MODE=gcf."
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
echo "MODE=${MODE}"
echo "DISTANCE_TYPE=${DISTANCE_TYPE}"
echo "DATASET_CSV=${DATASET_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "====================="

python scripts/evaluate_close_counterfactual_coverage.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --mode "${MODE}" \
  --dataset-csv "${DATASET_CSV}" \
  --selected-subgraphs-path "${SELECTED_SUBGRAPHS_PATH:-}" \
  --gcf-candidates-path "${GCF_CANDIDATES_PATH:-}" \
  --teacher-path "${TEACHER_PATH}" \
  --label "${LABEL}" \
  --distance-type "${DISTANCE_TYPE}" \
  --ged-mode "${GED_MODE}" \
  --thresholds "${THRESHOLDS}" \
  --embedding-thresholds "${EMBEDDING_THRESHOLDS}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --output-dir "${OUTPUT_DIR}"

cat "${OUTPUT_DIR}/report.md"
