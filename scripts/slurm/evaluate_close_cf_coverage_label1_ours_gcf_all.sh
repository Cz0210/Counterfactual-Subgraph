#!/bin/bash
#SBATCH -J close_cf_l1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

# Some HPC bashrc scripts reference unset variables, so nounset must not be
# enabled before shell/conda initialization.
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

LABEL=${LABEL:-1}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
GED_THRESHOLDS=${GED_THRESHOLDS:-0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.20}
EMBEDDING_THRESHOLDS=${EMBEDDING_THRESHOLDS:-0.02,0.05,0.10,0.15,0.20,0.25,0.30}

OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-${PROJECT_ROOT}/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/hpc/eval_close_cf_coverage/label1_ours_vs_gcf_$(date +%Y%m%d_%H%M%S)}

if [ -z "${DATASET_CSV:-}" ]; then
  for candidate in \
    "${PROJECT_ROOT}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv" \
    "${PROJECT_ROOT}/data/processed/label1_parents.csv" \
    "${PROJECT_ROOT}/data/raw/AIDS/HIV.csv"; do
    if [ -f "${candidate}" ]; then
      DATASET_CSV="${candidate}"
      break
    fi
  done
fi

if [ -z "${DATASET_CSV:-}" ] || [ ! -f "${DATASET_CSV}" ]; then
  echo "[ERROR] DATASET_CSV was not found automatically. Please submit with DATASET_CSV=/path/to/parents.csv"
  exit 2
fi

if [ -z "${GCF_CANDIDATES_PATH:-}" ]; then
  echo "[ERROR] GCF_CANDIDATES_PATH is required for gcf baseline evaluation."
  echo "Example: sbatch --export=ALL,GCF_CANDIDATES_PATH=/path/to/gcf_candidates.csv scripts/slurm/evaluate_close_cf_coverage_label1_ours_gcf_all.sh"
  exit 2
fi

if [ -z "${TEACHER_PATH:-}" ]; then
  for candidate in \
    "${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl" \
    "${PROJECT_ROOT}/outputs/oracle/aids_rf_model.pkl" \
    "${PROJECT_ROOT}/outputs/hpc/teacher/aids_rf_model.pkl"; do
    if [ -f "${candidate}" ]; then
      TEACHER_PATH="${candidate}"
      break
    fi
  done
fi

if [ -z "${TEACHER_PATH:-}" ] || [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] TEACHER_PATH was not found automatically. Please submit with TEACHER_PATH=/path/to/aids_rf_model.pkl"
  exit 2
fi

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
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
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATASET_CSV=${DATASET_CSV}"
echo "OURS_SELECTED_PATH=${OURS_SELECTED_PATH}"
echo "GCF_CANDIDATES_PATH=${GCF_CANDIDATES_PATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "GED_THRESHOLDS=${GED_THRESHOLDS}"
echo "EMBEDDING_THRESHOLDS=${EMBEDDING_THRESHOLDS}"
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

echo "===== COMBINED REPORT ====="
cat "${OUTPUT_ROOT}/combined/combined_report.md"
echo "===== COMBINED THRESHOLD SUMMARY ====="
cat "${OUTPUT_ROOT}/combined/combined_threshold_summary.csv"
echo "===== FIGURES ====="
find "${OUTPUT_ROOT}/combined/figures" -type f | sort
