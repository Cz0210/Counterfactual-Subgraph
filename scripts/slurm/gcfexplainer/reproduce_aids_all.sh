#!/bin/bash
#SBATCH -J gcf_aids_all
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT=/share/home/u20526/czx/counterfactual-subgraph
CONDA_SH=/share/home/u20526/anaconda3/etc/profile.d/conda.sh
cd "${PROJECT_ROOT}"

if [ ! -f "${CONDA_SH}" ]; then
  echo "[ERROR] conda profile script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV:-gcfexplainer_py38}"
set -u

export PYTHONPATH=$PWD

JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
OUT_DIR="${PROJECT_ROOT}/outputs/hpc/gcfexplainer/official_aids/${JOB_ID}"
OFFICIAL_DIR="${PROJECT_ROOT}/baselines/gcfexplainer_official"
GNN_MODEL="${OFFICIAL_DIR}/data/aids/gnn/model_best.pth"
VRRW_LOG="${OUT_DIR}/vrrw.log"
SUMMARY_LOG="${OUT_DIR}/summary.log"
SUMMARY_JSON="${OUT_DIR}/official_aids_summary.json"
SUMMARY_CSV="${OUT_DIR}/official_aids_summary.csv"

mkdir -p logs
mkdir -p "${OUT_DIR}"

echo "===== GCFEXPLAINER OFFICIAL AIDS ALL ENV CHECK ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "PYTHONPATH=${PYTHONPATH}"
python --version
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "CONDA_ENV override source: ${CONDA_ENV:-gcfexplainer_py38}"
echo "OUT_DIR=${OUT_DIR}"
echo "OFFICIAL_DIR=${OFFICIAL_DIR}"
echo "===================================================="

if [ ! -d "${OFFICIAL_DIR}" ]; then
  echo "[ERROR] official GCFExplainer directory not found: ${OFFICIAL_DIR}" >&2
  exit 1
fi
if [ ! -f "${OFFICIAL_DIR}/vrrw.py" ]; then
  echo "[ERROR] required official script missing: ${OFFICIAL_DIR}/vrrw.py" >&2
  exit 1
fi
if [ ! -f "${OFFICIAL_DIR}/summary.py" ]; then
  echo "[ERROR] required official script missing: ${OFFICIAL_DIR}/summary.py" >&2
  exit 1
fi
if [ ! -f "${GNN_MODEL}" ]; then
  echo "[ERROR] required official AIDS GNN model missing: ${GNN_MODEL}" >&2
  echo "[ERROR] Train it first with:" >&2
  echo "  sbatch scripts/slurm/gcfexplainer/train_aids_gnn.sh" >&2
  exit 1
fi

cd "${OFFICIAL_DIR}"

echo "===== RUNNING: python vrrw.py --dataset aids ====="
python vrrw.py --dataset aids 2>&1 | tee "${VRRW_LOG}"

echo "===== RUNNING: python summary.py --dataset aids ====="
python summary.py --dataset aids 2>&1 | tee "${SUMMARY_LOG}"

echo "===== COLLECTING SUMMARY METRICS ====="
cd "${PROJECT_ROOT}"
python scripts/eval/collect_gcf_official_results.py \
  --summary-log "${SUMMARY_LOG}" \
  --out-json "${SUMMARY_JSON}" \
  --out-csv "${SUMMARY_CSV}"

echo "===== SYNCING OFFICIAL RESULTS ====="
if [ -d "${OFFICIAL_DIR}/results/aids" ]; then
  mkdir -p "${OUT_DIR}/results"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${OFFICIAL_DIR}/results/aids/" "${OUT_DIR}/results/aids/"
  else
    mkdir -p "${OUT_DIR}/results/aids"
    cp -a "${OFFICIAL_DIR}/results/aids/." "${OUT_DIR}/results/aids/"
  fi
  find "${OUT_DIR}/results/aids" -maxdepth 3 -type f | sort
else
  echo "[WARN] official results directory not found after all-in-one run: ${OFFICIAL_DIR}/results/aids"
fi

echo "===== GCFEXPLAINER AIDS ALL DONE ====="
echo "vrrw_log=${VRRW_LOG}"
echo "summary_log=${SUMMARY_LOG}"
echo "summary_json=${SUMMARY_JSON}"
echo "summary_csv=${SUMMARY_CSV}"
