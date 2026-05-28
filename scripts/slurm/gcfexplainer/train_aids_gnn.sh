#!/bin/bash
#SBATCH -J gcf_aids_gnn
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
OUT_DIR="${PROJECT_ROOT}/outputs/hpc/gcfexplainer/official_aids_gnn_train/${JOB_ID}"
OFFICIAL_DIR="${PROJECT_ROOT}/baselines/gcfexplainer_official"
GNN_LOG="${OUT_DIR}/gnn_train.log"
GNN_MODEL="${OFFICIAL_DIR}/data/aids/gnn/model_best.pth"
OFFICIAL_TRAIN_LOG="${OFFICIAL_DIR}/data/aids/gnn/log.txt"

mkdir -p logs
mkdir -p "${OUT_DIR}"

exec > >(tee -a "${GNN_LOG}") 2>&1

echo "===== GCFEXPLAINER OFFICIAL AIDS GNN TRAIN ENV CHECK ====="
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
echo "GNN_EPOCHS=${GNN_EPOCHS:-1000}"
echo "CUDA_ID=${CUDA_ID:-0}"
echo "OUT_DIR=${OUT_DIR}"
echo "OFFICIAL_DIR=${OFFICIAL_DIR}"
echo "=========================================================="

if [ ! -d "${OFFICIAL_DIR}" ]; then
  echo "[ERROR] official GCFExplainer directory not found: ${OFFICIAL_DIR}" >&2
  exit 1
fi
if [ ! -f "${OFFICIAL_DIR}/gnn.py" ]; then
  echo "[ERROR] required official script missing: ${OFFICIAL_DIR}/gnn.py" >&2
  exit 1
fi

cd "${OFFICIAL_DIR}"
echo "===== RUNNING: python gnn.py --dataset aids --epochs ${GNN_EPOCHS:-1000} --cuda ${CUDA_ID:-0} ====="
python gnn.py --dataset aids --epochs "${GNN_EPOCHS:-1000}" --cuda "${CUDA_ID:-0}"

echo "===== CHECKING TRAINED AIDS GNN ARTIFACTS ====="
if [ ! -f "${GNN_MODEL}" ]; then
  echo "[ERROR] expected trained GNN model not found: ${GNN_MODEL}" >&2
  exit 1
fi
ls -lh "${GNN_MODEL}"

if [ -f "${OFFICIAL_TRAIN_LOG}" ]; then
  echo "===== tail -n 80 ${OFFICIAL_TRAIN_LOG} ====="
  tail -n 80 "${OFFICIAL_TRAIN_LOG}"
else
  echo "[WARN] official GNN train log not found: ${OFFICIAL_TRAIN_LOG}"
fi

echo "===== GCFEXPLAINER AIDS GNN TRAIN DONE ====="
echo "gnn_train_log=${GNN_LOG}"
echo "gnn_model=${GNN_MODEL}"
