#!/bin/bash
#SBATCH --job-name=clear
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baselines/clear/slurm_%x_%j.out
#SBATCH --error=logs/baselines/clear/slurm_%x_%j.err

set -euo pipefail

DATASET="${1:-}"
STAGE="${2:-}"
CLEAR_CONDA_ENV="${CLEAR_CONDA_ENV:-smiles_pip118}"

if [ -z "${DATASET}" ] || [ -z "${STAGE}" ]; then
  echo "Usage: sbatch scripts/baselines/clear/slurm_clear.sbatch <dataset> <stage>" >&2
  exit 2
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/baselines/clear/run_clear.sh" ]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${SCRIPT_DIR}/../../.."
fi

mkdir -p logs/baselines/clear

echo "===== CLEAR SLURM ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "dataset=${DATASET}"
echo "stage=${STAGE}"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unknown}"
echo "clear_conda_env=${CLEAR_CONDA_ENV}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================="

set +u
source ~/.bashrc
conda activate "${CLEAR_CONDA_ENV}"
set -u

echo "===== CLEAR RUNTIME ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "dataset: ${DATASET}"
echo "stage: ${STAGE}"
echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
echo "slurm_partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "slurm_job_gpus: ${SLURM_JOB_GPUS:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unknown}"
echo "which python: $(which python)"
echo "python version: $(python --version)"
python - <<'PY'
import importlib.util

try:
    import torch
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))

if importlib.util.find_spec("torch_geometric") is not None:
    try:
        import torch_geometric
        print("torch_geometric version:", getattr(torch_geometric, "__version__", "unknown"))
    except Exception as exc:
        print("torch_geometric import failed:", repr(exc))
else:
    print("torch_geometric importable: False")

print("rdkit importable:", importlib.util.find_spec("rdkit") is not None)
PY
echo "========================="

bash scripts/baselines/clear/run_clear.sh "${DATASET}" "${STAGE}"
