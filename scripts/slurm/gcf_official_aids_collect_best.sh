#!/bin/bash
#SBATCH -J gcf_off_best
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

echo "===== GCF OFFICIAL COLLECT BEST ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version

python scripts/collect_gcf_official_runs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --runs-root "${RUNS_ROOT:-outputs/hpc/gcfexplainer_official}" \
  --out-dir "${OUT_DIR:-outputs/hpc/gcfexplainer_official/final}" \
  --select-k "${GCF_SELECT_K:-10}" \
  --select-theta "${GCF_SELECT_THETA:-0.1}"

echo "===== GCF OFFICIAL COLLECT BEST DONE ====="
cat "${OUT_DIR:-outputs/hpc/gcfexplainer_official/final}/gcf_official_best_summary.json"

