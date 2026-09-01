#!/usr/bin/env bash
#SBATCH --job-name=taste_t8_dual_recovery
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export T8_REPO_ROOT=$PWD
export T8_DUAL_GPU_INDEX=1
export RUN_GNN_ABLATION=0

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("cuda_devices=", torch.cuda.device_count())'

# This dataset-specific relay is valid only when Slurm exposes physical GPU1
# on a host with the pinned AutoDL runtime mounted.  Refuse other allocations
# instead of reaching outside the assigned device.
[[ "${CUDA_VISIBLE_DEVICES:-}" == "1" ]] || { echo "T8 dual recovery requires physical GPU1" >&2; exit 64; }
: "${AUTODL_RUNTIME_ROOT:?set the mounted AutoDL runtime root}"
: "${AUTODL_CONTROL_ROOT:?set the mounted AutoDL control root}"
: "${T8_DUAL_CONTROLLER_ROOT:?set one fresh controller root}"

bash scripts/autodl/run_tastemolnet_t8_dual_branch_recovery_v1.sh
