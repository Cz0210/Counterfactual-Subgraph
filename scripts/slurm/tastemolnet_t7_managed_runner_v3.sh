#!/usr/bin/env bash
#SBATCH --job-name=taste-t7-managed
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

echo "TASTE_T7_MANAGED_AUTODL_ONLY: physical GPU0 UUID lock is required" >&2
exit 64

# Documentation-only CLI parity (unreachable by design):
# python scripts/autodl/tastemolnet_t7_managed_runner_v3.py \
#   --mode run --config configs/hpc.yaml \
#   --release-root "$TASTEMOLNET_T7_RELEASE_ROOT" \
#   --stage-root "$TASTEMOLNET_T7_STAGE_ROOT" \
#   --final-path "$TASTEMOLNET_T7_OUTPUT" \
#   --run-id "$TASTEMOLNET_T7_RUN_ID" \
#   --gpu-uuid "$TASTEMOLNET_GPU0_UUID" \
#   --set inference.fallback_to_heuristic=false
