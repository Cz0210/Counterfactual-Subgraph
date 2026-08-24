#!/usr/bin/env bash
# Static CLI parity only.  The active campaign is AutoDL-only and this file is
# intentionally not submitted by the continuation agent.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
: "${ROOT_CAUSE_SPEC:?ROOT_CAUSE_SPEC is required}"
: "${ROOT_CAUSE_CONTROL:?ROOT_CAUSE_CONTROL is required}"
python scripts/autodl/run_root_cause_acceleration_controller.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --spec "$ROOT_CAUSE_SPEC" \
  --control-root "$ROOT_CAUSE_CONTROL" \
  --poll-seconds 60
