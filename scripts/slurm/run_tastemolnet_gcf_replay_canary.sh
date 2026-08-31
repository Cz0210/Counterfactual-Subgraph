#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${T12_UNINTERRUPTED_OBSERVATION:?set the real-GPU uninterrupted observation}"
: "${T12_RESUMED_OBSERVATION:?set the new-process resumed observation}"
: "${T12_CANARY_GATE:?set one fresh canary gate path}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'

python scripts/run_tastemolnet_gcf_replay_canary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --uninterrupted "$T12_UNINTERRUPTED_OBSERVATION" \
  --cross-process-resumed "$T12_RESUMED_OBSERVATION" \
  --output "$T12_CANARY_GATE"
