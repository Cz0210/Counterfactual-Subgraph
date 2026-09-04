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

: "${MUT_SUCCESSOR_TEMPLATE:?absolute successor template required}"
: "${MUT_SUCCESSOR_BINDINGS:?absolute placeholder bindings required}"
: "${MUT_SUCCESSOR_SPEC:?fresh absolute successor spec required}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/build_mut_next_stage_executor_spec_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --template "$MUT_SUCCESSOR_TEMPLATE" \
  --bindings "$MUT_SUCCESSOR_BINDINGS" \
  --output "$MUT_SUCCESSOR_SPEC"
