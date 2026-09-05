#!/usr/bin/env bash
# CPU-only reconciliation; the A800 allocation is retained for project policy compatibility.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'
python scripts/autodl/reconcile_t12_diagnostic_510_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
