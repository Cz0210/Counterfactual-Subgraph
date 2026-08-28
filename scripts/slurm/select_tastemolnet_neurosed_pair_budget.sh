#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED fixed-budget work is AutoDL-only.
#SBATCH --job-name=taste-pair-budget-refuse
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
echo "REFUSING_HPC_EXECUTION: Taste NeuroSED pair planning is AutoDL-only." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/select_tastemolnet_neurosed_pair_budget.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --benchmark-100 /absolute/gedlib_benchmark_100.json \
  --benchmark-500 /absolute/gedlib_benchmark_500.json \
  --benchmark-1000 /absolute/gedlib_benchmark_1000.json \
  --selected-workers 1 \
  --disk-reservation-pass \
  --cpu-contention-gate-pass \
  --output-dir /absolute/fresh/plan
