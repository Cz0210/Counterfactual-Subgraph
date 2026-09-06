#!/bin/bash
#SBATCH --job-name=mut-trace-mode-gate
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
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
echo "AutoDL-only guarded trace-mode gate; do not submit this Slurm wrapper."
# A CPU-only plan is also available directly on the control host:
# python scripts/autodl/run_mut_trace_mode_equivalence.py --config configs/hpc.yaml \
#   plan-recovery --arm-root /absolute/failed/trace_on --trace-mode on
# `run-one --phase continuous --resume` requires a real jointly committed
# boundary from this repaired driver; it cannot adopt the legacy missing250 row.
# This wrapper intentionally remains non-submitting/non-scientific.
python scripts/autodl/run_mut_trace_mode_equivalence.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --help
exit 2
