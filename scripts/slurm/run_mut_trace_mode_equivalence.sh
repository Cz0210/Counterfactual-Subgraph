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
python scripts/autodl/run_mut_trace_mode_equivalence.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --help
exit 2
