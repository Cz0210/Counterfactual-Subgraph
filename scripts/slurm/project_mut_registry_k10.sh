#!/usr/bin/env bash
# CPU-only aggregate projection; A800 directives preserve the repository Slurm contract.
#SBATCH --job-name=mut_registry_k10_projection
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
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
echo "Requires --terminal-root, --reference-standardized-root, --startup-repair-receipt and fresh --output-root (all absolute paths)."
echo "Re-exports frozen K10 aggregates; no inference, OT, selector or test-dataset rerun."
exec python scripts/autodl/project_mut_registry_k10.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
