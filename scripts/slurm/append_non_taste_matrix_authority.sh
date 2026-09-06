#!/usr/bin/env bash
# CPU-only publication; A800 directives preserve repository Slurm CLI parity.
# Mut accepts either the legacy exact-postprocess terminal or the independent
# parity-v2 standardization terminal; both are reopened fail-closed by the CLI.
#SBATCH --job-name=append_non_taste_matrix
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
echo "Optional --startup-repair-receipt /absolute/receipt.json binds typed Mut startup repair; strict terminal checks remain required."
echo "Optional --registry-projection /absolute/fresh-projection-root uses a verified K10 registry export and requires --startup-repair-receipt."
exec python scripts/autodl/append_non_taste_matrix_authority.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
