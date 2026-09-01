#!/usr/bin/env bash
# CPU-only terminal/publication reconciliation; no scientific source is written.
#SBATCH --job-name=aids_terminal_reconcile
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
export CUDA_VISIBLE_DEVICES=""
export RUN_GNN_ABLATION=0
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "route=aids_completed_science_terminal_publication_reconciliation"
echo "source_science_mutated=false"
exec python scripts/autodl/reconcile_aids_comrecgc_terminal_publication.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
