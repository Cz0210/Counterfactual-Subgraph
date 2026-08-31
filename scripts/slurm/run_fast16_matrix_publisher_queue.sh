#!/usr/bin/env bash
# Persistent CPU-only publisher queue; it owns no science task or GPU memory.
#SBATCH --job-name=fast16_matrix_publish
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
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
exec python scripts/autodl/run_fast16_matrix_publisher_queue.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
