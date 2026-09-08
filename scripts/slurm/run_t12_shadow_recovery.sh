#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=t12-shadow-interface

# Repository submission baseline only; this is not an approved T12 science job.
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "Python: $(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
python -I -B scripts/autodl/run_t12_shadow_recovery.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
