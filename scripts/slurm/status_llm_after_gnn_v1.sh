#!/usr/bin/env bash
# CPU-only status, no inference/fallback/lease/matrix writes.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:",torch.cuda.is_available())'
exec python -I -B scripts/autodl/status_llm_after_gnn_v1.py --config configs/hpc.yaml "$@"
