#!/usr/bin/env bash
# Paired CLI only: allocation alone is not authorization or an existing lease.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:",torch.cuda.is_available())'
exec python -I -B scripts/ablations/llm/run_bace_llm_successor.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
