#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Paired CLI wrapper only: this repair runs CPU checkpoint inspection on
# AutoDL; do not submit this GPU template as a substitute for GPU2 authority.
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
# No inference/heuristic override: this entry only reads sealed checkpoints.
python -I -B scripts/autodl/diagnose_t14_causal_170.py --config configs/hpc.yaml "$@"
