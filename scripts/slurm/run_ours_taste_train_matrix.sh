#!/bin/bash
# CPU metadata validation only. This AutoDL-GPU entrypoint may NOT run on HPC.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "CPU-only CLI validation python=$(command -v python)"
python -V
# The actual GPU worker is AutoDL-only. No HPC scientific GPU fallback.
python -I -B scripts/run_ours_taste_train_matrix.py --help
