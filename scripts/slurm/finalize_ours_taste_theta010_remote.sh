#!/bin/bash
# Mac receiver, only --help on HPC; never a persistent service or GPU job.
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
echo "CPU-only receiver CLI python=$(command -v python)"
python -V
python -I -B scripts/finalize_ours_taste_theta010_remote.py --help
