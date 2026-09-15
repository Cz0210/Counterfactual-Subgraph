#!/bin/bash
# CPU-only saved-record audit and figures, explicitly no GPU.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "CPU finalizer python=$(command -v python)"
python -V
python -I -B scripts/finalize_ours_taste_focus.py --config configs/hpc.yaml "$@"
