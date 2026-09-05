#!/usr/bin/env bash
# AutoDL-only: preserve the cluster wrapper contract but never submit Taste data here.
#SBATCH --job-name=t13-indexed-canary-autodl-only
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python -I -B scripts/autodl/canary_t13_indexed_dataset.py --config configs/hpc.yaml --help >/dev/null
echo "REFUSING_HPC_EXECUTION: T13 train/GINE canary is AutoDL-only." >&2
exit 78
