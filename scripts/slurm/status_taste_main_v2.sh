#!/usr/bin/env bash
#SBATCH --job-name=taste-main-v2-status
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
echo "Taste main-v2 controller status is AutoDL-only; Slurm is refused." >&2
exit 64

python -I -B scripts/autodl/status_taste_main_v2.py --config configs/hpc.yaml --controller-root /absolute/controller
