#!/usr/bin/env bash
#SBATCH --job-name=t13-seal-autodl-only
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
python -I -B scripts/autodl/seal_t13_deterministic_formal.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --help
echo 'REFUSING_HPC_EXECUTION: T13 formal sealing must run against AutoDL authority.' >&2
exit 78
