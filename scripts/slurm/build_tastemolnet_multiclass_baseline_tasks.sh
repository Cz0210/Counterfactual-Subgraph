#!/usr/bin/env bash
# Static CLI-parity wrapper. The active TasteMolNet campaign is AutoDL-only and
# this script is never submitted by the continuation controller.
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
exec python scripts/autodl/build_tastemolnet_multiclass_baseline_tasks.py \
  --config configs/hpc.yaml "$@"
