#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=gps-param-dry-run

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/ablations/gnn/dry_run_graphgps_parameter_match.py \
  --config configs/hpc.yaml \
  --verify-runtime \
  --output "${GRAPHGPS_PARAMETER_MATCH_OUTPUT:?set output receipt path}"
