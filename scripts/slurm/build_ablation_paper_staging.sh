#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ablation-paper-staging

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
: "${ABLATION_PAPER_STAGING_ROOT:?}"
nice -n 10 python scripts/ablations/build_ablation_paper_staging.py \
  --config configs/hpc.yaml \
  --output-root "$ABLATION_PAPER_STAGING_ROOT"
