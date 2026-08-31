#!/usr/bin/env bash
#SBATCH --job-name=fast-16of16-v2
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
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
echo "AutoDL process adoption only; Slurm launch is disabled" >&2
exit 64
# python scripts/autodl/run_fast_16of16_v2.py --config configs/hpc.yaml --spec "$FAST_16OF16_V2_SPEC"
