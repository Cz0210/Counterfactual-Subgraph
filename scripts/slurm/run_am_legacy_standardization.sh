#!/bin/bash
#SBATCH --job-name=am-legacy-standardization
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
echo "cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())')"

# This static parity wrapper also accepts `reexport-mut-ours-matched`. That
# action aggregates only checksum-closed pair artifacts; it performs no RF,
# MolCLR, selector, or candidate inference. The AutoDL campaign does not submit
# this wrapper, and the inference fallback setting is intentionally inapplicable.
python scripts/autodl/run_am_legacy_standardization.py \
  --config configs/hpc.yaml \
  "$@"
