#!/usr/bin/env bash
#SBATCH --job-name=main-table-continuation
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
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

echo "AUTODL_ONLY: this sidecar observes exact AutoDL PIDs and physical GPU locks" >&2
exit 64

# Documentation-only CLI parity (unreachable by design):
# python scripts/autodl/run_main_table_continuation_sidecar.py \
#   --config configs/hpc.yaml run --spec "$AUTODL_CONTINUATION_SPEC"
