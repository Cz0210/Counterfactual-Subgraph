#!/usr/bin/env bash
# Static CLI-parity wrapper only. The active release supervisor is an AutoDL
# CPU-only sidecar; submitting this mandatory GPU-shaped wrapper is forbidden.
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
echo "AutoDL CPU-only release sidecar; do not submit this Slurm wrapper." >&2
exit 78

# Unreachable documentation-only CLI parity command:
python scripts/autodl/run_three_dataset_release_supervisor.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  run \
  --spec "${THREE_DATASET_RELEASE_SPEC:?}"
