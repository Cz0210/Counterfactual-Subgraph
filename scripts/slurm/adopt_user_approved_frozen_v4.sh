#!/usr/bin/env bash
# Static CLI-parity wrapper only. The authorized campaign is AutoDL-only;
# do not submit this file. Its presence keeps the repository CLI contract paired.
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
echo "This wrapper is static CLI parity only; the formal adoption is AutoDL-only." >&2
exit 78

# Unreachable command retained for exact CLI synchronization; never submitted.
python scripts/autodl/adopt_user_approved_frozen_v4.py \
  --config configs/hpc.yaml \
  --source-root "${FROZEN_V4_SOURCE_ROOT:?set FROZEN_V4_SOURCE_ROOT}" \
  --runtime-root "${AUTODL_RUNTIME_ROOT:?set AUTODL_RUNTIME_ROOT}" \
  --output-root "${FROZEN_V4_OUTPUT_ROOT:?set FROZEN_V4_OUTPUT_ROOT}"
