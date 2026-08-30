#!/usr/bin/env bash
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
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'

: "${BACE_COMRECGC_CONVERGENCE_HOOK_ROOT:?required}"
: "${BACE_COMRECGC_RESOURCE_CAP_STATE_ROOT:?required}"
: "${BACE_COMRECGC_SCIENCE_PID:?required}"
: "${BACE_COMRECGC_SCIENCE_START_TICKS:?required}"

python scripts/autodl/run_bace_comrecgc_resource_cap_observer.py \
  --config configs/hpc.yaml \
  --convergence-hook-root "$BACE_COMRECGC_CONVERGENCE_HOOK_ROOT" \
  --state-root "$BACE_COMRECGC_RESOURCE_CAP_STATE_ROOT" \
  --science-pid "$BACE_COMRECGC_SCIENCE_PID" \
  --science-start-ticks "$BACE_COMRECGC_SCIENCE_START_TICKS" \
  --poll-seconds "${SCHEDULER_POLL_SECONDS:-60}"
