#!/usr/bin/env bash
#SBATCH --job-name=taste-t2-adoption-v2-autodl-only
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
echo "TASTE_T2_ADOPTION_V2_AUTODL_ONLY: HPC execution is disabled" >&2
exit 64

# Documentation-only CLI parity (unreachable by design):
# python -I -B scripts/autodl/adopt_tastemolnet_gine_pass_v2.py \
#   --config configs/hpc.yaml \
#   --control-root /autodl-fs/data/counterfactual-subgraph-runtime/control \
#   --artifact-root "$TASTE_T2_ARTIFACT_ROOT" \
#   --controller-root "$TASTE_T2_CONTROLLER_ROOT" \
#   --training-state-root "$TASTE_T2_TRAINING_STATE_ROOT" \
#   --source-run-id "$TASTE_T2_SOURCE_RUN_ID" \
#   --source-controller-id "$TASTE_T2_SOURCE_CONTROLLER_ID"
