#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=bace-ablation-reference

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${BACE_FINAL_ROOT:?}"
: "${MATRIX_AUTHORITY_STATE:?}"
: "${BACE_GINE_ROOT:?}"
: "${BACE_PPO_ROOT:?}"
: "${BACE_PARENT_PREP_MANIFEST:?}"
: "${BACE_BASE_POOL_MANIFEST:?}"
: "${BACE_HIGHTEMP_POOL_MANIFEST:?}"
: "${BACE_MERGED_POOL_MANIFEST:?}"
: "${BACE_VERIFICATION_MANIFEST:?}"
: "${BACE_SELECTOR_MANIFEST:?}"
: "${MOLCLR_CHECKPOINT:?}"
: "${ABLATION_REFERENCE_OUTPUT:?}"

nice -n 10 python scripts/ablations/build_bace_ours_main_reference_contract.py \
  --config configs/hpc.yaml \
  --matrix-authority-state "$MATRIX_AUTHORITY_STATE" \
  --final-root "$BACE_FINAL_ROOT" \
  --oracle-root "$BACE_GINE_ROOT" \
  --ppo-root "$BACE_PPO_ROOT" \
  --train-parent-prep-manifest "$BACE_PARENT_PREP_MANIFEST" \
  --base-pool-manifest "$BACE_BASE_POOL_MANIFEST" \
  --high-temperature-pool-manifest "$BACE_HIGHTEMP_POOL_MANIFEST" \
  --merged-pool-manifest "$BACE_MERGED_POOL_MANIFEST" \
  --verification-manifest "$BACE_VERIFICATION_MANIFEST" \
  --selector-manifest "$BACE_SELECTOR_MANIFEST" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --output "$ABLATION_REFERENCE_OUTPUT"
