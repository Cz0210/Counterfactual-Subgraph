#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED fixed-budget work is AutoDL-only.
#SBATCH --job-name=taste-non-mip-verify-refuse
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
echo "REFUSING_HPC_EXECUTION: Taste non-MIP GEDLIB verification is AutoDL-only." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/verify_tastemolnet_neurosed_non_mip_gedlib.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --selection-manifest /absolute/non_mip_gedlib_selection.json \
  --receipt /absolute/fresh/non_mip_gedlib_verifier_receipt.json
