#!/usr/bin/env bash
# Static CLI parity only. The scoped TasteMolNet campaign is AutoDL-only.
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
echo "TasteMolNet research execution is authorized only on the reviewed AutoDL route; HPC remains forbidden." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python scripts/train_molecular_gnn.py \
  --config configs/hpc.yaml \
  --config configs/gnn/gine.yaml \
  --config configs/autodl/tastemolnet_gine_research_v1.yaml \
  --dataset tastemolnet \
  --data-dir /absolute/existing/private/splits \
  --output-dir /absolute/fresh/output \
  --profile full \
  --device cuda:0 \
  --graph-cache-root /absolute/existing/private/cache \
  --taste-policy-file configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --taste-policy-sha256 9a1bc033a0abd300e17bb79eb5f01a98accd790fc086d0f8119f289376e0d983 \
  --taste-policy-receipt /absolute/fresh/policy-audit/tastemolnet_policy_receipt.json \
  --taste-prepared-root /absolute/existing/private/prepared \
  --training-state-dir /absolute/fresh/private/training-state
