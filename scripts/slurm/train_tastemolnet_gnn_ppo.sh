#!/usr/bin/env bash
# Static CLI parity only. TasteMolNet policy-v2 science is AutoDL-only.
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
echo "TasteMolNet Ours PPO is authorized only through the reviewed AutoDL controller." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python scripts/train_tastemolnet_gnn_ppo.py \
  --config configs/hpc.yaml \
  --stage T6_OURS_SMOKE \
  --model-path /absolute/private/generic-chemllm-base \
  --dataset-path /absolute/private/tastemolnet/train.csv \
  --output-dir /absolute/fresh/private/t6-ours-smoke \
  --gnn-checkpoint /absolute/private/tastemolnet/gine/calibrated \
  --t5-output /absolute/private/tastemolnet/clean-policy-initializer \
  --downstream-policy configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json \
  --base-policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --gnn-device cuda \
  --updates 5 \
  --parent-count 16 \
  --batch-size 2 \
  --seed 7 \
  --set inference.fallback_to_heuristic=false
