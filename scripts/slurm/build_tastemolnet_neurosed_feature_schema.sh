#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED artifacts are AutoDL-only.
#SBATCH --job-name=taste-neurosed-schema-refuse
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
echo "REFUSING_HPC_EXECUTION: Taste NeuroSED feature-schema production is AutoDL-only." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/build_tastemolnet_neurosed_feature_schema.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --train-csv /absolute/private/tastemolnet/splits/train.csv \
  --expected-train-sha256 eac05f7003c37a24554aa2c22e1051edb90eb4a12f9b62ae6fd47d73efa59564 \
  --validation-csv /absolute/private/tastemolnet/splits/validation.csv \
  --expected-validation-sha256 eedb06c6997652113f234f085135acd4f6dafb10f0d5d4d8e3f432473712a016 \
  --output-json /absolute/fresh/tastemolnet-neurosed/feature_schema.json
