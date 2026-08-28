#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED fixed-budget work is AutoDL-only.
#SBATCH --job-name=taste-neurosed-pairs-refuse
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
echo "REFUSING_HPC_EXECUTION: Taste NeuroSED fixed-budget work is AutoDL-only." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/build_neurosed_fixed_budget_pairs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --split-csv /absolute/private/splits/train.csv \
  --expected-split train \
  --expected-split-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --feature-schema-json /absolute/private/feature_schema.json \
  --expected-feature-schema-sha256 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --output-dir /absolute/fresh/pairs \
  --pair-count 1600 \
  --seed 7 \
  --n-hops-query 5 \
  --traversal-probability-query 0.5 \
  --write-disjoint-benchmark-cohorts
