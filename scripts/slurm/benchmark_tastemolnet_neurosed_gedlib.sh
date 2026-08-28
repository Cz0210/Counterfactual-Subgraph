#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED fixed-budget work is AutoDL-only.
#SBATCH --job-name=taste-gedlib-bench-refuse
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
echo "REFUSING_HPC_EXECUTION: Taste NeuroSED GEDLIB benchmark is AutoDL-only." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/benchmark_tastemolnet_neurosed_gedlib.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --build-manifest /absolute/build-manifest.json \
  --pair-sampler-manifest /absolute/pair_sampler_manifest.json \
  --pairs-jsonl /absolute/benchmark_pairs_100.jsonl \
  --graph-inventory-jsonl /absolute/graph_inventory.jsonl \
  --benchmark-budget 100 \
  --workers 1 \
  --hard-wall-seconds 600 \
  --bace-legacy-throughput-drop-percent 0 \
  --aids-exact-throughput-drop-percent 0 \
  --host-load-gate-pass \
  --iowait-gate-pass \
  --output-dir /absolute/fresh/benchmark
