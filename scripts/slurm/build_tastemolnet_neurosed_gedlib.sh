#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED fixed-budget work is AutoDL-only.
#SBATCH --job-name=taste-gedlib-build-refuse
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
echo "REFUSING_HPC_EXECUTION: Taste NeuroSED GEDLIB build is AutoDL-only." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/build_tastemolnet_neurosed_gedlib.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --greed-root /absolute/pinned/greed \
  --greed-expts-root /absolute/pinned/greed-expts \
  --gedlib-root /absolute/pinned/gedlib \
  --expected-gedlib-commit 120856f670e013f080b116c0be4cc6bd72fc935d \
  --pybind11-cmake-dir /absolute/pinned/pybind11/share/cmake/pybind11 \
  --output-root /root/autodl-tmp/envs/taste-neurosed-gedlib-hash
