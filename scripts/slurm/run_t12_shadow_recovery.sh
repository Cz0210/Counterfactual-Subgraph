#!/bin/bash
# 2026-09-15 authorization: HPC CPU only; actual restore is AutoDL-owner only.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=t12-shadow-interface

# Interface/test baseline. shadow-segment is AutoDL-canonical-owner only:
# an ordinary sbatch job cannot invent its inherited owner FD/resource binding.
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
case "${1:-}" in
  status|restore500-preflight|compare|plan) ;;
  *) echo "CPU interface only: no GPU restoration/promotion through sbatch" >&2; exit 2 ;;
esac
echo "Python: $(command -v python)"
python --version
python -I -B scripts/autodl/run_t12_shadow_recovery.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
