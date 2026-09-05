#!/usr/bin/env bash
# CPU-only metadata preflight: explicit task exception to GPU script baseline.
# No model load, generation, SQLite read, matrix append, or GPU lease.
#SBATCH --job-name=mut_route_b_preflight
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "python=$(command -v python)"
python --version
echo "cpu_only_preflight=true; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec python scripts/autodl/preflight_mut_route_b_closeout_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false "$@"
