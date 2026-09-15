#!/bin/bash
# Explicit 2026-09-15 CPU-only authorization overrides generic GPU template.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=t14-saved-cpu
#SBATCH --output=/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/taste-cpu-closeout-20260915/logs/%j.out
#SBATCH --error=/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/taste-cpu-closeout-20260915/logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${TASTE_CODE_ROOT:?immutable code root required}"
export PYTHONPATH=$PWD
echo "Python: $(command -v python)"
python --version
python -I -B scripts/audit_t14_saved_followers.py --config configs/hpc.yaml \
 --set inference.fallback_to_heuristic=false --device cpu "$@"
