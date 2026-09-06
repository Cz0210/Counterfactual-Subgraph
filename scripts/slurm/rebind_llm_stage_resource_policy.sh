#!/usr/bin/env bash
# Explicit user-authorized exception to the repository GPU defaults:
# metadata-only seal/validate; no model, allocation lease or science launch.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "CPU-only dispatch metadata: python=$(command -v python); GPU science disabled"
python --version
exec python -I -B scripts/autodl/rebind_llm_stage_resource_policy.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
