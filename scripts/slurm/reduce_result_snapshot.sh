#!/usr/bin/env bash
# CPU-only offline artifact reduction: explicit exception to the A800 template.
# No science, source collection, authority writes or GPU reservation.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
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
echo "CUDA unavailable by task policy: CPU-only; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
: "${SNAPSHOT_ROOT:?read-only local snapshot root required}"
: "${OUTPUT_ROOT:?fresh external artifact directory required}"
exec nice -n 10 python scripts/paper/reduce_result_snapshot.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --snapshot-root "$SNAPSHOT_ROOT" --output-root "$OUTPUT_ROOT"
