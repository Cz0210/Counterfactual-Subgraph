#!/usr/bin/env bash
#SBATCH --job-name=bace-gnn-reach-pair
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Submit only through the existing <=2-heavy-job campaign; no GPU request.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Python: $(command -v python)"
python --version
echo "CPU-only new-pool evaluation; no training or temperature fitting"
# No heuristic inference fallback exists in the frozen scientific evaluator.
python -I -B scripts/hpc/gnn/run_bace_gnn_reach_v2_chunk.py --config configs/hpc.yaml "$@"
