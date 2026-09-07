#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -I -B scripts/benchmarks/benchmark_t13_bridge_materialization.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --device cuda:0 --output "outputs/hpc/audits/t13-bridge-benchmark-${SLURM_JOB_ID}.json"
