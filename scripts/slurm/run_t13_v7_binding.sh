#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# V7 explicitly forbids HPC GPU and scientific T13 CPU fallback.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
echo "CPU-only T13 binding status; no science on HPC"
command -v python
python --version
python -I -B scripts/run_t13_v7_binding.py --config configs/hpc.yaml --action status "$@"
