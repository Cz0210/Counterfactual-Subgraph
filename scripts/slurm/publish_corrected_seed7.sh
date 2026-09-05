#!/usr/bin/env bash
# Task-specific override: publication copies accepted small tables, CPU only.
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
echo "python=$(command -v python)"
python --version
echo 'CPU-only publication; no CUDA science'
exec python -I -B scripts/ablations/gnn/publish_corrected_seed7.py --config configs/hpc.yaml "$@"
