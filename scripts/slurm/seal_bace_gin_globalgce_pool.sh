#!/bin/bash
#SBATCH --job-name=bace-gin-global-pool
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Metadata-only task-specific override: no GPU, no model inference, no heuristic.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
echo 'CUDA not requested; metadata-only original80 evidence sealing'
python -I -B scripts/seal_bace_gin_globalgce_pool.py --config configs/hpc.yaml "$@"
