#!/bin/bash
#SBATCH --job-name=gpu_lane_release
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
cd "${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
export PYTHONPATH=$PWD
mkdir -p logs
echo "hostname=$(hostname) python=$(which python) commit=$(git rev-parse HEAD)"
python --version
echo '[GPU_LANE_RELEASE_BARRIER_PASS]'
