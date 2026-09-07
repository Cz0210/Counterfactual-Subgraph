#!/usr/bin/env bash
#SBATCH --job-name=reach-controls-plot
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Task-specific CPU-only rendering; no GPU/model/heuristic inference.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "Python: $(command -v python)"
python --version
echo 'CPU-only numerical display; no science recomputation'
python scripts/paper/render_bace_reach_controls.py --config configs/hpc.yaml "$@"
