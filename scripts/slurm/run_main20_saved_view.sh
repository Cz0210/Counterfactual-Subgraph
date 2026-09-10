#!/bin/bash
# Offline CPU-only export exception: no models or GPU are loaded.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "Offline source-only replot: $(command -v python)"
python -V
python -I -B scripts/run_main20_saved_view.py --config configs/hpc.yaml "$@"
