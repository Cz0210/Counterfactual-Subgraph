#!/bin/bash
# Record-only rendering: user CPU contract overrides the generic A800 template.
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
export CUDA_VISIBLE_DEVICES=""
echo "Record-only renderer; no CUDA/model/OT, Python=$(command -v python)"
python -V
python -I -B scripts/replot_cm_crem_comparison.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
