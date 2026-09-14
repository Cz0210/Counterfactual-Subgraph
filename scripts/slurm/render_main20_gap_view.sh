#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Paired CLI entry retained for repository convention; this task is offline Mac
# rendering, so DO NOT submit a GPU job to redraw saved CSVs.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python -V
echo "Offline rendering only: use the delivered run_replot.sh on Mac."
exit 2
