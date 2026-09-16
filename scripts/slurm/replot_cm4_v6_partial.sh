#!/bin/bash
# User V6 overrides repository GPU defaults: offline CPU rendering only.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM4_V6_CODE:?immutable code root}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "CPU-only replot python=$(command -v python)"
python -V
python -c 'import torch; print("CUDA available:",torch.cuda.is_available())'
python -I -B scripts/replot_cm4_v6_partial.py --out-dir "$CM4_V6_RELEASE" --replot-only
