#!/bin/bash
# Offline publication normally runs on Mac; this paired entry is CPU-only.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd "${CM4_V6_CODE:?immutable code root}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "Offline publication: $(command -v python)"
python -V
python -c 'import torch;print("CUDA available:",torch.cuda.is_available())'
# No scientific inference/config options: only already accepted CSV/compact raw.
python -I -B scripts/export_cm4_v6_release.py "$@"
