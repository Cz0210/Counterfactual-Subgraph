#!/bin/bash
# CPU-only preparation exception: no GPU, training, mining or test evaluation.
# Actual AutoDL recovery uses the same Python CLI in its authorized NVMe workspace.
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
cd "${T13_PREP_CODE:?immutable preparation worktree}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
echo "CPU-only T13 preparation python=$(command -v python)"
python -V
python -I -B scripts/prepare_t13_indexrebuilt.py --config configs/hpc.yaml --plan "${T13_PREP_PLAN:?bound plan}"
