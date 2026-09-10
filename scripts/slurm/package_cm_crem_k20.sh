#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# User-authorized record-only CPU packaging exception to generic GPU template.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_PACKAGE_CODE:?immutable package code required}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
echo "K20 package job=$SLURM_JOB_ID host=$(hostname) python=$(command -v python)"
python -V
python -I -B scripts/package_cm_crem_k20.py --config configs/hpc.yaml --root "${CM_PACKAGE_ROOT:?accepted K20 root required}"
