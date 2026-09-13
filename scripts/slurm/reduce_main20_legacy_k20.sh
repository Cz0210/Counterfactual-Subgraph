#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Task-authorized CPU record reduction, no oracle/inference fallback.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_K20_CODE:?}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "K20 raw reduction job=$SLURM_JOB_ID python=$(command -v python)"
python -V
python -I -B scripts/reduce_main20_legacy_k20.py --config configs/hpc.yaml --out-dir "${CM_K20_OUTPUT:?}"
