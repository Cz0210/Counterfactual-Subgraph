#!/bin/bash
# User-authorized CPU asset admission; no GPU or original oracle inference.
#SBATCH --partition=intel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=cm-crem-db
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
: "${CM_EXECUTION_ROOT:?}" "${CM_GENERATOR_PYTHON:?}" "${CM_SPEC:?}"
: "${CM_ASSET_ROOT:?}" "${CM_ARCHIVE:?}" "${CM_RUN_ROOT:?}"
cd "$CM_EXECUTION_ROOT"
export PYTHONPATH=$PWD
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "CM static asset host=$(hostname) job=$SLURM_JOB_ID python=$CM_GENERATOR_PYTHON CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$CM_GENERATOR_PYTHON" -V
unset PYTHONPATH
"$CM_GENERATOR_PYTHON" -s -B scripts/prepare_cm_crem_database.py --config configs/hpc.yaml --spec "$CM_SPEC" --archive "$CM_ARCHIVE" --output-root "$CM_ASSET_ROOT" --run-root "$CM_RUN_ROOT"
