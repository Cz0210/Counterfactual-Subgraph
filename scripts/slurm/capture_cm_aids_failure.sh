#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Authorized CPU-only diagnostic, not default GPU training.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_DATASET_CODE:?}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
echo "CM failed-parent capture job=$SLURM_JOB_ID host=$(hostname)"
"${CM_GENERATOR_PYTHON:?}" -V
"$CM_GENERATOR_PYTHON" -s -B scripts/capture_cm_aids_failure.py --config configs/hpc.yaml --source-root "${CM_FAILURE_SOURCE:?}" --output-root "${CM_FAILURE_OUTPUT:?}" --parent-id AIDS_CM_train_02e48091bf6db2ef
