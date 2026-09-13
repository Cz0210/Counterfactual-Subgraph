#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# User-authorized CM CPU exception to the repository A800 template.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_DATASET_CODE:?immutable code required}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "CM full job=$SLURM_JOB_ID host=$(hostname) python=$(command -v python)"
python -V
python -I -B scripts/run_cm_crem_dataset_full.py --config configs/hpc.yaml --spec "${CM_DATASET_SPEC:?spec required}" --action "${CM_ACTION:-run-shard}" --shard "${CM_SHARD:-0}"
