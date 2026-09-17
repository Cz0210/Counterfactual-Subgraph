#!/bin/bash
# V7 CPU-only status; T12 GPU science must run via the AutoDL canonical owner.
# V9 --observer-receipt belongs to AutoDL owner activation, never this status job.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${T12_EXECUTION_ROOT:?}"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
python -I -B scripts/run_t12_v7_owner.py --config configs/hpc.yaml --action status --root "${T12_STATUS_ROOT:?}"
