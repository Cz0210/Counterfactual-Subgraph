#!/bin/bash
# Task-specific CPU diagnostic exception; no GPU, generation, heuristic or OT.
#SBATCH --partition=intel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
cd "${CM_DIAGNOSTIC_CODE:?}"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "diagnostic host=$(hostname) job=$SLURM_JOB_ID python=$(command -v python)"
python -V
python -I -B scripts/diagnose_cm_crem_oracle.py --config configs/hpc.yaml --spec "$CM_SPEC" --source-root "$CM_SOURCE" --output-root "$CM_DIAGNOSTIC_OUT"
