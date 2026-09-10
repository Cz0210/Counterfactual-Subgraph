#!/bin/bash
# Explicit user CPU evaluation exception; no generation and no GPU claim.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
: "${CM_K20_CODE:?immutable execution root required}"
: "${CM_K20_SPEC:?sealed scope spec required}"
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
cd "$CM_K20_CODE"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "CM K20 job=$SLURM_JOB_ID node=$(hostname) Python=$(command -v python) CPU-only"
python -V
# No heuristic inference path exists in this saved-record/exact-distance driver.
python -I -B scripts/run_cm_crem_k20.py --config configs/hpc.yaml --spec "$CM_K20_SPEC" --action run
