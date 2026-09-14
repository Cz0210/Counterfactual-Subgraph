#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Record-only CPU packaging: no GPU, inference, OT or heuristic fallback.
# AIDS SOURCE-DESCRIPTIVE requires scope/overlap binding; never main authority.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_RELEASE_CODE:?immutable transport code}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
echo "Record-only CM package job=$SLURM_JOB_ID python=$(command -v python)"
python -V
python -I -B scripts/package_cm_postfilter.py --config configs/hpc.yaml --root "${CM_POST_ROOT:?accepted root}"
