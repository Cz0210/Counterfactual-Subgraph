#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicit native CPU controls; no oracle, OT, training or new candidate budget.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_CONTROL_CODE:?immutable reviewed code}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
echo "CM AIDS controls job=$SLURM_JOB_ID interpreter=$CM_GENERATOR_PYTHON"
"$CM_GENERATOR_PYTHON" -V
"$CM_GENERATOR_PYTHON" -I -B scripts/verify_cm_aids_native_controls.py --config configs/hpc.yaml --source-root "$CM_AIDS_SOURCE" --diagnosis-root "$CM_AIDS_DIAGNOSIS" --output-root "$CM_AIDS_CONTROLS"
