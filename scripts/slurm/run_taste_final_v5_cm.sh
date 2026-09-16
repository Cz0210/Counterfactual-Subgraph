#!/bin/bash
# Explicit V5 user override: HPC CPU only; no GPU or generation/model fallback.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${TASTE_V5_CODE:?immutable worktree}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "V5 CPU job=$SLURM_JOB_ID host=$(hostname) python=$(command -v python)"
python -V
python -c 'import torch; print("cuda_available=",torch.cuda.is_available())'
for stage in select evaluate audit export; do
 python -I -B scripts/run_taste_final_v5_cm.py --config configs/hpc.yaml \
  --contract "$TASTE_V5_CONTRACT" --source-root "$TASTE_V5_CM_SOURCE" \
  --output-root "$TASTE_V5_CM_OUTPUT" --action "$stage"
done
