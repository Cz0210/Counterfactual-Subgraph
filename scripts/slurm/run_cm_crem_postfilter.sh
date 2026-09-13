#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicit user-authorized CPU evaluation, no heuristic generation fallback.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM_POST_CODE:?immutable code}"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "CM evaluation job=$SLURM_JOB_ID host=$(hostname) python=$(command -v python)"
python -V
run_stage() {
 python -I -B scripts/run_cm_crem_postfilter.py --config configs/hpc.yaml --spec "$CM_POST_SPEC" --action "$1" --split "${2:-calibration}" --shard "${CM_SHARD:-0}" --shards "${CM_SHARDS:-1}"
}
case "${CM_POST_STAGE:?actual stage}" in
 fixture|complete)
  run_stage seal
  run_stage blocks calibration
  run_stage select
  run_stage blocks test
  run_stage audit
  run_stage export
  run_stage package;;
 prepare) run_stage seal; run_stage prepare calibration;;
 calibration) run_stage blocks calibration;;
 closeout) run_stage select; run_stage blocks test; run_stage audit; run_stage export; run_stage package;;
 *) exit 2;;
esac
