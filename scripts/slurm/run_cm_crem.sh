#!/bin/bash
# Pilot closeout measures model/filter/I/O plus generator RSS; full stages require
# both the 168h horizon and this job's 12h/32GiB resource admission. No GPU used.
# Task-authorized CPU exception to the repository A800 template; no GPU request.
#SBATCH --partition=intel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=cm-crem-bace

# Site bashrc is sourced before nounset (known /etc/bashrc bootstrap contract).
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
: "${CM_EXECUTION_ROOT:?immutable worktree required}"
: "${CM_SPEC:?resolved absolute spec required}"
: "${CM_RUN_ROOT:?fresh task root required}"
: "${CM_ACTION:?explicit stage required}"
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
cd "$CM_EXECUTION_ROOT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CM_PYTHON="/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python"
if [[ "$CM_ACTION" == generate ]]; then
  : "${CM_GENERATOR_PYTHON:?isolated generator interpreter required}"
  CM_PYTHON="$CM_GENERATOR_PYTHON"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fi
echo "CM action=$CM_ACTION Slurm=$SLURM_JOB_ID host=$(hostname) Python=$CM_PYTHON"
"$CM_PYTHON" -V
CM_ARGS=(--config configs/hpc.yaml --set inference.fallback_to_heuristic=false --spec "$CM_SPEC" --run-root "$CM_RUN_ROOT" --action "$CM_ACTION")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  CM_ARGS+=(--shard "$SLURM_ARRAY_TASK_ID" --shards 2)
fi
if [[ "${CM_PILOT_ONLY:-0}" == 1 ]]; then CM_ARGS+=(--pilot-only); fi
if [[ "$CM_ACTION" == generate ]]; then
  # -I implies -E and ignores PYTHONHASHSEED; the generator verifies actual flags.
  # Its dedicated environment and explicit trusted bootstrap isolate imports.
  unset PYTHONPATH
  "$CM_PYTHON" -s -B scripts/run_cm_crem.py "${CM_ARGS[@]}"
else
  "$CM_PYTHON" -I -B scripts/run_cm_crem.py "${CM_ARGS[@]}"
fi
