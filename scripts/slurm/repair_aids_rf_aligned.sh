#!/bin/bash
# AutoDL same-root resource-stop resume uses current private anonymous RSS,
# never high-water/file-cache RSS; this CPU wrapper keeps the scientific CLI.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=aids-rf-pool
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# This taskbook explicitly authorizes CPU-only AIDS screening, overriding the
# repository's generic A800 submission template. No GPU request is made.
set -eo pipefail
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate smiles_pip118
set -u
: "${AIDS_WORKTREE:?immutable worktree required}"
: "${AIDS_RUN_MANIFEST:?run manifest required}"
: "${AIDS_OUTPUT_ROOT:?fresh output root required}"
cd "$AIDS_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$(dirname "$AIDS_OUTPUT_ROOT")/scratch/slurm-${SLURM_JOB_ID:?}"
export XDG_CACHE_HOME="$TMPDIR/cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
echo "python=$(command -v python)"
python --version
echo "cpu_only=true job=${SLURM_JOB_ID:-local}"
if [ "${AIDS_ACTION:-screen-pool}" = "repair-gaps" ] || [ "${AIDS_ACTION:-screen-pool}" = "recourse" ]; then
  python -B -m unittest tests.test_aids_rf_aligned_pool -v
fi
if [ "${AIDS_ACTION:-screen-pool}" = "recourse" ] || [ "${AIDS_ACTION:-screen-pool}" = "repair-gaps" ] || [ "${AIDS_ACTION:-screen-pool}" = "freeze-summary" ] || [ "${AIDS_ACTION:-screen-pool}" = "evaluate" ] || [ "${AIDS_ACTION:-screen-pool}" = "release" ] || [ "${AIDS_ACTION:-screen-pool}" = "release-after-recourse" ]; then
  : "${AIDS_POOL_ROOT:?completed pool root required}"
  exec nice -n 10 python -B scripts/repair_aids_rf_aligned.py --config configs/hpc.yaml --run-manifest "$AIDS_RUN_MANIFEST" --output-root "$AIDS_OUTPUT_ROOT" --pool-root "$AIDS_POOL_ROOT" --action "$AIDS_ACTION"
fi
exec nice -n 10 python -B scripts/repair_aids_rf_aligned.py --config configs/hpc.yaml --run-manifest "$AIDS_RUN_MANIFEST" --output-root "$AIDS_OUTPUT_ROOT" --action screen-pool
