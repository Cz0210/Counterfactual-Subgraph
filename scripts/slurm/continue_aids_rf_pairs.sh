#!/bin/bash
# A+ explicitly authorizes AIDS CPU-only phase continuation; no GPU allocation.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=aids-rf-existing-pairs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate smiles_pip118
set -u
: "${AIDS_WORKTREE:?immutable worktree required}"
: "${AIDS_RUN_MANIFEST:?phase run manifest required}"
: "${AIDS_RECOURSE_ROOT:?completed original pair root required}"
: "${AIDS_POOL_ROOT:?completed frozen pool required}"
: "${AIDS_OUTPUT_ROOT:?fresh owner root required}"
cd "$AIDS_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$AIDS_OUTPUT_ROOT/scratch"
export TMPDIR="$AIDS_OUTPUT_ROOT/scratch" XDG_CACHE_HOME="$AIDS_OUTPUT_ROOT/scratch/cache"
echo "python=$(command -v python)"
python --version
echo "cpu_only=true worker_count=1 thread_limit=2 job=${SLURM_JOB_ID:-local}"
# Source vectors remain on AutoDL. This paired wrapper documents the same CPU
# CLI; do not transfer/recompute the 10GB pair store to use it on HPC.
exec nice -n 10 python -B scripts/continue_aids_rf_pairs.py --config configs/hpc.yaml \
  --run-manifest "$AIDS_RUN_MANIFEST" --recourse-root "$AIDS_RECOURSE_ROOT" \
  --pool-root "$AIDS_POOL_ROOT" --output-root "$AIDS_OUTPUT_ROOT" --action "${AIDS_ACTION:-owner}"
