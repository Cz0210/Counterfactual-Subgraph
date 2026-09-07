#!/bin/bash
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
echo "python=$(command -v python)"
python --version
echo "cpu_only=true job=${SLURM_JOB_ID:-local}"
exec nice -n 10 python -B scripts/repair_aids_rf_aligned.py --config configs/hpc.yaml --run-manifest "$AIDS_RUN_MANIFEST" --output-root "$AIDS_OUTPUT_ROOT" --action screen-pool
