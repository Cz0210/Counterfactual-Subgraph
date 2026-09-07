#!/bin/bash
# Explicit taskbook CPU exception: no GPU, no MolCLR/OT or model training.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
# Submit from the verified immutable execution worktree using --chdir.
test -f scripts/experiments/run_bace_gin_reach_supplement.py
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/run_bace_gin_reach_supplement.py --config configs/hpc.yaml "$@"
