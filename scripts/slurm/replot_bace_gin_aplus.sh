#!/bin/bash
# CPU-only display reducer; no oracle, training, selector or OT.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
test -f scripts/experiments/replot_bace_gin_aplus.py
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/replot_bace_gin_aplus.py --config configs/hpc.yaml "$@"
