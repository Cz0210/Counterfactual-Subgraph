#!/bin/bash
# V9: AutoDL-only inherited-FD child; explicit same-GPU identity adapter.
# The reviewed four scientific files and original cross-GPU guard are retained.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${T12_EXECUTION_ROOT:?}"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
echo 'T12 science requires the AutoDL owner and inherited descriptor; HPC science is not permitted.'
exit 2
