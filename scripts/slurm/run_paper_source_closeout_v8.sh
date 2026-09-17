#!/bin/bash
# Local-disk sources must be explicitly staged before using this CPU wrapper.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${PAPER_EXECUTION_ROOT:?}"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python) CPU-only"
python --version
python -I -B scripts/run_paper_source_closeout_v8.py --config configs/hpc.yaml \
  --base "${PAPER_SOURCE_BASE:?}" --paper "${PAPER_COPY_ROOT:?}" --out-dir "${PAPER_OUTPUT_ROOT:?}"
