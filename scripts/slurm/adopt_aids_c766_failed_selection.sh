#!/bin/bash
# Static CLI parity for an AutoDL-only recovery-evidence adoption; do not submit.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=aids-c766-failed-selection-adopt

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

echo "This read-only recovery-evidence adoption is AutoDL-only; do not submit." >&2
echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
PY
exit 78

# Unreachable documentation-only CLI parity. A failed source is never a normal
# PASS dependency, and this script intentionally does not authorize HPC use.
python scripts/autodl/adopt_aids_c766_failed_selection.py \
  --config configs/hpc.yaml \
  --output-dir /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery_evidence/aids_c766_failed_selection_v1/FRESH_CHILD_REQUIRED
