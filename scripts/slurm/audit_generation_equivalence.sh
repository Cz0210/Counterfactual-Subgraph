#!/bin/bash
#SBATCH --job-name=comrecgc_ab_audit
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

python scripts/baselines/comrecgc/audit_generation_equivalence.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --legacy-root "${LEGACY_ROOT:?}" \
  --optimized-root "${OPTIMIZED_ROOT:?}" \
  --output-dir "${OUTPUT_DIR:?}" \
  --expected-steps "${EXPECTED_STEPS:?500 or 1000}"
