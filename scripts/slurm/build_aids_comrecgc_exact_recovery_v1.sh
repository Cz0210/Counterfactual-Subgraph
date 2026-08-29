#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper only: this recovery is explicitly AutoDL-only/CPU-only.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'
echo "do not submit: AIDS disconnected-exact recovery is AutoDL-only" >&2
echo "reference only: generate-production requires explicit reviewed commit pins and --authorize-production-deployment for a launchable initial 8-worker spec" >&2
echo "validate: python scripts/autodl/build_aids_comrecgc_exact_recovery_v1.py --config configs/hpc.yaml validate --spec /absolute/spec.json" >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python scripts/autodl/build_aids_comrecgc_exact_recovery_v1.py \
  --config configs/hpc.yaml generate-production \
  --adoption-output /absolute/adoption \
  --controller-parent /absolute/controllers \
  --python /absolute/python \
  --project-root /absolute/immutable-execution \
  --controller-manifest /absolute/controller.json \
  --thread-count 8 \
  --adoption-commit 7370006da6175851def0f151ca6fb4dfb44f2ab7 \
  --controller-commit 0000000000000000000000000000000000000000 \
  --exact-runner-commit ab14be7c70803384eb6904d85bbf87b070d8d961 \
  --subset-runner-commit ab14be7c70803384eb6904d85bbf87b070d8d961 \
  --downstream-runner-commit ab14be7c70803384eb6904d85bbf87b070d8d961 \
  --standardization-runner-commit ab14be7c70803384eb6904d85bbf87b070d8d961 \
  --authorize-production-deployment \
  --output /absolute/spec.json
