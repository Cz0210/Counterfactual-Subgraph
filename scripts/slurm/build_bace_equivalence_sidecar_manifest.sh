#!/bin/bash
#SBATCH --job-name=bace_eq_sidecar_build
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

# Static CLI parity only.  The active campaign is AutoDL-only and this wrapper
# must not be submitted for the current continuation.
python scripts/autodl/build_bace_equivalence_sidecar_manifest.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --controller-id "${CONTROLLER_ID:?}" \
  --runtime-root "${AUTODL_RUNTIME_ROOT:?}" \
  --python "$(command -v python)" \
  --output-root "${OUTPUT_ROOT:?}" \
  --output-manifest "${OUTPUT_MANIFEST:?}" \
  --build-audit "${BUILD_AUDIT:?}" \
  --dataset-dir "${BACE_GCF_DATASET_DIR:?}" \
  --gcf-official-root "${GCF_OFFICIAL_ROOT:?}" \
  --gine-checkpoint "${BACE_GINE_CHECKPOINT:?}" \
  --neurosed-checkpoint "${BACE_NEUROSED_CHECKPOINT:?}" \
  --neurosed-manifest "${BACE_NEUROSED_MANIFEST:?}"
