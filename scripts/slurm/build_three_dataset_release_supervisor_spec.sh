#!/usr/bin/env bash
# Static CLI-parity wrapper only. The active release supervisor is an AutoDL
# CPU-only sidecar; submitting this mandatory GPU-shaped wrapper is forbidden.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "AutoDL CPU-only release sidecar; do not submit this Slurm wrapper." >&2
exit 78

# Unreachable documentation-only CLI parity command:
python scripts/autodl/build_three_dataset_release_supervisor_spec.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --catalog "${RELEASE_CATALOG:?}" \
  --controller-id "${RELEASE_CONTROLLER_ID:?}" \
  --project-root "$PWD" \
  --runtime-root "${AUTODL_RUNTIME_ROOT:?}" \
  --python "$(command -v python)" \
  --state-root "${RELEASE_STATE_ROOT:?}" \
  --registry-root "${RELEASE_REGISTRY_ROOT:?}" \
  --output-root "${RELEASE_OUTPUT_ROOT:?}" \
  --paper-staging-root "${RELEASE_PAPER_STAGING_ROOT:?}" \
  --expectations-json "${RELEASE_EXPECTATIONS_JSON:?}" \
  --taste-license-gate-json "${TASTE_LICENSE_GATE_JSON:?}" \
  --spec-output "${RELEASE_SPEC_OUTPUT:?}" \
  --build-audit "${RELEASE_BUILD_AUDIT:?}"
