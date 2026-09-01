#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=bace-globalgce-calibration-successor

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export RUN_GNN_ABLATION=0

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch

print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

: "${OLD_CONTROLLER_ROOT:?Set the physical predecessor controller root}"
: "${OLD_CONTROLLER_ID:?Set the exact predecessor controller ID}"
: "${REGISTRY_RUN_ROOT:?Set the exact experiment_registry/run_state root}"
: "${CANDIDATE_RUN_ID:?Set the completed candidate run ID}"
: "${CANDIDATE_OUTPUT:?Set the immutable 20-rule candidate output}"
: "${SUCCESSOR_CONTROLLER_ID:?Set a fresh successor controller ID}"
: "${SUCCESSOR_OUTPUT_ROOT:?Set a fresh persistent output root}"
: "${SUCCESSOR_FRAGMENT:?Set a fresh fragment JSON path}"
: "${SUCCESSOR_MANIFEST:?Set a fresh controller manifest JSON path}"

python scripts/autodl/build_bace_globalgce_calibration_successor_v1.py \
  --config configs/hpc.yaml \
  --project-root "$PWD" \
  --python "$(command -v python)" \
  --old-controller-root "$OLD_CONTROLLER_ROOT" \
  --old-controller-id "$OLD_CONTROLLER_ID" \
  --registry-run-root "$REGISTRY_RUN_ROOT" \
  --run-id "$CANDIDATE_RUN_ID" \
  --candidate-output "$CANDIDATE_OUTPUT" \
  --controller-id "$SUCCESSOR_CONTROLLER_ID" \
  --output-root "$SUCCESSOR_OUTPUT_ROOT" \
  --fragment "$SUCCESSOR_FRAGMENT" \
  --manifest "$SUCCESSOR_MANIFEST"
