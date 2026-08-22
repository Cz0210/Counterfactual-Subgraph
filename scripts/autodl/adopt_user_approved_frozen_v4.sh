#!/usr/bin/env bash
# AutoDL-only CPU adoption wrapper. It never invokes Slurm or reserves a GPU.
set -euo pipefail

PY="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
RUNTIME="${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}"
SOURCE_ROOT="${FROZEN_V4_SOURCE_ROOT:-/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_matched_aids_v4}"
: "${FROZEN_V4_OUTPUT_ROOT:?set FROZEN_V4_OUTPUT_ROOT to a fresh persistent root}"

export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" scripts/autodl/adopt_user_approved_frozen_v4.py \
  --config configs/hpc.yaml \
  --source-root "$SOURCE_ROOT" \
  --runtime-root "$RUNTIME" \
  --output-root "$FROZEN_V4_OUTPUT_ROOT"
