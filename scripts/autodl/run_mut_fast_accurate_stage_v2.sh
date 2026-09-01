#!/usr/bin/env bash
# Thin stage adapter for the Mut historical-50k successor.  Admission is owned
# by the persistent sidecar, before the four-GPU controller launches this file.

set -euo pipefail

: "${MUT_FAST_STAGE:?MUT_FAST_STAGE is required}"
: "${MUT_STAGE_OUTPUT:?MUT_STAGE_OUTPUT is required}"
: "${AUTODL_PYTHON:?AUTODL_PYTHON is required}"
: "${MUT_FAST_SPEC:?MUT_FAST_SPEC is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
[[ "$MUT_STAGE_OUTPUT" == /* && ! -e "$MUT_STAGE_OUTPUT" ]] || {
  echo "fresh absolute MUT_STAGE_OUTPUT required: $MUT_STAGE_OUTPUT" >&2
  exit 64
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RUN_GNN_ABLATION=0
cd "$PROJECT_ROOT"

case "$MUT_FAST_STAGE" in
  equivalence)
    [[ "${GPU_REQUIRED:-}" == "1" && "${DEVICE:-}" == "cuda:0" ]] || {
      echo "equivalence requires the controller's exclusive GPU" >&2
      exit 64
    }
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
      echo "four-GPU controller did not assign CUDA_VISIBLE_DEVICES" >&2
      exit 64
    }
    # A race after the sidecar's stable admission must not consume retries.
    # Hold the already-exclusive slot and keep waiting; semantic/cgroup-shape
    # errors still fail closed in `wait-admission`.
    "$AUTODL_PYTHON" scripts/autodl/run_mut_fast_accurate_v2.py \
      --config configs/hpc.yaml wait-admission --spec "$MUT_FAST_SPEC" \
      --require-assigned-gpu
    "$AUTODL_PYTHON" scripts/autodl/run_mut_fast_accurate_v2.py \
      --config configs/hpc.yaml run-equivalence --spec "$MUT_FAST_SPEC" \
      --run-root "$MUT_EQUIVALENCE_RUN_ROOT" \
      --output-dir "$MUT_STAGE_OUTPUT"
    ;;
  bind-adoption)
    "$AUTODL_PYTHON" scripts/autodl/run_mut_fast_accurate_v2.py \
      --config configs/hpc.yaml bind-adoption --spec "$MUT_FAST_SPEC" \
      --inventory-gate "$MUT_INVENTORY_GATE" \
      --equivalence-gate "$MUT_EQUIVALENCE_GATE" \
      --output-dir "$MUT_STAGE_OUTPUT"
    ;;
  standardize)
    # The existing standardizer owns chemistry -> evaluation -> freeze.  Its
    # historical-adoption flag emits the truthful non-parity schema.
    "$AUTODL_PYTHON" scripts/autodl/run_mut_comrecgc_parity_standardization.py \
      --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
      --source-generation-root "$MUT_SOURCE_GENERATION_ROOT" \
      --upstream-root "$MUT_UPSTREAM_ROOT" \
      --dataset-dir "$MUT_DATASET_DIR" \
      --distance-checkpoint "$MUT_DISTANCE_CHECKPOINT" \
      --dataset-csv "$MUT_DATASET_CSV" \
      --teacher-path "$MUT_TEACHER_PATH" \
      --molclr-root "$MUT_MOLCLR_ROOT" \
      --molclr-checkpoint "$MUT_MOLCLR_CHECKPOINT" \
      --thresholds-path "$MUT_THRESHOLDS_PATH" \
      --historical-adoption "$MUT_HISTORICAL_ADOPTION" \
      --output-root "$MUT_STAGE_OUTPUT" --device cpu
    ;;
  *)
    echo "unsupported MUT_FAST_STAGE=$MUT_FAST_STAGE" >&2
    exit 64
    ;;
esac
