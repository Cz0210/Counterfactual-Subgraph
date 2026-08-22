#!/bin/bash
: <<'PYDOC'
Fail-closed resource wrapper for the Mutagenicity trace-off continuation.

Generation owns one exclusive controller GPU and a stable scientific output
root so a single bounded transient retry can restore an exact mirrored
checkpoint.  Parity and standardization are CPU-only.  Every stage shares the
same host-memory flock used by the reviewed AIDS repair and publishes PASS only
after its terminal validator succeeds.
PYDOC

set -euo pipefail

: "${MUT_TRACEOFF_STAGE:?MUT_TRACEOFF_STAGE is required}"
: "${MUT_STAGE_OUTPUT:?MUT_STAGE_OUTPUT is required}"
: "${MUT_SOURCE_ROOT:?MUT_SOURCE_ROOT is required}"
: "${MUT_CONTROLLER_PROJECT_ROOT:?MUT_CONTROLLER_PROJECT_ROOT is required}"
: "${MUT_CONTROLLER_COMMIT:?MUT_CONTROLLER_COMMIT is required}"
: "${MUT_INSTRUMENTATION_PROJECT_ROOT:?MUT_INSTRUMENTATION_PROJECT_ROOT is required}"
: "${MUT_EXECUTION_COMMIT:?MUT_EXECUTION_COMMIT is required}"
: "${AUTODL_PYTHON:?AUTODL_PYTHON is required}"
: "${COMRECGC_HIGHMEM_LOCK_PATH:?COMRECGC_HIGHMEM_LOCK_PATH is required}"
: "${COMRECGC_CGROUP_MEMORY_ROOT:?COMRECGC_CGROUP_MEMORY_ROOT is required}"
: "${COMRECGC_MIN_CGROUP_FREE_BYTES:?COMRECGC_MIN_CGROUP_FREE_BYTES is required}"
: "${COMRECGC_PROC_ROOT:=/proc}"

if [[ "$MUT_STAGE_OUTPUT" != /* || -e "$MUT_STAGE_OUTPUT" ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] stage output must be a fresh absolute path: $MUT_STAGE_OUTPUT" >&2
  exit 2
fi
for required_path in \
  "$MUT_SOURCE_ROOT" \
  "$MUT_CONTROLLER_PROJECT_ROOT" \
  "$MUT_INSTRUMENTATION_PROJECT_ROOT" \
  "$COMRECGC_CGROUP_MEMORY_ROOT" \
  "$COMRECGC_PROC_ROOT"; do
  if [[ "$required_path" != /* || ! -d "$required_path" ]]; then
    echo "[MUT_TRACEOFF_STAGE_FAIL] required physical directory is unavailable: $required_path" >&2
    exit 2
  fi
done
observed_controller_commit="$(git -C "$MUT_CONTROLLER_PROJECT_ROOT" rev-parse HEAD)"
if [[ "$observed_controller_commit" != "$MUT_CONTROLLER_COMMIT" ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] immutable controller commit changed" >&2
  exit 2
fi
observed_commit="$(git -C "$MUT_INSTRUMENTATION_PROJECT_ROOT" rev-parse HEAD)"
if [[ "$observed_commit" != "$MUT_EXECUTION_COMMIT" ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] immutable execution commit changed" >&2
  exit 2
fi
cd "$MUT_CONTROLLER_PROJECT_ROOT"
if [[ "$COMRECGC_HIGHMEM_LOCK_PATH" != /* ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] high-memory lock must be absolute" >&2
  exit 2
fi
if [[ ! "$COMRECGC_MIN_CGROUP_FREE_BYTES" =~ ^[0-9]+$ ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] cgroup headroom must be an integer" >&2
  exit 2
fi

flock_bin="${COMRECGC_FLOCK_BIN:-}"
if [[ -z "$flock_bin" ]]; then
  flock_bin="$(command -v flock || true)"
fi
if [[ "$flock_bin" != /* || ! -x "$flock_bin" ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] flock executable is unavailable" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$COMRECGC_HIGHMEM_LOCK_PATH")"
exec 9>"$COMRECGC_HIGHMEM_LOCK_PATH"
"$flock_bin" --exclusive 9

limit_path="$COMRECGC_CGROUP_MEMORY_ROOT/memory.limit_in_bytes"
usage_path="$COMRECGC_CGROUP_MEMORY_ROOT/memory.usage_in_bytes"
if [[ ! -r "$limit_path" || ! -r "$usage_path" ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] cgroup-v1 memory counters are unavailable" >&2
  exit 2
fi
read -r memory_limit < "$limit_path"
read -r memory_usage < "$usage_path"
if [[ ! "$memory_limit" =~ ^[0-9]+$ || ! "$memory_usage" =~ ^[0-9]+$ ]]; then
  echo "[MUT_TRACEOFF_STAGE_FAIL] cgroup memory counters are malformed" >&2
  exit 2
fi
if (( memory_usage >= memory_limit )); then
  echo "[MUT_TRACEOFF_STAGE_RETRY] cgroup has no free memory" >&2
  exit 75
fi
memory_free=$((memory_limit - memory_usage))
if (( memory_free < COMRECGC_MIN_CGROUP_FREE_BYTES )); then
  echo "[MUT_TRACEOFF_STAGE_RETRY] insufficient cgroup headroom free=$memory_free required=$COMRECGC_MIN_CGROUP_FREE_BYTES" >&2
  exit 75
fi

for command_file in "$COMRECGC_PROC_ROOT"/[0-9]*/cmdline; do
  [[ -r "$command_file" ]] || continue
  command_text="$(tr '\0' ' ' < "$command_file" 2>/dev/null || true)"
  if [[ "$command_text" == *"scripts/baselines/comrecgc/run_common_recourse.py"* ]]; then
    echo "[MUT_TRACEOFF_STAGE_RETRY] another common-recourse process is active" >&2
    exit 75
  fi
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

case "$MUT_TRACEOFF_STAGE" in
  instrumentation-equivalence)
    : "${MUT_LEGACY_PROJECT_ROOT:?MUT_LEGACY_PROJECT_ROOT is required}"
    : "${MUT_SOURCE_PROJECT_COMMIT:?MUT_SOURCE_PROJECT_COMMIT is required}"
    : "${MUT_LEGACY_SOURCE_INVENTORY_SHA256:?MUT_LEGACY_SOURCE_INVENTORY_SHA256 is required}"
    : "${MUT_INSTRUMENTATION_SOURCE_INVENTORY_SHA256:?MUT_INSTRUMENTATION_SOURCE_INVENTORY_SHA256 is required}"
    : "${MUT_EQUIVALENCE_RUN_ROOT:?MUT_EQUIVALENCE_RUN_ROOT is required}"
    : "${MUT_UPSTREAM_ROOT:?MUT_UPSTREAM_ROOT is required}"
    : "${MUT_DATASET_DIR:?MUT_DATASET_DIR is required}"
    : "${MUT_GNN_CHECKPOINT:?MUT_GNN_CHECKPOINT is required}"
    : "${MUT_DISTANCE_CHECKPOINT:?MUT_DISTANCE_CHECKPOINT is required}"
    : "${MUT_BATCH_SIZE:=128}"
    if [[ "${GPU_REQUIRED:-}" != "1" || "${DEVICE:-}" != "cuda:0" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] equivalence requires exclusive GPU_REQUIRED=1 DEVICE=cuda:0" >&2
      exit 2
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] controller did not assign an equivalence GPU" >&2
      exit 2
    fi
    legacy_commit="$(git -C "$MUT_LEGACY_PROJECT_ROOT" rev-parse HEAD)"
    if [[ "$legacy_commit" != "$MUT_SOURCE_PROJECT_COMMIT" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] legacy scientific worktree changed" >&2
      exit 2
    fi
    "$AUTODL_PYTHON" scripts/autodl/run_mut_checkpoint_instrumentation_equivalence.py \
      --config configs/hpc.yaml \
      --set inference.fallback_to_heuristic=false \
      run-pair \
      --python "$AUTODL_PYTHON" \
      --legacy-project-root "$MUT_LEGACY_PROJECT_ROOT" \
      --execution-project-root "$MUT_INSTRUMENTATION_PROJECT_ROOT" \
      --execution-commit "$MUT_EXECUTION_COMMIT" \
      --expected-legacy-inventory-sha256 "$MUT_LEGACY_SOURCE_INVENTORY_SHA256" \
      --expected-instrumentation-inventory-sha256 "$MUT_INSTRUMENTATION_SOURCE_INVENTORY_SHA256" \
      --run-root "$MUT_EQUIVALENCE_RUN_ROOT" \
      --output-dir "$MUT_STAGE_OUTPUT" \
      --upstream-root "$MUT_UPSTREAM_ROOT" \
      --dataset-dir "$MUT_DATASET_DIR" \
      --gnn-checkpoint "$MUT_GNN_CHECKPOINT" \
      --distance-checkpoint "$MUT_DISTANCE_CHECKPOINT" \
      --parent-limit 1448 \
      --device cuda:0 \
      --batch-size "$MUT_BATCH_SIZE"
    ;;
  generation)
    : "${MUT_UPSTREAM_ROOT:?MUT_UPSTREAM_ROOT is required}"
    : "${MUT_DATASET_DIR:?MUT_DATASET_DIR is required}"
    : "${MUT_GNN_CHECKPOINT:?MUT_GNN_CHECKPOINT is required}"
    : "${MUT_DISTANCE_CHECKPOINT:?MUT_DISTANCE_CHECKPOINT is required}"
    : "${MUT_GENERATION_OUTPUT:?MUT_GENERATION_OUTPUT is required}"
    : "${MUT_CHECKPOINT_ROOT:?MUT_CHECKPOINT_ROOT is required}"
    : "${MUT_CHECKPOINT_MIRROR_ROOT:?MUT_CHECKPOINT_MIRROR_ROOT is required}"
    : "${MUT_INSTRUMENTATION_EQUIVALENCE_GATE:?MUT_INSTRUMENTATION_EQUIVALENCE_GATE is required}"
    : "${MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256:?MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256 is required}"
    : "${MUT_BATCH_SIZE:=128}"
    if [[ "${GPU_REQUIRED:-}" != "1" || "${DEVICE:-}" != "cuda:0" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] generation requires exclusive GPU_REQUIRED=1 DEVICE=cuda:0" >&2
      exit 2
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] controller did not assign a physical GPU" >&2
      exit 2
    fi
    "$AUTODL_PYTHON" scripts/autodl/manage_mut_traceoff_parity_v1.py \
      --config configs/hpc.yaml verify-instrumentation-equivalence \
      --gate "$MUT_INSTRUMENTATION_EQUIVALENCE_GATE" \
      --expected-legacy-inventory-sha256 "$MUT_LEGACY_SOURCE_INVENTORY_SHA256" \
      --expected-instrumentation-inventory-sha256 "$MUT_INSTRUMENTATION_SOURCE_INVENTORY_SHA256"
    resume_args=()
    if [[ -f "$MUT_GENERATION_OUTPUT/_RUN_COMPLETE.json" ]]; then
      :
    elif [[ -e "$MUT_GENERATION_OUTPUT" ]]; then
      "$AUTODL_PYTHON" scripts/autodl/manage_mut_traceoff_parity_v1.py \
        --config configs/hpc.yaml prepare-resume \
        --output-root "$MUT_GENERATION_OUTPUT" \
        --checkpoint-root "$MUT_CHECKPOINT_ROOT" \
        --mirror-root "$MUT_CHECKPOINT_MIRROR_ROOT"
      resume_args+=(--resume)
    else
      "$AUTODL_PYTHON" "$MUT_INSTRUMENTATION_PROJECT_ROOT/scripts/baselines/comrecgc/run_generation.py" \
        --config configs/hpc.yaml \
        --set inference.fallback_to_heuristic=false \
        --route project \
        --dataset mutagenicity \
        --mode full \
        --project-root "$MUT_INSTRUMENTATION_PROJECT_ROOT" \
        --upstream-root "$MUT_UPSTREAM_ROOT" \
        --dataset-dir "$MUT_DATASET_DIR" \
        --gnn-checkpoint "$MUT_GNN_CHECKPOINT" \
        --distance-checkpoint "$MUT_DISTANCE_CHECKPOINT" \
        --output-dir "$MUT_GENERATION_OUTPUT" \
        --parent-limit 1448 \
        --device cuda:0 \
        --batch-size "$MUT_BATCH_SIZE" \
        --graph-state-dir "$MUT_GENERATION_OUTPUT/graph_state" \
        --storage-guard-root "$MUT_GENERATION_OUTPUT" \
        --storage-check-every-steps 500 \
        --storage-min-free-gib 50 \
        --storage-min-free-ratio 0.02 \
        --storage-min-free-inodes 100000 \
        --checkpoint-root "$MUT_CHECKPOINT_ROOT" \
        --checkpoint-mirror-root "$MUT_CHECKPOINT_MIRROR_ROOT" \
        --checkpoint-interval-steps 500 \
        --checkpoint-keep-last 2 \
        --progress-interval-steps 25
    fi
    if (( ${#resume_args[@]} )); then
      "$AUTODL_PYTHON" "$MUT_INSTRUMENTATION_PROJECT_ROOT/scripts/baselines/comrecgc/run_generation.py" \
        --config configs/hpc.yaml \
        --set inference.fallback_to_heuristic=false \
        --route project \
        --dataset mutagenicity \
        --mode full \
        --project-root "$MUT_INSTRUMENTATION_PROJECT_ROOT" \
        --upstream-root "$MUT_UPSTREAM_ROOT" \
        --dataset-dir "$MUT_DATASET_DIR" \
        --gnn-checkpoint "$MUT_GNN_CHECKPOINT" \
        --distance-checkpoint "$MUT_DISTANCE_CHECKPOINT" \
        --output-dir "$MUT_GENERATION_OUTPUT" \
        --parent-limit 1448 \
        --device cuda:0 \
        --batch-size "$MUT_BATCH_SIZE" \
        --graph-state-dir "$MUT_GENERATION_OUTPUT/graph_state" \
        --storage-guard-root "$MUT_GENERATION_OUTPUT" \
        --storage-check-every-steps 500 \
        --storage-min-free-gib 50 \
        --storage-min-free-ratio 0.02 \
        --storage-min-free-inodes 100000 \
        --checkpoint-root "$MUT_CHECKPOINT_ROOT" \
        --checkpoint-mirror-root "$MUT_CHECKPOINT_MIRROR_ROOT" \
        --checkpoint-interval-steps 500 \
        --checkpoint-keep-last 2 \
        --progress-interval-steps 25 \
        "${resume_args[@]}"
    fi
    "$AUTODL_PYTHON" scripts/autodl/manage_mut_traceoff_parity_v1.py \
      --config configs/hpc.yaml verify-traceoff-reference \
      --reference-root "$MUT_GENERATION_OUTPUT" \
      --traced-source-root "$MUT_SOURCE_ROOT" \
      --expected-project-commit "$MUT_EXECUTION_COMMIT" \
      --expected-scientific-command-sha256 "$MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256" \
      --checkpoint-root "$MUT_CHECKPOINT_ROOT" \
      --mirror-root "$MUT_CHECKPOINT_MIRROR_ROOT" \
      --proc-root "$COMRECGC_PROC_ROOT" \
      --output-dir "$MUT_STAGE_OUTPUT"
    ;;
  parity)
    : "${MUT_GENERATION_OUTPUT:?MUT_GENERATION_OUTPUT is required}"
    : "${MUT_CHECKPOINT_ROOT:?MUT_CHECKPOINT_ROOT is required}"
    : "${MUT_CHECKPOINT_MIRROR_ROOT:?MUT_CHECKPOINT_MIRROR_ROOT is required}"
    : "${MUT_EXECUTION_COMMIT:?MUT_EXECUTION_COMMIT is required}"
    : "${MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256:?MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256 is required}"
    if [[ "${GPU_REQUIRED:-}" != "0" || "${DEVICE:-}" != "cpu" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] parity must be CPU-only" >&2
      exit 2
    fi
    "$AUTODL_PYTHON" scripts/autodl/manage_mut_traceoff_parity_v1.py \
      --config configs/hpc.yaml assert-parity \
      --reference-root "$MUT_GENERATION_OUTPUT" \
      --traced-source-root "$MUT_SOURCE_ROOT" \
      --expected-project-commit "$MUT_EXECUTION_COMMIT" \
      --expected-scientific-command-sha256 "$MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256" \
      --checkpoint-root "$MUT_CHECKPOINT_ROOT" \
      --mirror-root "$MUT_CHECKPOINT_MIRROR_ROOT" \
      --proc-root "$COMRECGC_PROC_ROOT" \
      --output-dir "$MUT_STAGE_OUTPUT"
    ;;
  standardization)
    : "${MUT_UPSTREAM_ROOT:?MUT_UPSTREAM_ROOT is required}"
    : "${MUT_DATASET_DIR:?MUT_DATASET_DIR is required}"
    : "${MUT_DISTANCE_CHECKPOINT:?MUT_DISTANCE_CHECKPOINT is required}"
    : "${MUT_DATASET_CSV:?MUT_DATASET_CSV is required}"
    : "${MUT_TEACHER_PATH:?MUT_TEACHER_PATH is required}"
    : "${MUT_MOLCLR_ROOT:?MUT_MOLCLR_ROOT is required}"
    : "${MUT_MOLCLR_CHECKPOINT:?MUT_MOLCLR_CHECKPOINT is required}"
    : "${MUT_THRESHOLDS_PATH:?MUT_THRESHOLDS_PATH is required}"
    : "${MUT_COMMON_ADOPTION_GATE:?MUT_COMMON_ADOPTION_GATE is required}"
    : "${MUT_PARITY_GATE:?MUT_PARITY_GATE is required}"
    if [[ "${GPU_REQUIRED:-}" != "0" || "${DEVICE:-}" != "cpu" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[MUT_TRACEOFF_STAGE_FAIL] standardization must be CPU-only" >&2
      exit 2
    fi
    "$AUTODL_PYTHON" scripts/autodl/run_mut_comrecgc_parity_standardization.py \
      --config configs/hpc.yaml \
      --set inference.fallback_to_heuristic=false \
      --source-generation-root "$MUT_SOURCE_ROOT" \
      --upstream-root "$MUT_UPSTREAM_ROOT" \
      --dataset-dir "$MUT_DATASET_DIR" \
      --distance-checkpoint "$MUT_DISTANCE_CHECKPOINT" \
      --dataset-csv "$MUT_DATASET_CSV" \
      --teacher-path "$MUT_TEACHER_PATH" \
      --molclr-root "$MUT_MOLCLR_ROOT" \
      --molclr-checkpoint "$MUT_MOLCLR_CHECKPOINT" \
      --thresholds-path "$MUT_THRESHOLDS_PATH" \
      --common-adoption "$MUT_COMMON_ADOPTION_GATE" \
      --trace-parity "$MUT_PARITY_GATE" \
      --output-root "$MUT_STAGE_OUTPUT" \
      --device cpu
    ;;
  *)
    echo "[MUT_TRACEOFF_STAGE_FAIL] unsupported stage: $MUT_TRACEOFF_STAGE" >&2
    exit 2
    ;;
esac
