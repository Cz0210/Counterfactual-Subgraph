#!/bin/bash
: <<'PYDOC'
CPU-only, host-memory-exclusive launcher for the AIDS ComRecGC continuation.

The scientific continuation remains in
``run_comrecgc_standardized_continuation.sh``.  This wrapper owns only the
resource boundary that the first retry lacked: one persistent advisory lock,
an exact cgroup-v1 headroom check, and a fail-closed scan for an older
``run_common_recourse.py`` process that does not yet participate in the lock.
PYDOC

set -euo pipefail

: "${DATASET:?DATASET is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${COMRECGC_HIGHMEM_LOCK_PATH:?COMRECGC_HIGHMEM_LOCK_PATH is required}"
: "${COMRECGC_CGROUP_MEMORY_ROOT:?COMRECGC_CGROUP_MEMORY_ROOT is required}"
: "${COMRECGC_MIN_CGROUP_FREE_BYTES:?COMRECGC_MIN_CGROUP_FREE_BYTES is required}"
: "${COMRECGC_PROC_ROOT:=/proc}"

if [[ "$DATASET" != "aids" ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] dataset must be aids" >&2
  exit 2
fi
if [[ "${DEVICE:-}" != "cpu" || "${GPU_REQUIRED:-}" != "0" ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] DEVICE=cpu and GPU_REQUIRED=0 are mandatory" >&2
  exit 2
fi
if [[ "$COMRECGC_HIGHMEM_LOCK_PATH" != /* ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] lock path must be absolute" >&2
  exit 2
fi
if [[ "$COMRECGC_CGROUP_MEMORY_ROOT" != /* ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] cgroup root must be absolute" >&2
  exit 2
fi
if [[ "$COMRECGC_PROC_ROOT" != /* || ! -d "$COMRECGC_PROC_ROOT" ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] proc root must be an absolute directory" >&2
  exit 2
fi
if [[ ! "$COMRECGC_MIN_CGROUP_FREE_BYTES" =~ ^[0-9]+$ ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] minimum free bytes must be an integer" >&2
  exit 2
fi

lock_parent="$(dirname -- "$COMRECGC_HIGHMEM_LOCK_PATH")"
mkdir -p -- "$lock_parent"
exec 9>"$COMRECGC_HIGHMEM_LOCK_PATH"
flock_bin="${COMRECGC_FLOCK_BIN:-}"
if [[ -z "$flock_bin" ]]; then
  flock_bin="$(command -v flock || true)"
fi
if [[ "$flock_bin" != /* || ! -x "$flock_bin" ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] flock executable is unavailable" >&2
  exit 2
fi
"$flock_bin" --exclusive 9

limit_path="$COMRECGC_CGROUP_MEMORY_ROOT/memory.limit_in_bytes"
usage_path="$COMRECGC_CGROUP_MEMORY_ROOT/memory.usage_in_bytes"
if [[ ! -r "$limit_path" || ! -r "$usage_path" ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] cgroup-v1 memory counters are unavailable" >&2
  exit 2
fi
read -r memory_limit < "$limit_path"
read -r memory_usage < "$usage_path"
if [[ ! "$memory_limit" =~ ^[0-9]+$ || ! "$memory_usage" =~ ^[0-9]+$ ]]; then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] cgroup memory counters are malformed" >&2
  exit 2
fi
if (( memory_usage >= memory_limit )); then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] cgroup has no free memory" >&2
  exit 75
fi
memory_free=$((memory_limit - memory_usage))
if (( memory_free < COMRECGC_MIN_CGROUP_FREE_BYTES )); then
  echo "[COMRECGC_HIGHMEM_GATE_FAIL] insufficient cgroup headroom free=$memory_free required=$COMRECGC_MIN_CGROUP_FREE_BYTES" >&2
  exit 75
fi

for command_file in "$COMRECGC_PROC_ROOT"/[0-9]*/cmdline; do
  [[ -r "$command_file" ]] || continue
  command_text="$(tr '\0' ' ' < "$command_file" 2>/dev/null || true)"
  if [[ "$command_text" == *"scripts/baselines/comrecgc/run_common_recourse.py"* ]]; then
    echo "[COMRECGC_HIGHMEM_GATE_FAIL] another common-recourse process is active command=$command_text" >&2
    exit 75
  fi
done

if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ "${COMMON_RECOURSE_ENGINE:-}" != "external_memory_exact_v1" \
        || "${COMRECGC_COMMON_RECOURSE_RESUME:-0}" != "1" \
        || ! -d "$OUTPUT_ROOT" \
        || -L "$OUTPUT_ROOT" \
        || ! -s "$OUTPUT_ROOT/continuation_resume_contract.json" \
        || -e "$OUTPUT_ROOT/PASS" ]]; then
    echo "[COMRECGC_HIGHMEM_GATE_FAIL] OUTPUT_ROOT must be fresh or an exact v4 resume: $OUTPUT_ROOT" >&2
    exit 2
  fi
  echo "[COMRECGC_HIGHMEM_RESUME_GATE_PASS] output=$OUTPUT_ROOT"
fi

export CUDA_VISIBLE_DEVICES=""
export DEVICE=cpu
export GPU_REQUIRED=0
echo "[COMRECGC_HIGHMEM_GATE_PASS] dataset=aids device=cpu gpu_required=false memory_limit=$memory_limit memory_usage=$memory_usage memory_free=$memory_free lock=$COMRECGC_HIGHMEM_LOCK_PATH"

exec bash "$(dirname -- "$0")/run_comrecgc_standardized_continuation.sh"
