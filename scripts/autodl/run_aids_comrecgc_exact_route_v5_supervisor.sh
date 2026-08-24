#!/usr/bin/env bash
# Bounded same-root supervisor for the fresh AIDS exact-Cartesian v5 route.
# The old repair-v4 output is only a read-only source and is never resumed here.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
INNER="$SCRIPT_DIR/run_comrecgc_standardized_continuation_cpu_highmem.sh"
VERIFY_CLI="$PROJECT_ROOT/scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py"

: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT:?automatic pair-store root is required}"
: "${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT:?old read-only owner root is required}"
: "${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT:?closed-chunk fallback checkpoint is required}"
: "${COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT:?fresh local vector-cache root is required}"
: "${COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK:?local cache allocation lock is required}"
: "${COMRECGC_EXTERNAL_ROUTE_LOCK:?route-wide scratch lock is required}"
: "${COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT:?procfs root is required}"

[[ "${DATASET:-}" == "aids" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] DATASET must be aids" >&2; exit 64; }
[[ "${DEVICE:-}" == "cpu" && "${GPU_REQUIRED:-}" == "0" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] CPU-only contract changed" >&2; exit 64; }
[[ "${COMMON_RECOURSE_ENGINE:-}" == "external_memory_exact_v1" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] exact external engine required" >&2; exit 64; }
[[ "${COMRECGC_COMMON_RECOURSE_RESUME:-}" == "1" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] same-root resume must be enabled" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES:-1}" == "1" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] resume bound must equal one" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_MAX_RSS_GB:-}" == "96" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] RSS contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE:-}" == "8" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] DBSCAN block contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS:-}" == "1" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] checkpoint interval contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE:-}" == "all_core_one_component_adaptive_anchor_v1" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] adaptive exact shortcut required" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT:-}" == "3" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] seed-count contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP:-}" == "4096" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] failure-cap contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE:-}" == "65536" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] shortcut block contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES:-}" == "0" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] dense fallback must be disabled" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE:-}" == "65536" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] summary block contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXPECTED_SKLEARN_VERSION:-}" == "1.7.2" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] sklearn contract changed" >&2; exit 64; }
[[ "${COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB:-}" == "3" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] local floor contract changed" >&2; exit 64; }
[[ -z "${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] explicit terminal source bypasses automatic priority gate" >&2; exit 64; }
[[ -z "${AIDS_COMRECGC_V5_TEST_INNER:-}${AIDS_COMRECGC_V5_TEST_VERIFY:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] production test hooks are forbidden" >&2; exit 64; }
[[ -x "$PYTHON" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] Python unavailable" >&2; exit 66; }
for path in \
  "$OUTPUT_ROOT" \
  "$COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT" \
  "$COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" \
  "$COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT" \
  "$COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT" \
  "$COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK" \
  "$COMRECGC_EXTERNAL_ROUTE_LOCK" \
  "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT"; do
  [[ "$path" == /* ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] absolute path required: $path" >&2; exit 64; }
done
[[ -d "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" && ! -L "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] physical procfs root required" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT" == /root/autodl-tmp/* ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] vector cache must stay on local AutoDL scratch" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK" == /root/autodl-tmp/* && "$COMRECGC_EXTERNAL_ROUTE_LOCK" == /root/autodl-tmp/* ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] scratch locks must stay on local AutoDL storage" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK" != "$COMRECGC_EXTERNAL_ROUTE_LOCK" && "$COMRECGC_EXTERNAL_ROUTE_LOCK" != "${COMRECGC_HIGHMEM_LOCK_PATH:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] route/cache/highmem locks must be distinct" >&2; exit 64; }

flock_bin="${COMRECGC_FLOCK_BIN:-}"
if [[ -z "$flock_bin" ]]; then
  flock_bin="$(command -v flock || true)"
fi
[[ "$flock_bin" == /* && -x "$flock_bin" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] flock unavailable" >&2; exit 66; }
mkdir -p -- "$(dirname -- "$COMRECGC_EXTERNAL_ROUTE_LOCK")"
exec 8>"$COMRECGC_EXTERNAL_ROUTE_LOCK"
"$flock_bin" --exclusive 8

resume_count=0
while true; do
  bash "$INNER"
  child_status=$?
  if (( child_status == 0 )); then
    echo "[AIDS_COMRECGC_EXACT_ROUTE_V5_SUPERVISOR_PASS] resumes=$resume_count"
    exit 0
  fi
  if (( resume_count >= 1 )); then
    echo "[AIDS_V5_SUPERVISOR_FAIL] bounded resume exhausted status=$child_status" >&2
    exit "$child_status"
  fi
  if ! PYTHONPATH="$PROJECT_ROOT" "$PYTHON" \
      "$VERIFY_CLI" \
      --config configs/hpc.yaml \
      verify-resume-failure \
      --output-root "$OUTPUT_ROOT" \
      --exit-code "$child_status"; then
    echo "[AIDS_V5_SUPERVISOR_FAIL] failure is not a resumable process loss" >&2
    exit "$child_status"
  fi
  resume_count=$((resume_count + 1))
  echo "[AIDS_COMRECGC_EXACT_ROUTE_V5_SAME_ROOT_RESUME] count=$resume_count output=$OUTPUT_ROOT"
done
