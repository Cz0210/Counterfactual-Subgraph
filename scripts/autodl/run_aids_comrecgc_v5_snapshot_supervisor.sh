#!/usr/bin/env bash
# Bounded same-root supervisor for the CPU-only promoted pair-store snapshot.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
SNAPSHOT_CLI="$PROJECT_ROOT/scripts/autodl/snapshot_aids_comrecgc_pair_store.py"

: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT:?snapshot source required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256:?source manifest hash required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT:?procfs root required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_PID:?old PID required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS:?old start ticks required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256:?old command hash required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT:?old output root required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT:?old project root required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_MIN_FREE_AFTER_BYTES:?persistent free floor required}"

[[ -z "${AIDS_COMRECGC_V5_SNAPSHOT_TEST_INNER:-}${AIDS_COMRECGC_V5_SNAPSHOT_TEST_VERIFY:-}" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] test hooks are forbidden" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_TEST_MODE:-0}" == "0" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] production test mode must be zero" >&2; exit 64; }
[[ "${GPU_REQUIRED:-}" == "0" && -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] CPU-only contract changed" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_MAX_SAME_ROOT_RESUMES:-}" == "1" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] resume bound changed" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS:-}" == "91916686" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] row count changed" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM:-}" == "64" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] vector dimension changed" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT:-}" == "1283" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] parent count changed" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT:-}" == "71642" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] candidate count changed" >&2; exit 64; }
[[ "$OUTPUT_ROOT" == /* && "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT" == /* && "$AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT" == /* ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] absolute paths required" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] invalid source hash" >&2; exit 64; }
[[ -x "$PYTHON" ]] || { echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] Python unavailable" >&2; exit 66; }

export GPU_REQUIRED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

common_args=(
  --config configs/hpc.yaml
  --source-root "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT"
  --expected-source-manifest-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256"
  --output-dir "$OUTPUT_ROOT"
  --proc-root "$AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT"
  --allowed-pid "$AIDS_COMRECGC_V5_ALLOWED_OLD_PID"
  --allowed-start-ticks "$AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS"
  --allowed-cmdline-sha256 "$AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256"
  --allowed-output-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT"
  --allowed-project-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT"
  --min-free-after-bytes "$AIDS_COMRECGC_V5_SNAPSHOT_MIN_FREE_AFTER_BYTES"
  --expected-row-count "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS"
  --expected-vector-dim "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM"
  --expected-parent-count "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT"
  --expected-candidate-count "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT"
)

child_pid=""
terminate_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap 'terminate_child; exit 143' TERM
trap 'terminate_child; exit 130' INT

run_snapshot() {
  PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$SNAPSHOT_CLI" "${common_args[@]}" "$@" &
  child_pid=$!
  wait "$child_pid"
  local status=$?
  child_pid=""
  return "$status"
}

resume_count=0
while true; do
  extra=()
  (( resume_count > 0 )) && extra+=(--resume)
  run_snapshot "${extra[@]}"
  status=$?
  if (( status == 0 )); then
    break
  fi
  if (( resume_count >= 1 )) || (( status != 137 && status != 143 )); then
    echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] status=$status resumes=$resume_count" >&2
    exit "$status"
  fi
  resume_count=$((resume_count + 1))
  echo "[AIDS_COMRECGC_V5_SNAPSHOT_SAME_ROOT_RESUME] count=$resume_count output=$OUTPUT_ROOT"
done

run_snapshot --validate-only || {
  status=$?
  echo "[AIDS_V5_SNAPSHOT_SUPERVISOR_FAIL] terminal validation failed status=$status" >&2
  exit "$status"
}
echo "[AIDS_COMRECGC_V5_SNAPSHOT_SUPERVISOR_PASS] resumes=$resume_count output=$OUTPUT_ROOT"
