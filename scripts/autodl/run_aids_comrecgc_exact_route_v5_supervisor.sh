#!/usr/bin/env bash
# Bounded same-root supervisor for the fresh AIDS exact-Cartesian v5 route.
# The old repair-v4 output is only a read-only source and is never resumed here.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
INNER="$SCRIPT_DIR/run_comrecgc_standardized_continuation.sh"
VERIFY_CLI="$PROJECT_ROOT/scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py"
PROCESS_GATE_CLI="$PROJECT_ROOT/scripts/autodl/verify_aids_comrecgc_v5_process_set.py"
ADOPTION_CLI="$PROJECT_ROOT/scripts/autodl/adopt_aids_comrecgc_v5_snapshot.py"
HANDOVER_MODULE="src.utils.aids_comrecgc_v5_lock_handover"
SCIENCE_EXEC_MODULE="src.utils.aids_comrecgc_v5_science_exec"

: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT:?automatic pair-store root is required}"
: "${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT:?old read-only owner root is required}"
: "${COMRECGC_EXTERNAL_ROUTE_LOCK:?route-wide scratch lock is required}"
: "${COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT:?procfs root is required}"
: "${COMRECGC_CGROUP_MEMORY_ROOT:?cgroup-v1 memory root is required}"
: "${AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES:?minimum cgroup headroom is required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_PID:?allowed old read-only PID is required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS:?allowed old process start ticks are required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256:?allowed old process command hash is required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT:?allowed old output root is required}"
: "${AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT:?allowed old project root is required}"
: "${COMRECGC_HIGHMEM_LOCK_PATH:?global high-memory lock is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_ROOT:?physical snapshot root is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT:?snapshot adoption gate is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST:?snapshot owner manifest is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST_SHA256:?snapshot owner manifest hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE:?snapshot owner task gate is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE_SHA256:?snapshot owner task gate hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_MANIFEST_SHA256:?snapshot manifest hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_DBSCAN_SHA256:?snapshot DBSCAN hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_PAIR_MANIFEST_SHA256:?snapshot pair manifest hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_PAIRS_SHA256:?snapshot pair array hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_VECTORS_SHA256:?snapshot vector array hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT:?snapshot source root is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256:?snapshot source hash is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT:?snapshot procfs root is required}"
: "${AIDS_COMRECGC_V5_SNAPSHOT_MIN_FREE_AFTER_BYTES:?snapshot free floor is required}"

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
[[ "${COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL:-}" == "1" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] promoted terminal source is mandatory" >&2; exit 64; }
[[ -z "${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] explicit terminal source bypasses automatic priority gate" >&2; exit 64; }
[[ -z "${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT:-}${COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT:-}${COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK:-}${COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] promoted-final route forbids chunk/cache fallback" >&2; exit 64; }
[[ -z "${AIDS_COMRECGC_V5_TEST_INNER:-}${AIDS_COMRECGC_V5_TEST_VERIFY:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] production test hooks are forbidden" >&2; exit 64; }
[[ -x "$PYTHON" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] Python unavailable" >&2; exit 66; }
for path in \
  "$OUTPUT_ROOT" \
  "$COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT" \
  "$COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" \
  "$COMRECGC_EXTERNAL_ROUTE_LOCK" \
  "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" \
  "$COMRECGC_CGROUP_MEMORY_ROOT" \
  "$AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT" \
  "$AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_ROOT" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE" \
  "$COMRECGC_HIGHMEM_LOCK_PATH"; do
  [[ "$path" == /* ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] absolute path required: $path" >&2; exit 64; }
done
[[ -d "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" && ! -L "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] physical procfs root required" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" == "$COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] terminal owner must be the exact pair-store root" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_ROUTE_LOCK" == /root/autodl-tmp/* ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] scratch lock must stay on local AutoDL storage" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_ROUTE_LOCK" != "${COMRECGC_HIGHMEM_LOCK_PATH:-}" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] route/highmem locks must be distinct" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES" =~ ^[0-9]+$ && "$AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES" -ge 137438953472 ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] cgroup headroom must be at least 128 GiB" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_ALLOWED_OLD_PID" =~ ^[1-9][0-9]*$ ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] invalid old PID" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS" =~ ^[1-9][0-9]*$ ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] invalid old start ticks" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] invalid old command SHA256" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] invalid snapshot source SHA256" >&2; exit 64; }
for value in \
  "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST_SHA256" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE_SHA256" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_MANIFEST_SHA256" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_DBSCAN_SHA256" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_PAIR_MANIFEST_SHA256" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_PAIRS_SHA256" \
  "$AIDS_COMRECGC_V5_SNAPSHOT_VECTORS_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] invalid adopted snapshot SHA256" >&2; exit 64; }
done
[[ "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS:-}" == "91916686" \
   && "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM:-}" == "64" \
   && "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT:-}" == "1283" \
   && "${AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT:-}" == "71642" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] snapshot scientific dimensions changed" >&2; exit 64; }
[[ "$AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT" == "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] snapshot procfs root changed" >&2; exit 64; }
[[ "$COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT" == "$AIDS_COMRECGC_V5_SNAPSHOT_ROOT/pair_store" \
   && "$COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" == "$AIDS_COMRECGC_V5_SNAPSHOT_ROOT/pair_store" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] science source is not the exact physical snapshot" >&2; exit 64; }

export DEVICE=cpu
export GPU_REQUIRED=0
export CUDA_VISIBLE_DEVICES=""

# The adoption task publishes PASS-last, but science independently reopens its
# exact owner PASS plus the full source/destination/DBSCAN closure.  This call
# is read-only and never copies or hardlinks the 25 GB snapshot.
if ! PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$ADOPTION_CLI" \
    --config configs/hpc.yaml \
    --output-dir "$AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT" \
    --proc-root "$AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT" \
    --owner-manifest "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST" \
    --owner-manifest-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST_SHA256" \
    --owner-task-gate "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE" \
    --owner-task-gate-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE_SHA256" \
    --snapshot-root "$AIDS_COMRECGC_V5_SNAPSHOT_ROOT" \
    --snapshot-manifest-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_MANIFEST_SHA256" \
    --dbscan-contract-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_DBSCAN_SHA256" \
    --pair-store-manifest-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_PAIR_MANIFEST_SHA256" \
    --pairs-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_PAIRS_SHA256" \
    --vectors-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_VECTORS_SHA256" \
    --source-root "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT" \
    --source-manifest-sha256 "$AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256" \
    --allowed-pid "$AIDS_COMRECGC_V5_ALLOWED_OLD_PID" \
    --allowed-start-ticks "$AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS" \
    --allowed-cmdline-sha256 "$AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256" \
    --allowed-output-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT" \
    --allowed-project-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT" \
    --expected-row-count "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS" \
    --expected-vector-dim "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM" \
    --expected-parent-count "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT" \
    --expected-candidate-count "$AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT" \
    --validate-only; then
  echo "[AIDS_V5_SUPERVISOR_FAIL] snapshot adoption closure validation failed" >&2
  exit 75
fi

check_cgroup_headroom() {
  local quiet="${1:-0}"
  local limit_path="$COMRECGC_CGROUP_MEMORY_ROOT/memory.limit_in_bytes"
  local usage_path="$COMRECGC_CGROUP_MEMORY_ROOT/memory.usage_in_bytes"
  local memory_limit memory_usage memory_free
  [[ -r "$limit_path" && -r "$usage_path" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] cgroup-v1 counters unavailable" >&2; return 75; }
  read -r memory_limit < "$limit_path"
  read -r memory_usage < "$usage_path"
  [[ "$memory_limit" =~ ^[0-9]+$ && "$memory_usage" =~ ^[0-9]+$ && "$memory_usage" -lt "$memory_limit" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] cgroup-v1 counters invalid" >&2; return 75; }
  memory_free=$((memory_limit - memory_usage))
  (( memory_free >= AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES )) || { echo "[AIDS_V5_SUPERVISOR_FAIL] insufficient cgroup headroom free=$memory_free required=$AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES" >&2; return 75; }
  if [[ "$quiet" != "1" ]]; then
    echo "[AIDS_COMRECGC_EXACT_ROUTE_V5_MEMORY_GATE_PASS] limit=$memory_limit usage=$memory_usage free=$memory_free"
  fi
}

flock_bin="${COMRECGC_FLOCK_BIN:-}"
if [[ -z "$flock_bin" ]]; then
  flock_bin="$(command -v flock || true)"
fi
[[ "$flock_bin" == /* && -x "$flock_bin" ]] || { echo "[AIDS_V5_SUPERVISOR_FAIL] flock unavailable" >&2; exit 66; }
mkdir -p -- "$(dirname -- "$COMRECGC_EXTERNAL_ROUTE_LOCK")"
exec 8>"$COMRECGC_EXTERNAL_ROUTE_LOCK"
"$flock_bin" --exclusive 8

handover_state="$(mktemp "${COMRECGC_EXTERNAL_ROUTE_LOCK}.handover.XXXXXX")"
handover_pid=""
handover_start_ticks=""
science_pid=""
science_start_ticks=""

proc_generation() {
  local pid="$1" raw remainder
  [[ "$pid" =~ ^[1-9][0-9]*$ && -r "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT/$pid/stat" ]] || return 1
  IFS= read -r raw < "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT/$pid/stat" || return 1
  remainder="${raw##*) }"
  read -r -a fields <<< "$remainder"
  (( ${#fields[@]} > 19 )) || return 1
  printf '%s %s\n' "${fields[0]}" "${fields[19]}"
}

same_live_generation() {
  local pid="$1" expected_ticks="$2" state ticks
  read -r state ticks < <(proc_generation "$pid") || return 1
  [[ "$state" != "Z" && "$ticks" == "$expected_ticks" ]]
}

same_healthy_helper_generation() {
  local pid="$1" expected_ticks="$2" state ticks
  read -r state ticks < <(proc_generation "$pid") || return 1
  [[ "$state" =~ ^[RSDI]$ && "$ticks" == "$expected_ticks" ]]
}

capture_start_ticks() {
  local pid="$1" attempt state ticks
  for attempt in $(seq 1 100); do
    if read -r state ticks < <(proc_generation "$pid") && [[ "$state" != "Z" ]]; then
      printf '%s\n' "$ticks"
      return 0
    fi
    sleep 0.01
  done
  return 1
}

science_group_ready() {
  local pid="$1" expected_ticks="$2" raw remainder
  [[ -r "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT/$pid/stat" ]] || return 1
  IFS= read -r raw < "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT/$pid/stat" || return 1
  remainder="${raw##*) }"
  read -r -a fields <<< "$remainder"
  (( ${#fields[@]} > 19 )) || return 1
  [[ "${fields[0]}" != "Z" \
     && "${fields[19]}" == "$expected_ticks" \
     && "${fields[2]}" == "$pid" \
     && "${fields[3]}" == "$pid" ]]
}

terminate_science_group() {
  if [[ -n "$science_pid" && -n "$science_start_ticks" ]] \
      && same_live_generation "$science_pid" "$science_start_ticks"; then
    kill -TERM -- "-$science_pid" 2>/dev/null || true
  fi
}

cleanup_v5_supervisor() {
  local status=$? current_state current_ticks
  terminate_science_group
  if [[ -n "$science_pid" ]]; then
    wait "$science_pid" 2>/dev/null || true
  fi
  if [[ -n "$handover_pid" && -n "$handover_start_ticks" ]] \
      && read -r current_state current_ticks < <(proc_generation "$handover_pid") \
      && [[ "$current_state" != "Z" && "$current_ticks" == "$handover_start_ticks" ]]; then
    kill -TERM "$handover_pid" 2>/dev/null || true
  fi
  if [[ -n "$handover_pid" ]]; then
    wait "$handover_pid" 2>/dev/null || true
  fi
  rm -f -- "$handover_state"
  return "$status"
}
trap cleanup_v5_supervisor EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

PYTHONPATH="$PROJECT_ROOT" "$PYTHON" -m "$HANDOVER_MODULE" \
  --lock-path "$COMRECGC_HIGHMEM_LOCK_PATH" \
  --state-path "$handover_state" \
  --supervisor-pid "$$" \
  --proc-root "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" \
  --poll-seconds 1 &
handover_pid=$!
handover_start_ticks="$(capture_start_ticks "$handover_pid")" || {
  echo "[AIDS_V5_SUPERVISOR_FAIL] cannot bind handover helper generation" >&2
  exit 75
}
handover_ready=0
for _attempt in $(seq 1 200); do
  if [[ -s "$handover_state" ]] && grep -Eq '"status": "(QUEUED|ACQUIRED)"' "$handover_state"; then
    handover_ready=1
    break
  fi
  if ! same_live_generation "$handover_pid" "$handover_start_ticks"; then
    break
  fi
  sleep 0.01
done
if (( handover_ready != 1 )) || ! same_healthy_helper_generation "$handover_pid" "$handover_start_ticks"; then
  echo "[AIDS_V5_SUPERVISOR_FAIL] global high-memory handover did not queue" >&2
  exit 75
fi
echo "[AIDS_COMRECGC_EXACT_ROUTE_V5_HIGHMEM_HANDOVER_QUEUED] helper_pid=$handover_pid lock=$COMRECGC_HIGHMEM_LOCK_PATH"

resume_count=0
while true; do
  check_cgroup_headroom || exit $?
  if ! PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$PROCESS_GATE_CLI" \
      --config configs/hpc.yaml \
      --proc-root "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" \
      --allowed-pid "$AIDS_COMRECGC_V5_ALLOWED_OLD_PID" \
      --allowed-start-ticks "$AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS" \
      --allowed-cmdline-sha256 "$AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256" \
      --allowed-output-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT" \
      --allowed-project-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT"; then
    echo "[AIDS_V5_SUPERVISOR_FAIL] common-recourse process set changed" >&2
    exit 75
  fi
  PYTHONPATH="$PROJECT_ROOT" "$PYTHON" -m "$SCIENCE_EXEC_MODULE" \
    --project-root "$PROJECT_ROOT" \
    --script "$INNER" &
  science_pid=$!
  science_start_ticks="$(capture_start_ticks "$science_pid")" || {
    echo "[AIDS_V5_SUPERVISOR_FAIL] cannot bind science child generation" >&2
    exit 75
  }
  science_group_bound=0
  for _attempt in $(seq 1 200); do
    if science_group_ready "$science_pid" "$science_start_ticks"; then
      science_group_bound=1
      break
    fi
    if ! same_live_generation "$science_pid" "$science_start_ticks"; then
      break
    fi
    sleep 0.01
  done
  if (( science_group_bound != 1 )); then
    echo "[AIDS_V5_SUPERVISOR_FAIL] science process group was not established" >&2
    exit 75
  fi
  while same_live_generation "$science_pid" "$science_start_ticks"; do
    if ! check_cgroup_headroom 1; then
      echo "[AIDS_V5_SUPERVISOR_FAIL] mid-run cgroup headroom gate failed" >&2
      terminate_science_group
      wait "$science_pid" 2>/dev/null || true
      science_pid=""
      science_start_ticks=""
      exit 75
    fi
    if ! PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$PROCESS_GATE_CLI" \
        --config configs/hpc.yaml \
        --proc-root "$COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT" \
        --allowed-pid "$AIDS_COMRECGC_V5_ALLOWED_OLD_PID" \
        --allowed-start-ticks "$AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS" \
        --allowed-cmdline-sha256 "$AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256" \
        --allowed-output-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT" \
        --allowed-project-root "$AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT" \
        --allowed-route-root-pid "$science_pid" \
        --allowed-route-root-start-ticks "$science_start_ticks" \
        --allowed-route-output-root "$OUTPUT_ROOT/common_recourse" \
        --allowed-route-project-root "$PROJECT_ROOT" \
        --quiet; then
      echo "[AIDS_V5_SUPERVISOR_FAIL] mid-run common-recourse process set changed" >&2
      terminate_science_group
      wait "$science_pid" 2>/dev/null || true
      science_pid=""
      science_start_ticks=""
      exit 75
    fi
    if ! same_healthy_helper_generation "$handover_pid" "$handover_start_ticks" \
        || [[ ! -s "$handover_state" ]] \
        || ! grep -Eq '"status": "(QUEUED|ACQUIRED)"' "$handover_state"; then
      echo "[AIDS_V5_SUPERVISOR_FAIL] global high-memory handover helper failed" >&2
      terminate_science_group
      wait "$science_pid" 2>/dev/null || true
      science_pid=""
      science_start_ticks=""
      exit 75
    fi
    sleep 1
  done
  wait "$science_pid"
  child_status=$?
  science_pid=""
  science_start_ticks=""
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
