#!/usr/bin/env bash
# Keep one exp_run alive while allowing exactly one evidence-gated same-root
# recovery of an interrupted external-memory common-recourse child.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
INNER="$SCRIPT_DIR/run_comrecgc_standardized_continuation_cpu_highmem.sh"
VERIFY_CLI="$PROJECT_ROOT/scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py"

case "${AIDS_COMRECGC_V4_TEST_MODE:-0}" in
  0)
    if [[ -n "${AIDS_COMRECGC_V4_TEST_INNER:-}${AIDS_COMRECGC_V4_TEST_VERIFY:-}" ]]; then
      echo "[AIDS_V4_SUPERVISOR_FAIL] test hooks require explicit test mode" >&2
      exit 64
    fi
    ;;
  1)
    : "${AIDS_COMRECGC_V4_TEST_INNER:?test inner is required}"
    : "${AIDS_COMRECGC_V4_TEST_VERIFY:?test verifier is required}"
    INNER="$AIDS_COMRECGC_V4_TEST_INNER"
    VERIFY_CLI="$AIDS_COMRECGC_V4_TEST_VERIFY"
    ;;
  *)
    echo "[AIDS_V4_SUPERVISOR_FAIL] test mode must be exactly 0 or 1" >&2
    exit 64
    ;;
esac

: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
[[ "${DATASET:-}" == "aids" ]] || { echo "[AIDS_V4_SUPERVISOR_FAIL] DATASET must be aids" >&2; exit 64; }
[[ "${COMMON_RECOURSE_ENGINE:-}" == "external_memory_exact_v1" ]] || { echo "[AIDS_V4_SUPERVISOR_FAIL] external engine required" >&2; exit 64; }
[[ "${COMRECGC_COMMON_RECOURSE_RESUME:-}" == "1" ]] || { echo "[AIDS_V4_SUPERVISOR_FAIL] exact resume must be enabled" >&2; exit 64; }
[[ "${AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES:-1}" == "1" ]] || { echo "[AIDS_V4_SUPERVISOR_FAIL] resume bound must equal one" >&2; exit 64; }
[[ -x "$PYTHON" ]] || { echo "[AIDS_V4_SUPERVISOR_FAIL] Python unavailable" >&2; exit 66; }

resume_count=0
while true; do
  bash "$INNER"
  child_status=$?
  if (( child_status == 0 )); then
    echo "[AIDS_COMRECGC_REPAIR_V4_SUPERVISOR_PASS] resumes=$resume_count"
    exit 0
  fi
  if (( resume_count >= 1 )); then
    echo "[AIDS_V4_SUPERVISOR_FAIL] bounded resume exhausted status=$child_status" >&2
    exit "$child_status"
  fi
  if ! PYTHONPATH="$PROJECT_ROOT" "$PYTHON" \
      "$VERIFY_CLI" \
      --config configs/hpc.yaml \
      verify-resume-failure \
      --output-root "$OUTPUT_ROOT" \
      --exit-code "$child_status"; then
    echo "[AIDS_V4_SUPERVISOR_FAIL] failure is not a resumable process loss" >&2
    exit "$child_status"
  fi
  resume_count=$((resume_count + 1))
  echo "[AIDS_COMRECGC_REPAIR_V4_SAME_ROOT_RESUME] count=$resume_count output=$OUTPUT_ROOT"
done
