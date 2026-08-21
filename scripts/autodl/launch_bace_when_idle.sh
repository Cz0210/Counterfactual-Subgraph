#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

export RUN_TASTEMOLNET=0
POLL_SECONDS="${AUTODL_WAIT_POLL_SECONDS:-60}"
if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECONDS" -lt 60 ]]; then
  echo "AUTODL_WAIT_POLL_SECONDS must be an integer >= 60" >&2
  exit 2
fi

while true; do
  STATUS_JSON="$("$AUTODL_PYTHON" "$SCRIPT_DIR/status.py" --project-root "$PROJECT_ROOT" --data-root "$AUTODL_DATA_ROOT" --format json)"
  NEXT_STAGE="$(printf '%s' "$STATUS_JSON" | "$AUTODL_PYTHON" -c '
import json, sys
p=json.load(sys.stdin)
s={r["stage"]:(r.get("state"),r.get("gate")) for r in p["bace_stages"]}
if s.get("B4_GNN_CALIBRATED")==("PASS","PASS") and s.get("B5_ORACLE_SMOKE",("",))[0] not in {"PASS","STARTING","RUNNING"}:
    print("oracle-smoke")
elif s.get("B3_GNN_FULL")==("PASS","PASS") and s.get("B4_GNN_CALIBRATED",("",))[0] not in {"PASS","STARTING","RUNNING"}:
    print("calibrate")
elif s.get("B2_GNN_SMOKE")==("PASS","PASS") and s.get("B3_GNN_FULL",("",))[0] not in {"PASS","STARTING","RUNNING"}:
    print("full")
elif s.get("B1_DATA_READY")==("PASS","PASS") and s.get("B2_GNN_SMOKE",("",))[0] not in {"PASS","STARTING","RUNNING"}:
    print("smoke")
else:
    print("blocked")
')"
  if [[ "$NEXT_STAGE" == "blocked" ]]; then
    echo "BACE launcher has no eligible next stage; inspect with AUTODL_PYTHON" >&2
    exit 4
  fi
  set +e
  case "$NEXT_STAGE" in
    smoke) "$SCRIPT_DIR/run_bace_gnn_smoke.sh" ;;
    full) "$SCRIPT_DIR/run_bace_gnn_full.sh" ;;
    calibrate) "$SCRIPT_DIR/run_bace_gnn_calibration.sh" ;;
    oracle-smoke) "$SCRIPT_DIR/run_bace_gnn_oracle_smoke.sh" ;;
    *) echo "Unsupported BACE stage: $NEXT_STAGE" >&2; exit 2 ;;
  esac
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "BACE $NEXT_STAGE launched; TasteMolNet remains disabled."
    exit 0
  fi
  if [[ $rc -ne 75 ]]; then
    exit "$rc"
  fi
  echo "[$(date -u +%FT%TZ)] WAITING_FOR_IDLE_GPU; next check in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done
