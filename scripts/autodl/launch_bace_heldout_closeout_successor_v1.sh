#!/usr/bin/env bash
# Launch one persistent, dataset-specific BACE held-out closeout successor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "${ALLOW_BACE_HELDOUT_CLOSEOUT_SUCCESSOR:-0}" == "1" ]] || {
  echo "ALLOW_BACE_HELDOUT_CLOSEOUT_SUCCESSOR=1 is required" >&2
  exit 64
}
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || {
  echo "RUN_GNN_ABLATION must remain 0" >&2
  exit 64
}

MATRIX_ROOT=${BACE_HELDOUT_MATRIX_ROOT:-$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1}
SOURCE_ROOT=${BACE_HELDOUT_SOURCE_ROOT:-$MATRIX_ROOT/repairs/bace_baseline_merge_closeout_0e5d31f_20260901T114200Z}
OLD_CONTROL=${BACE_HELDOUT_OLD_CONTROL_ROOT:-$AUTODL_CONTROL_ROOT/fast16_bace_test_closeout/bace-test-closeout-016005e-20260901T120405Z}
SELECTION_RECEIPT=${BACE_HELDOUT_SELECTION_RECEIPT:-$OLD_CONTROL/selection_adoption_receipt.json}
EXPECTED_RECEIPT_SHA256=${BACE_HELDOUT_SELECTION_RECEIPT_SHA256:-3c303375ccace046c27b0e5c4aa2321a0fa7f2893e77d7cf39c53df062fcd3e5}
CONTROL_BASE=${BACE_HELDOUT_CONTROL_BASE:-$AUTODL_CONTROL_ROOT/fast16_bace_test_closeout}
AUTHORITY_STATE=${MATRIX_AUTHORITY_STATE:-$AUTODL_CONTROL_ROOT/fast16_matrix_authority/state.json}
AUTHORITY_LOCK=${MATRIX_AUTHORITY_LOCK:-$AUTODL_CONTROL_ROOT/fast16_matrix_authority/publish.lock}
GNN_CHECKPOINT=${BACE_GNN_CHECKPOINT:-$AUTODL_RUNTIME_ROOT/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689}
TEST_SPLIT=${BACE_TEST_SPLIT:-$BACE_SPLIT_ROOT/test.csv}
STEP0_ROOT=${AUTODL_STEP0_ROOT:-$AUTODL_DATA_ROOT/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project}
MOLCLR_ROOT=${MOLCLR_ROOT:-$STEP0_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
GPU_INDEX=${BACE_HELDOUT_GPU_INDEX:-0}
POLL_SECONDS=${SCHEDULER_POLL_SECONDS:-30}

for required in "$SOURCE_ROOT" "$GNN_CHECKPOINT" "$MOLCLR_ROOT"; do
  autodl_require_dir "$required"
done
for required in "$SELECTION_RECEIPT" "$TEST_SPLIT" "$MOLCLR_CHECKPOINT" "$AUTHORITY_STATE"; do
  autodl_require_file "$required"
done

RUN_TAG=${BACE_HELDOUT_RUN_TAG:-$($AUTODL_PYTHON -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])')}
CONTROLLER_ID=${BACE_HELDOUT_CONTROLLER_ID:-bace-heldout-closeout-successor-$RUN_TAG}
CONTROL_DIR=${BACE_HELDOUT_CONTROL_DIR:-$CONTROL_BASE/$CONTROLLER_ID}
OUTPUT_ROOT=${BACE_HELDOUT_OUTPUT_ROOT:-$MATRIX_ROOT/repairs/$CONTROLLER_ID}

[[ "$CONTROLLER_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "unsafe BACE held-out controller id: $CONTROLLER_ID" >&2
  exit 64
}
[[ ! -e "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || {
  echo "BACE held-out control directory must be fresh: $CONTROL_DIR" >&2
  exit 73
}
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "BACE held-out output root must be fresh: $OUTPUT_ROOT" >&2
  exit 73
}

COMMON_ARGS=(
  --project-root "$PROJECT_ROOT"
  --python "$AUTODL_PYTHON"
  --runtime-root "$AUTODL_RUNTIME_ROOT"
  --controller-id "$CONTROLLER_ID"
  --control-dir "$CONTROL_DIR"
  --output-root "$OUTPUT_ROOT"
  --source-root "$SOURCE_ROOT"
  --selection-adoption-receipt "$SELECTION_RECEIPT"
  --expected-selection-receipt-sha256 "$EXPECTED_RECEIPT_SHA256"
  --gnn-checkpoint "$GNN_CHECKPOINT"
  --test-split "$TEST_SPLIT"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --matrix-authority-state "$AUTHORITY_STATE"
  --matrix-authority-lock "$AUTHORITY_LOCK"
  --gpu-index "$GPU_INDEX"
  --min-free-memory-mb "${AUTODL_MIN_FREE_MEMORY_MB:-16000}"
  --poll-seconds "$POLL_SECONDS"
)

export RUN_GNN_ABLATION=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
cd "$PROJECT_ROOT"
"$AUTODL_PYTHON" "$SCRIPT_DIR/run_bace_heldout_closeout_successor_v1.py" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  preflight "${COMMON_ARGS[@]}" >/dev/null

mkdir -p "$CONTROL_DIR"
RUN_COMMAND=(
  "$AUTODL_PYTHON"
  "$SCRIPT_DIR/run_bace_heldout_closeout_successor_v1.py"
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  run
  "${COMMON_ARGS[@]}"
)
nohup setsid "${RUN_COMMAND[@]}" >"$CONTROL_DIR/controller.log" 2>&1 </dev/null &
CONTROLLER_PID=$!
PID_TEMP="$CONTROL_DIR/.controller.pid.$$"
printf '%s\n' "$CONTROLLER_PID" >"$PID_TEMP"
mv "$PID_TEMP" "$CONTROL_DIR/controller.pid"

for _ in $(seq 1 30); do
  [[ -s "$CONTROL_DIR/heartbeat.json" ]] && break
  if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    echo "BACE held-out successor exited during launch; see $CONTROL_DIR/controller.log" >&2
    exit 1
  fi
  sleep 1
done
[[ -s "$CONTROL_DIR/heartbeat.json" ]] || {
  echo "BACE held-out successor did not publish a heartbeat" >&2
  exit 1
}

printf 'controller_id=%s\ncontroller_pid=%s\ncontroller_root=%s\noutput_root=%s\ngpu_index=%s\n' \
  "$CONTROLLER_ID" "$CONTROLLER_PID" "$CONTROL_DIR" "$OUTPUT_ROOT" "$GPU_INDEX"
printf 'status_command=%q %q --config configs/hpc.yaml status --control-dir %q\n' \
  "$AUTODL_PYTHON" "$SCRIPT_DIR/run_bace_heldout_closeout_successor_v1.py" "$CONTROL_DIR"
