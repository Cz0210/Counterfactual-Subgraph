#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "${AUTO_TERMINATE_UNCONTROLLED_CHILDREN:-0}" == "0" ]] \
  || { echo "automatic process termination is forbidden" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] \
  || { echo "GNN backbone ablation is disabled" >&2; exit 64; }
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
export RUN_GNN_ABLATION=0

inside_gpu_lock=0
if [[ "${1:-}" == "--inside-gpu-lock" ]]; then
  inside_gpu_lock=1
  shift
fi

if [[ "$inside_gpu_lock" == "0" ]]; then
  [[ "${RUN_TASTEMOLNET:-0}" == "1" ]] \
    || { echo "RUN_TASTEMOLNET=1 is required" >&2; exit 64; }
  [[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-0}" == "1" ]] \
    || { echo "Taste research compute is not authorized" >&2; exit 64; }
  [[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-1}" == "0" ]] \
    || { echo "Taste data redistribution must remain forbidden" >&2; exit 64; }
  [[ "${AUTO_TERMINATE_UNCONTROLLED_CHILDREN:-0}" == "0" ]] \
    || { echo "automatic process termination is forbidden" >&2; exit 64; }
  [[ "${RUN_GNN_ABLATION:-0}" == "0" ]] \
    || { echo "GNN backbone ablation is disabled" >&2; exit 64; }
  : "${TASTEMOLNET_MAIN_V2_CONTROLLER_ID:?set the managed-v2 Taste controller ID}"

  gpu_json="$(
    "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
      --project-root "$PROJECT_ROOT" \
      --data-root "$AUTODL_DATA_ROOT" \
      --max-gpus 4 \
      --gpu-hard-limit 4 \
      --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
      --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
      --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" \
      --format json
  )"
  gpu_uuid="$(
    printf '%s' "$gpu_json" | "$AUTODL_PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
rows = [row for row in payload["gpus"] if row["index"] == 1 and row["selected"] and row["stable_idle"]]
if len(rows) != 1:
    raise SystemExit(75)
print(rows[0]["uuid"])
'
  )" || {
    rc=$?
    [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU1" >&2
    exit "$rc"
  }
  [[ "$gpu_uuid" == GPU-* ]] \
    || { echo "Taste NeuroSED GPU1 UUID binding failed" >&2; exit 64; }
  exec "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_lock.py" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$AUTODL_DATA_ROOT" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    run \
    --gpu-index 1 \
    --gpu-uuid "$gpu_uuid" \
    --run-id "${TASTEMOLNET_MAIN_V2_CONTROLLER_ID}:TASTE_GCF_NEUROSED" \
    -- \
    bash "$0" --inside-gpu-lock \
      "$gpu_uuid" \
      "$TASTEMOLNET_MAIN_V2_CONTROLLER_ID" \
      "$AUTODL_DATA_ROOT" \
      "$AUTODL_RUNTIME_ROOT" \
      "$AUTODL_CONTROL_ROOT" \
      "$TASTEMOLNET_SPLIT_ROOT" \
      "${TASTEMOLNET_NEUROSED_FINAL_ROOT:-}"
fi

[[ $# -eq 7 ]] || { echo "internal Taste NeuroSED launch arguments changed" >&2; exit 64; }
gpu_uuid="$1"
controller_id="$2"
AUTODL_DATA_ROOT="$3"
AUTODL_RUNTIME_ROOT="$4"
AUTODL_CONTROL_ROOT="$5"
TASTEMOLNET_SPLIT_ROOT="$6"
requested_final_root="$7"
export AUTODL_DATA_ROOT AUTODL_RUNTIME_ROOT AUTODL_CONTROL_ROOT TASTEMOLNET_SPLIT_ROOT
export RUN_TASTEMOLNET=1
export TASTE_RESEARCH_COMPUTE_ALLOWED=1
export TASTE_DATA_REDISTRIBUTION_ALLOWED=0
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
export RUN_GNN_ABLATION=0

[[ "${AUTODL_PHYSICAL_GPU_INDEX:-}" == "1" ]] \
  || { echo "Taste NeuroSED is not inside the physical GPU1 lock" >&2; exit 64; }
[[ "${AUTODL_PHYSICAL_GPU_UUID:-}" == "$gpu_uuid" ]] \
  || { echo "Taste NeuroSED GPU UUID changed after lock acquisition" >&2; exit 64; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "1" ]] \
  || { echo "Taste NeuroSED CUDA visibility changed" >&2; exit 64; }

train_csv="$TASTEMOLNET_SPLIT_ROOT/train.csv"
validation_csv="$TASTEMOLNET_SPLIT_ROOT/validation.csv"
split_manifest="$TASTEMOLNET_SPLIT_ROOT/split_manifest.json"
neurosed_config="$PROJECT_ROOT/configs/autodl/tastemolnet_neurosed_v1.yaml"
for required in "$train_csv" "$validation_csv" "$split_manifest" "$neurosed_config"; do
  autodl_require_file "$required"
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
final_parent="$AUTODL_RUNTIME_ROOT/outputs/autodl/tastemolnet/gcfexplainer/neurosed/seed7"
mkdir -p "$final_parent"
final_root="${requested_final_root:-$final_parent/$stamp}"
[[ "$final_root" == "$final_parent"/* ]] \
  || { echo "Taste NeuroSED final root escaped its fresh namespace" >&2; exit 64; }
[[ ! -e "$final_root" && ! -L "$final_root" ]] \
  || { echo "Taste NeuroSED final root must be absent" >&2; exit 64; }

stage_root="$AUTODL_CONTROL_ROOT/tastemolnet-main-v2/stages/TASTE_GCF_NEUROSED"
mkdir -p "$stage_root"
attempt_id="$($AUTODL_PYTHON -c 'import uuid; print(uuid.uuid4())')"
git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
git_tree="$(git -C "$PROJECT_ROOT" rev-parse HEAD^{tree})"
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all)" ]] \
  || { echo "Taste NeuroSED requires a clean immutable execution worktree" >&2; exit 64; }
config_hash="$(sha256sum "$neurosed_config" | awk '{print $1}')"
train_hash="$(sha256sum "$train_csv" | awk '{print $1}')"
validation_hash="$(sha256sum "$validation_csv" | awk '{print $1}')"
split_hash="$(sha256sum "$split_manifest" | awk '{print $1}')"

"$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/managed_worker_v2.py" \
  --stage-root "$stage_root" \
  --controller-id "$controller_id" \
  --task-id TASTE_GCF_NEUROSED \
  --git-commit "$git_commit" \
  --config-hash "$config_hash" \
  --input-hash "train_csv=$train_hash" \
  --input-hash "validation_csv=$validation_hash" \
  --input-hash "preparation_split_manifest=$split_hash" \
  --attempt-id "$attempt_id" \
  --cwd "$PROJECT_ROOT" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  -- \
  "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/train_tastemolnet_neurosed.py" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --neurosed-config "$neurosed_config" \
    --train-csv "$train_csv" \
    --validation-csv "$validation_csv" \
    --preparation-split-manifest "$split_manifest" \
    --execution-git-commit "$git_commit" \
    --execution-git-tree "$git_tree" \
    --device cuda:0

shopt -s nullglob
sealed_paths=("$stage_root/attempts/$attempt_id"/worker_staging/*/SEALED.json)
shopt -u nullglob
[[ ${#sealed_paths[@]} -eq 1 ]] \
  || { echo "Taste NeuroSED worker did not produce exactly one SEALED artifact" >&2; exit 75; }

"$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/verify_tastemolnet_neurosed.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  --sealed "${sealed_paths[0]}" \
  --final-path "$final_root" \
  --expected-attempt-id "$attempt_id" \
  --require-cuda-tolerance

echo "neurosed_attempt_id=$attempt_id"
echo "neurosed_gpu=1"
echo "neurosed_gpu_uuid=$gpu_uuid"
echo "neurosed_root=$final_root"
echo "neurosed_checkpoint=$final_root/artifacts/best.pt"
