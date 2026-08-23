#!/usr/bin/env bash
set -euo pipefail

# Run exactly one fresh same-GPU legacy/ordered-v2 replay.  Quick 50/100 gates
# are diagnostic guards only.  M=500 is formal equivalence evidence but still
# cannot authorize the 50k optimized route without the independent M=1000 and
# benchmark gates.
: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${GCF_OFFICIAL_ROOT:?set GCF_OFFICIAL_ROOT}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_NEUROSED_CHECKPOINT:?set BACE_NEUROSED_CHECKPOINT}"
: "${BACE_NEUROSED_MANIFEST:?set BACE_NEUROSED_MANIFEST}"
: "${BACE_GCF_REPLAY_OUTPUT:?set a fresh persistent output root}"
: "${BACE_GCF_REPLAY_BUDGET:?set 50, 100, 500, or 1000}"
: "${BACE_GCF_REPLAY_CLASS:?set quick or formal}"
: "${AUTODL_PHYSICAL_GPU_UUID:?launch through the UUID-lock AutoDL runner}"

budget="$BACE_GCF_REPLAY_BUDGET"
replay_class="$BACE_GCF_REPLAY_CLASS"
case "$replay_class:$budget" in
  quick:50|quick:100|formal:500|formal:1000) ;;
  *)
    echo "invalid replay class/budget: $replay_class/$budget" >&2
    exit 64
    ;;
esac
if [[ -e "$BACE_GCF_REPLAY_OUTPUT" ]]; then
  echo "fresh output root already exists: $BACE_GCF_REPLAY_OUTPUT" >&2
  exit 2
fi
mkdir -p "$BACE_GCF_REPLAY_OUTPUT"

workers="${BACE_GCF_CPU_NEIGHBOR_WORKERS:-4}"
batch_size="${BACE_GCF_GINE_BATCH_SIZE:-256}"
cache_capacity="${BACE_GCF_GRAPH_CACHE_CAPACITY:-100000}"
profile="smoke"
progress_every=100
if [[ "$replay_class" == "quick" ]]; then
  profile="equivalence_quick"
  progress_every=25
fi

run_one() {
  local mode="$1"
  local root="$BACE_GCF_REPLAY_OUTPUT/$mode"
  local cache=0
  local cpu_workers=1
  if [[ "$mode" == "ordered_v2" ]]; then
    cache="$cache_capacity"
    cpu_workers="$workers"
  fi
  "$AUTODL_PYTHON" scripts/baselines/gcfexplainer/run_bace_vrrw.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --dataset-dir "$BACE_GCF_DATASET_DIR" \
    --official-root "$GCF_OFFICIAL_ROOT" \
    --gnn-checkpoint "$BACE_GINE_CHECKPOINT" \
    --neurosed-checkpoint "$BACE_NEUROSED_CHECKPOINT" \
    --neurosed-manifest "$BACE_NEUROSED_MANIFEST" \
    --output-dir "$root" \
    --profile "$profile" \
    --parent-limit 64 \
    --m "$budget" \
    --seed 13 \
    --device1 cuda:0 \
    --device2 cuda:0 \
    --acceleration-mode "$mode" \
    --gine-batch-size "$batch_size" \
    --graph-cache-capacity "$cache" \
    --cpu-neighbor-workers "$cpu_workers" \
    --progress-every "$progress_every"
}

run_one legacy
run_one ordered_v2
equivalence="$BACE_GCF_REPLAY_OUTPUT/equivalence-m${budget}.json"
"$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py equivalence \
  --legacy-root "$BACE_GCF_REPLAY_OUTPUT/legacy" \
  --optimized-root "$BACE_GCF_REPLAY_OUTPUT/ordered_v2" \
  --budget "$budget" \
  --output "$equivalence"

"$AUTODL_PYTHON" - "$BACE_GCF_REPLAY_OUTPUT" "$budget" "$replay_class" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
budget = int(sys.argv[2])
replay_class = sys.argv[3]
gate = root / f"equivalence-m{budget}.json"
payload = json.loads(gate.read_text(encoding="utf-8"))
if payload.get("status") != "PASS" or int(payload.get("budget", -1)) != budget:
    raise SystemExit(f"GCF replay gate is not PASS: {gate}")
manifest = {
    "schema_version": "bace_gcf_single_replay_v1",
    "status": "PASS",
    "budget": budget,
    "replay_class": replay_class,
    "diagnostic_only": True,
    "paper_eligible": False,
    "full_acceleration_authorized": False,
    "equivalence_gate": str(gate),
    "equivalence_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
    "legacy_root": str(root / "legacy"),
    "ordered_v2_root": str(root / "ordered_v2"),
}
(root / "replay_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "PASS").write_text(
    f"BACE GCF {replay_class} M={budget} equivalence passed.\n",
    encoding="utf-8",
)
PY

if [[ "$replay_class" == "quick" ]]; then
  echo "[BACE_GCF_QUICK_${budget}_PASS]"
else
  echo "[BACE_GCF_EQUIVALENCE_M${budget}_PASS]"
fi
