#!/usr/bin/env bash
set -euo pipefail

# Fresh diagnostic-only replay used after an equivalence implementation change.
# It never authorizes a full run; the formal 500/1000 + A/B gate remains
# mandatory and separate.
: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${GCF_OFFICIAL_ROOT:?set GCF_OFFICIAL_ROOT}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_NEUROSED_CHECKPOINT:?set BACE_NEUROSED_CHECKPOINT}"
: "${BACE_NEUROSED_MANIFEST:?set BACE_NEUROSED_MANIFEST}"
: "${BACE_GCF_QUICK_REPLAY_OUTPUT:?set a fresh persistent output root}"
: "${AUTODL_PHYSICAL_GPU_UUID:?launch through the AutoDL GPU runner}"

if [[ -e "$BACE_GCF_QUICK_REPLAY_OUTPUT" ]]; then
  echo "fresh output root already exists: $BACE_GCF_QUICK_REPLAY_OUTPUT" >&2
  exit 2
fi
mkdir -p "$BACE_GCF_QUICK_REPLAY_OUTPUT"

workers="${BACE_GCF_CPU_NEIGHBOR_WORKERS:-4}"
batch_size="${BACE_GCF_GINE_BATCH_SIZE:-256}"
cache_capacity="${BACE_GCF_GRAPH_CACHE_CAPACITY:-100000}"

run_one() {
  local budget="$1"
  local mode="$2"
  local root="$BACE_GCF_QUICK_REPLAY_OUTPUT/m${budget}-${mode}"
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
    --profile equivalence_quick \
    --parent-limit 64 \
    --m "$budget" \
    --seed 13 \
    --device1 cuda:0 \
    --device2 cuda:0 \
    --acceleration-mode "$mode" \
    --gine-batch-size "$batch_size" \
    --graph-cache-capacity "$cache" \
    --cpu-neighbor-workers "$cpu_workers" \
    --progress-every 50
}

for budget in 50 100; do
  run_one "$budget" legacy
  run_one "$budget" ordered_v2
  "$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py equivalence \
    --legacy-root "$BACE_GCF_QUICK_REPLAY_OUTPUT/m${budget}-legacy" \
    --optimized-root "$BACE_GCF_QUICK_REPLAY_OUTPUT/m${budget}-ordered_v2" \
    --budget "$budget" \
    --output "$BACE_GCF_QUICK_REPLAY_OUTPUT/equivalence-m${budget}.json"
done

"$AUTODL_PYTHON" - "$BACE_GCF_QUICK_REPLAY_OUTPUT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
markers = []
for budget in (50, 100):
    path = root / f"equivalence-m{budget}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("budget") != budget:
        raise SystemExit(f"quick replay gate is not PASS for M={budget}: {path}")
    markers.append(
        {
            "budget": budget,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
summary = {
    "schema_version": 1,
    "status": "PASS",
    "diagnostic_only": True,
    "eligible_for_full_acceleration_gate": False,
    "markers": markers,
}
(root / "QUICK_REPLAY_PASS.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "[BACE_GCF_QUICK_REPLAY_PASS]"
