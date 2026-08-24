#!/usr/bin/env bash
set -euo pipefail

# CPU-only deterministic diagnosis.  CUDA is hidden before importing torch;
# this task owns no GPU lock and cannot interfere with the protected 50k run.
: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${GCF_OFFICIAL_ROOT:?set GCF_OFFICIAL_ROOT}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_NEUROSED_CHECKPOINT:?set BACE_NEUROSED_CHECKPOINT}"
: "${BACE_NEUROSED_MANIFEST:?set BACE_NEUROSED_MANIFEST}"
: "${BACE_GCF_CPU_LOCKSTEP_OUTPUT:?set a fresh persistent output root}"

if [[ -e "$BACE_GCF_CPU_LOCKSTEP_OUTPUT" ]]; then
  echo "fresh output root already exists: $BACE_GCF_CPU_LOCKSTEP_OUTPUT" >&2
  exit 2
fi
mkdir -p "$BACE_GCF_CPU_LOCKSTEP_OUTPUT"
export CUDA_VISIBLE_DEVICES=""

run_one() {
  local budget="$1"
  local mode="$2"
  local name="$3"
  local root="$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/$name"
  local cache=0
  local workers=1
  if [[ "$mode" == "ordered_v2" ]]; then
    cache="${BACE_GCF_GRAPH_CACHE_CAPACITY:-100000}"
    workers="${BACE_GCF_CPU_NEIGHBOR_WORKERS:-4}"
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
    --device1 cpu \
    --device2 cpu \
    --acceleration-mode "$mode" \
    --gine-batch-size "${BACE_GCF_GINE_BATCH_SIZE:-256}" \
    --graph-cache-capacity "$cache" \
    --cpu-neighbor-workers "$workers" \
    --progress-every 25
}

compare_legacy_replays() {
  local budget="$1"
  "$AUTODL_PYTHON" - \
    "$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/legacy_a/lockstep_trace.json" \
    "$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/legacy_b/lockstep_trace.json" \
    "$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/legacy_a_vs_legacy_b.json" <<'PY'
from pathlib import Path
import sys
from src.baselines.gcfexplainer_acceleration import (
    compare_lockstep_traces,
    write_fresh_json,
)

payload = compare_lockstep_traces(Path(sys.argv[1]), Path(sys.argv[2]))
write_fresh_json(Path(sys.argv[3]), payload)
if payload.get("status") != "PASS":
    raise SystemExit(3)
PY
}

run_budget() {
  local budget="$1"
  run_one "$budget" legacy legacy_a
  run_one "$budget" legacy legacy_b
  compare_legacy_replays "$budget"
  run_one "$budget" ordered_v2 ordered_v2
  "$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py equivalence \
    --legacy-root "$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/legacy_a" \
    --optimized-root "$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/ordered_v2" \
    --budget "$budget" \
    --output "$BACE_GCF_CPU_LOCKSTEP_OUTPUT/m${budget}/legacy_a_vs_ordered_v2.json"
}

# Quick-100 is unreachable unless all Quick-50 commands and exact gates pass.
run_budget 50
run_budget 100

"$AUTODL_PYTHON" - "$BACE_GCF_CPU_LOCKSTEP_OUTPUT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
evidence = []
for budget in (50, 100):
    for name in ("legacy_a_vs_legacy_b.json", "legacy_a_vs_ordered_v2.json"):
        path = root / f"m{budget}/{name}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise SystemExit(f"CPU lockstep evidence failed: {path}")
        evidence.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
manifest = {
    "schema_version": "bace_gcf_cpu_lockstep_quick50_100_v1",
    "status": "PASS",
    "diagnostic_only": True,
    "device1": "cpu",
    "device2": "cpu",
    "cuda_visible_devices": "",
    "paper_eligible": False,
    "full_acceleration_authorized": False,
    "evidence": evidence,
}
(root / "diagnostic_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "PASS").write_text("BACE GCF CPU Quick-50/100 lockstep passed.\n", encoding="utf-8")
PY

echo "[BACE_GCF_CPU_LOCKSTEP_QUICK50_100_PASS]"
