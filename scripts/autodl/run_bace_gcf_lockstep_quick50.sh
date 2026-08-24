#!/usr/bin/env bash
set -euo pipefail

# One controller-owned, fresh-root diagnostic sequence.  It stops at the first
# failed gate and never signals or writes into the protected legacy 50k run.
: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${GCF_OFFICIAL_ROOT:?set GCF_OFFICIAL_ROOT}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_NEUROSED_CHECKPOINT:?set BACE_NEUROSED_CHECKPOINT}"
: "${BACE_NEUROSED_MANIFEST:?set BACE_NEUROSED_MANIFEST}"
: "${BACE_GCF_LOCKSTEP_OUTPUT:?set a fresh persistent output root}"
: "${AUTODL_PHYSICAL_GPU_UUID:?launch through the UUID-lock AutoDL runner}"

if [[ -e "$BACE_GCF_LOCKSTEP_OUTPUT" ]]; then
  echo "fresh output root already exists: $BACE_GCF_LOCKSTEP_OUTPUT" >&2
  exit 2
fi
mkdir -p "$BACE_GCF_LOCKSTEP_OUTPUT"

"$AUTODL_PYTHON" scripts/autodl/benchmark_bace_frozen_gine_batch.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "$BACE_GCF_DATASET_DIR" \
  --checkpoint-dir "$BACE_GINE_CHECKPOINT" \
  --output-dir "$BACE_GCF_LOCKSTEP_OUTPUT/frozen_gine_cpu_gpu_benchmark" \
  --rows "${BACE_GINE_BENCHMARK_ROWS:-64}" \
  --repeats "${BACE_GINE_BENCHMARK_REPEATS:-5}"

run_one() {
  local mode="$1"
  local name="$2"
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
    --output-dir "$BACE_GCF_LOCKSTEP_OUTPUT/$name" \
    --profile equivalence_quick \
    --parent-limit 64 \
    --m 50 \
    --seed 13 \
    --device1 cuda:0 \
    --device2 cuda:0 \
    --acceleration-mode "$mode" \
    --gine-batch-size "${BACE_GCF_GINE_BATCH_SIZE:-256}" \
    --graph-cache-capacity "$cache" \
    --cpu-neighbor-workers "$workers" \
    --progress-every 25
}

run_one legacy legacy_a
run_one legacy legacy_b

"$AUTODL_PYTHON" - \
  "$BACE_GCF_LOCKSTEP_OUTPUT/legacy_a/lockstep_trace.json" \
  "$BACE_GCF_LOCKSTEP_OUTPUT/legacy_b/lockstep_trace.json" \
  "$BACE_GCF_LOCKSTEP_OUTPUT/legacy_a_vs_legacy_b.json" <<'PY'
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

run_one ordered_v2 ordered_v2
"$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py equivalence \
  --legacy-root "$BACE_GCF_LOCKSTEP_OUTPUT/legacy_a" \
  --optimized-root "$BACE_GCF_LOCKSTEP_OUTPUT/ordered_v2" \
  --budget 50 \
  --output "$BACE_GCF_LOCKSTEP_OUTPUT/legacy_a_vs_ordered_v2.json"

"$AUTODL_PYTHON" - "$BACE_GCF_LOCKSTEP_OUTPUT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
evidence = [
    root / "frozen_gine_cpu_gpu_benchmark/benchmark.json",
    root / "legacy_a_vs_legacy_b.json",
    root / "legacy_a_vs_ordered_v2.json",
]
for path in evidence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise SystemExit(f"diagnostic evidence failed: {path}")
manifest = {
    "schema_version": "bace_gcf_lockstep_quick50_v1",
    "status": "PASS",
    "diagnostic_only": True,
    "paper_eligible": False,
    "full_acceleration_authorized": False,
    "evidence": [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in evidence
    ],
}
(root / "diagnostic_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "PASS").write_text("BACE GCF Quick-50 lockstep passed.\n", encoding="utf-8")
PY

echo "[BACE_GCF_LOCKSTEP_QUICK50_PASS]"
