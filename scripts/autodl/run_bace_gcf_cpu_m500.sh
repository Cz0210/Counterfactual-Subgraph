#!/usr/bin/env bash
set -euo pipefail

# CPU-only formal M=500 equivalence follows, but never mutates, the completed
# Quick-50/100 diagnostic.  CUDA remains hidden and this route owns no GPU lock.
: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${GCF_OFFICIAL_ROOT:?set GCF_OFFICIAL_ROOT}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_NEUROSED_CHECKPOINT:?set BACE_NEUROSED_CHECKPOINT}"
: "${BACE_NEUROSED_MANIFEST:?set BACE_NEUROSED_MANIFEST}"
: "${BACE_GCF_CPU_QUICK_ROOT:?set completed Quick-50/100 root}"
: "${BACE_GCF_CPU_QUICK_MANIFEST_SHA256:?bind diagnostic_manifest.json}"
: "${BACE_GCF_CPU_M500_OUTPUT:?set a fresh persistent output root}"

if [[ -e "$BACE_GCF_CPU_M500_OUTPUT" || -L "$BACE_GCF_CPU_M500_OUTPUT" ]]; then
  echo "fresh output root already exists: $BACE_GCF_CPU_M500_OUTPUT" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Validate the upstream terminal proof before creating this route's root.
"$AUTODL_PYTHON" - \
  "$BACE_GCF_CPU_QUICK_ROOT" \
  "$BACE_GCF_CPU_QUICK_MANIFEST_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve(strict=True)
expected_sha256 = sys.argv[2]
manifest_path = root / "diagnostic_manifest.json"
pass_path = root / "PASS"
if (
    not manifest_path.is_file()
    or manifest_path.is_symlink()
    or not pass_path.is_file()
    or pass_path.is_symlink()
):
    raise SystemExit("Quick-50/100 terminal proof is missing or a symlink")
actual_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit("Quick-50/100 diagnostic manifest SHA-256 mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "schema_version": "bace_gcf_cpu_lockstep_quick50_100_v1",
    "status": "PASS",
    "diagnostic_only": True,
    "device1": "cpu",
    "device2": "cpu",
    "cuda_visible_devices": "",
    "paper_eligible": False,
    "full_acceleration_authorized": False,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"Quick-50/100 terminal binding mismatch: {key}")
evidence = manifest.get("evidence")
if not isinstance(evidence, list) or len(evidence) != 4:
    raise SystemExit("Quick-50/100 must bind exactly four gate artifacts")
expected_names = {
    "m50/legacy_a_vs_legacy_b.json",
    "m50/legacy_a_vs_ordered_v2.json",
    "m100/legacy_a_vs_legacy_b.json",
    "m100/legacy_a_vs_ordered_v2.json",
}
observed_names = set()
for row in evidence:
    source_path = Path(str(row.get("path", ""))).expanduser()
    if source_path.is_symlink():
        raise SystemExit("Quick evidence path must not be a symlink")
    path = source_path.resolve(strict=True)
    if source_path.absolute() != path:
        raise SystemExit("Quick evidence path contains a symlink component")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SystemExit("Quick evidence escapes its authority root") from exc
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row.get(
        "sha256"
    ):
        raise SystemExit(f"Quick evidence hash mismatch: {relative}")
    if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit(f"Quick evidence is not PASS: {relative}")
    observed_names.add(relative)
if observed_names != expected_names:
    raise SystemExit("Quick-50/100 evidence inventory mismatch")
PY

mkdir -p "$BACE_GCF_CPU_M500_OUTPUT"

run_one() {
  local mode="$1"
  local name="$2"
  local root="$BACE_GCF_CPU_M500_OUTPUT/$name"
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
    --profile smoke \
    --parent-limit 64 \
    --m 500 \
    --seed 13 \
    --device1 cpu \
    --device2 cpu \
    --acceleration-mode "$mode" \
    --gine-batch-size "${BACE_GCF_GINE_BATCH_SIZE:-256}" \
    --graph-cache-capacity "$cache" \
    --cpu-neighbor-workers "$workers" \
    --progress-every 25
}

run_one legacy legacy_m500
run_one ordered_v2 patched_m500

"$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py equivalence \
  --legacy-root "$BACE_GCF_CPU_M500_OUTPUT/legacy_m500" \
  --optimized-root "$BACE_GCF_CPU_M500_OUTPUT/patched_m500" \
  --budget 500 \
  --output "$BACE_GCF_CPU_M500_OUTPUT/legacy_vs_patched_m500.json"

# PASS is deliberately the final write.
"$AUTODL_PYTHON" - \
  "$BACE_GCF_CPU_M500_OUTPUT" \
  "$BACE_GCF_CPU_QUICK_ROOT/diagnostic_manifest.json" \
  "$BACE_GCF_CPU_QUICK_MANIFEST_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from src.baselines.gcfexplainer_acceleration import write_fresh_json

root = Path(sys.argv[1]).resolve(strict=True)
quick_manifest = Path(sys.argv[2]).resolve(strict=True)
quick_sha256 = sys.argv[3]
gate_path = root / "legacy_vs_patched_m500.json"
gate = json.loads(gate_path.read_text(encoding="utf-8"))
if gate.get("status") != "PASS" or gate.get("equivalence") != "CANONICAL_EXACT":
    raise SystemExit("CPU M500 equivalence did not pass exactly")
for name in ("legacy_m500", "patched_m500"):
    run_root = root / name
    if not (run_root / "_RUN_COMPLETE.json").is_file():
        raise SystemExit(f"M500 run incomplete: {name}")
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("M") != 500
        or manifest.get("profile") != "smoke"
        or manifest.get("oracle_backend") != "gnn"
        or manifest.get("classifier_family") != "gine"
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("gpu_uuid") is not None
    ):
        raise SystemExit(f"M500 scientific/device binding mismatch: {name}")
payload = {
    "schema_version": "bace_gcf_cpu_m500_equivalence_v1",
    "status": "PASS",
    "budget": 500,
    "device1": "cpu",
    "device2": "cpu",
    "cuda_visible_devices": "",
    "diagnostic_only": True,
    "paper_eligible": False,
    "full_acceleration_authorized": False,
    "quick_manifest": str(quick_manifest),
    "quick_manifest_sha256": quick_sha256,
    "equivalence_gate": str(gate_path),
    "equivalence_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
}
write_fresh_json(root / "diagnostic_manifest.json", payload)
write_fresh_json(
    root / "PASS",
    {
        "schema_version": "bace_gcf_cpu_m500_pass_v1",
        "status": "PASS",
        "diagnostic_manifest_sha256": hashlib.sha256(
            (root / "diagnostic_manifest.json").read_bytes()
        ).hexdigest(),
    },
)
PY

echo "[GCF_M500_EQUIVALENCE_PASS]"
