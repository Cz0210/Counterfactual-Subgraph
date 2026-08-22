#!/usr/bin/env bash
set -euo pipefail

# Fresh, sequential, same-GPU A/B validation. Launch this script through
# exp_run/controller with an exclusive GPU; it never touches an existing VRRW root.
: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${GCF_OFFICIAL_ROOT:?set GCF_OFFICIAL_ROOT}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_NEUROSED_CHECKPOINT:?set BACE_NEUROSED_CHECKPOINT}"
: "${BACE_NEUROSED_MANIFEST:?set BACE_NEUROSED_MANIFEST}"
: "${BACE_GCF_ACCELERATION_OUTPUT:?set a fresh persistent output root}"
: "${AUTODL_PHYSICAL_GPU_UUID:?launch through the AutoDL GPU runner}"

if [[ -e "$BACE_GCF_ACCELERATION_OUTPUT" ]]; then
  echo "fresh output root already exists: $BACE_GCF_ACCELERATION_OUTPUT" >&2
  exit 2
fi
mkdir -p "$BACE_GCF_ACCELERATION_OUTPUT"

workers="${BACE_GCF_CPU_NEIGHBOR_WORKERS:-4}"
batch_size="${BACE_GCF_GINE_BATCH_SIZE:-256}"
cache_capacity="${BACE_GCF_GRAPH_CACHE_CAPACITY:-100000}"

run_one() {
  local budget="$1"
  local mode="$2"
  local root="$BACE_GCF_ACCELERATION_OUTPUT/m${budget}-${mode}"
  local cache=0
  local cpu_workers=1
  if [[ "$mode" == "ordered_v2" ]]; then
    cache="$cache_capacity"
    cpu_workers="$workers"
  fi
  "$AUTODL_PYTHON" scripts/baselines/gcfexplainer/run_bace_vrrw.py \
    --config configs/hpc.yaml \
    --dataset-dir "$BACE_GCF_DATASET_DIR" \
    --official-root "$GCF_OFFICIAL_ROOT" \
    --gnn-checkpoint "$BACE_GINE_CHECKPOINT" \
    --neurosed-checkpoint "$BACE_NEUROSED_CHECKPOINT" \
    --neurosed-manifest "$BACE_NEUROSED_MANIFEST" \
    --output-dir "$root" \
    --profile smoke \
    --parent-limit 64 \
    --m "$budget" \
    --seed 13 \
    --device1 cuda:0 \
    --device2 cuda:0 \
    --acceleration-mode "$mode" \
    --gine-batch-size "$batch_size" \
    --graph-cache-capacity "$cache" \
    --cpu-neighbor-workers "$cpu_workers" \
    --progress-every 1000
}

for budget in 500 1000; do
  run_one "$budget" legacy
  run_one "$budget" ordered_v2
  "$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py equivalence \
    --legacy-root "$BACE_GCF_ACCELERATION_OUTPUT/m${budget}-legacy" \
    --optimized-root "$BACE_GCF_ACCELERATION_OUTPUT/m${budget}-ordered_v2" \
    --budget "$budget" \
    --output "$BACE_GCF_ACCELERATION_OUTPUT/equivalence-m${budget}.json"
done

"$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py benchmark \
  --legacy-root "$BACE_GCF_ACCELERATION_OUTPUT/m1000-legacy" \
  --optimized-root "$BACE_GCF_ACCELERATION_OUTPUT/m1000-ordered_v2" \
  --equivalence-marker "$BACE_GCF_ACCELERATION_OUTPUT/equivalence-m1000.json" \
  --output "$BACE_GCF_ACCELERATION_OUTPUT/same-gpu-ab-benchmark.json"

"$AUTODL_PYTHON" scripts/autodl/gate_bace_gcf_acceleration.py aggregate \
  --equivalence-marker "$BACE_GCF_ACCELERATION_OUTPUT/equivalence-m500.json" \
  --equivalence-marker "$BACE_GCF_ACCELERATION_OUTPUT/equivalence-m1000.json" \
  --benchmark-marker "$BACE_GCF_ACCELERATION_OUTPUT/same-gpu-ab-benchmark.json" \
  --output "$BACE_GCF_ACCELERATION_OUTPUT/GCF_ACCELERATION_GATE.json"

echo "[BACE_GCF_ACCELERATION_SMOKES_PASS]"
