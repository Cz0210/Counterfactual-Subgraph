#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_PYTHON:=/root/miniconda3/envs/smiles_pip118/bin/python}"
: "${T6_OUTPUT_ROOT:?set the independently verified T6 science root}"
: "${T11_PPO_OUTPUT_ROOT:?set one fresh persistent T11 PPO root}"
: "${T11_SCIENCE_ROOT:?set one fresh/resumable persistent T11 science root}"
: "${T11_FINAL_ROOT:?set one fresh independent-verifier root}"
: "${TASTEMOLNET_BASE_MODEL:?set the exact T6 base model}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set the frozen three-class GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set the frozen train CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set the frozen calibration CSV}"
: "${TASTEMOLNET_TEST_CSV:?set the held-out test CSV}"
: "${MOLCLR_ROOT:?set the pinned MolCLR source root}"
: "${MOLCLR_CHECKPOINT:?set the pinned MolCLR checkpoint}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set the adopted shared WNode threshold contract}"
: "${WNODE_CACHE_DB:?set the persistent T11 WNode SQLite cache}"
: "${NODE_EMBEDDING_CACHE_DIR:?set the persistent MolCLR node cache}"

[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
  echo "T11 requires exactly one physical GPU in CUDA_VISIBLE_DEVICES" >&2
  exit 64
}
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || {
  echo "T11 forbids GNN backbone ablation before 16/16" >&2
  exit 64
}

export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

resume_ppo=()
if [[ -n "${T11_PPO_RESUME_CHECKPOINT:-}" ]]; then
  resume_ppo+=(--resume-from-checkpoint "$T11_PPO_RESUME_CHECKPOINT")
fi

if [[ -f "$T11_PPO_OUTPUT_ROOT/PASS" ]] \
  && [[ "$(<"$T11_PPO_OUTPUT_ROOT/PASS")" == '[TASTE_T11_OURS_PPO_FULL_PASS]' ]]; then
  echo "adopting existing exact T11 PPO PASS; downstream will revalidate all bytes"
else
  "$AUTODL_PYTHON" scripts/train_tastemolnet_ours_full.py \
    --config configs/hpc.yaml \
    --model-path "$TASTEMOLNET_BASE_MODEL" \
    --dataset-path "$TASTEMOLNET_TRAIN_CSV" \
    --output-dir "$T11_PPO_OUTPUT_ROOT" \
    --t6-output "$T6_OUTPUT_ROOT" \
    --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
    --batch-size "${T11_PPO_BATCH_SIZE:-8}" \
    "${resume_ppo[@]}"
fi

resume_science=()
if [[ -f "$T11_SCIENCE_ROOT/checkpoint.json" ]]; then
  resume_science+=(--resume)
fi

"$AUTODL_PYTHON" scripts/run_tastemolnet_ours_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --science-root "$T11_SCIENCE_ROOT" \
  --ppo-root "$T11_PPO_OUTPUT_ROOT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --calibration-csv "$TASTEMOLNET_CALIBRATION_CSV" \
  --test-csv "$TASTEMOLNET_TEST_CSV" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --threshold-contract "$TASTEMOLNET_THRESHOLD_CONTRACT" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR" \
  "${resume_science[@]}"

# A distinct process is the only publisher of terminal PASS.
"$AUTODL_PYTHON" scripts/run_tastemolnet_ours_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --science-root "$T11_SCIENCE_ROOT" \
  --final-root "$T11_FINAL_ROOT" \
  --threshold-contract "$TASTEMOLNET_THRESHOLD_CONTRACT" \
  --verify-only
