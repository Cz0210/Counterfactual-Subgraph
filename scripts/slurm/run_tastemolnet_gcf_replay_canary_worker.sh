#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7

: "${T12_CANARY_MODE:?set uninterrupted, checkpoint, or resume}"
: "${T12_CANARY_OUTPUT_ROOT:?set one physical absolute output root}"
: "${T12_CANARY_ATTEMPT_ID:?set a UUIDv4}"
: "${T12_CANARY_GENERATION_TOKEN:?set a 64-character lowercase SHA-256 token}"
: "${T12_GPU_UUID:?set the exact physical A800 GPU UUID}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the adopted managed NeuroSED root}"
: "${TASTE_T3_ROOT:?set the managed calibrated T3 root}"
: "${TASTE_OFFICIAL_GCF_ROOT:?set the integrated official GCF root}"
: "${TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY:?set selector t7 authority JSON}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv

phase_args=()
case "$T12_CANARY_MODE" in
  uninterrupted)
    : "${T12_CANARY_OBSERVATION:?set the uninterrupted observation path}"
    phase_args+=(--observation "$T12_CANARY_OBSERVATION")
    ;;
  checkpoint)
    if [[ -n "${T12_CANARY_OBSERVATION:-}" || -n "${T12_CANARY_CHECKPOINT_MANIFEST:-}" ]]; then
      echo "checkpoint mode forbids observation/checkpoint-manifest inputs" >&2
      exit 2
    fi
    ;;
  resume)
    : "${T12_CANARY_OBSERVATION:?set the resumed observation path}"
    : "${T12_CANARY_CHECKPOINT_MANIFEST:?set the committed prefix manifest}"
    phase_args+=(
      --observation "$T12_CANARY_OBSERVATION"
      --checkpoint-manifest "$T12_CANARY_CHECKPOINT_MANIFEST"
    )
    ;;
  *)
    echo "invalid T12_CANARY_MODE=$T12_CANARY_MODE" >&2
    exit 2
    ;;
esac

python scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --mode "$T12_CANARY_MODE" \
  --output-root "$T12_CANARY_OUTPUT_ROOT" \
  --attempt-id "$T12_CANARY_ATTEMPT_ID" \
  --generation-token "$T12_CANARY_GENERATION_TOKEN" \
  --gpu-uuid "$T12_GPU_UUID" \
  --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" \
  --t3-root "$TASTE_T3_ROOT" \
  --official-root "$TASTE_OFFICIAL_GCF_ROOT" \
  --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY" \
  "${phase_args[@]}"
