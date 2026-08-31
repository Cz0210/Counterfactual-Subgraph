#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7

: "${T12_MODE:?set fresh or resume}"
: "${T12_OUTPUT_ROOT:?set the fresh absolute production root}"
: "${T12_ATTEMPT_ID:?set one UUIDv4 reused across both segments}"
: "${T12_GENERATION_TOKEN:?set one 64-character lowercase token reused across both segments}"
: "${T12_GPU_UUID:?set the exact physical A800 UUID}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the adopted managed NeuroSED root}"
: "${TASTE_T3_ROOT:?set the managed calibrated T3 root}"
: "${TASTE_OFFICIAL_GCF_ROOT:?set the integrated official GCF root}"
: "${TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY:?set the typed threshold authority}"
: "${T12_EXACT_REPLAY_GATE:?set the exact gate-v2 JSON}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv

phase_args=()
case "$T12_MODE" in
  fresh)
    if [[ -n "${T12_CHECKPOINT_MANIFEST:-}" ]]; then
      echo "fresh T12 forbids T12_CHECKPOINT_MANIFEST" >&2
      exit 2
    fi
    ;;
  resume)
    : "${T12_CHECKPOINT_MANIFEST:?set the committed 10k manifest}"
    phase_args+=(--checkpoint-manifest "$T12_CHECKPOINT_MANIFEST")
    ;;
  *)
    echo "invalid T12_MODE=$T12_MODE" >&2
    exit 2
    ;;
esac

python scripts/run_tastemolnet_gcf_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --mode "$T12_MODE" \
  --output-root "$T12_OUTPUT_ROOT" \
  --attempt-id "$T12_ATTEMPT_ID" \
  --generation-token "$T12_GENERATION_TOKEN" \
  --gpu-uuid "$T12_GPU_UUID" \
  --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" \
  --t3-root "$TASTE_T3_ROOT" \
  --official-root "$TASTE_OFFICIAL_GCF_ROOT" \
  --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY" \
  --exact-replay-gate "$T12_EXACT_REPLAY_GATE" \
  "${phase_args[@]}"
