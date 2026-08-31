#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# One allocation, three distinct science Python processes, then one independent
# gate process.  Keeping all roles in this job guarantees the physical GPU UUID
# cannot change between the uninterrupted and restart routes.
set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7

: "${T12_CANARY_OUTPUT_BASE:?set one fresh physical absolute output base}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the adopted managed NeuroSED root}"
: "${TASTE_T3_ROOT:?set the managed calibrated T3 root}"
: "${TASTE_OFFICIAL_GCF_ROOT:?set the integrated official GCF root}"
: "${TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY:?set selector t7 authority JSON}"

if [[ "$T12_CANARY_OUTPUT_BASE" != /* || -e "$T12_CANARY_OUTPUT_BASE" ]]; then
  echo "T12_CANARY_OUTPUT_BASE must be fresh and absolute" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "the canary requires exactly one CUDA_VISIBLE_DEVICES selector" >&2
  exit 2
fi
allocated_gpu_uuid="$(
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=uuid --format=csv,noheader,nounits \
    | tr -d '[:space:]'
)"
if [[ "$allocated_gpu_uuid" != GPU-* ]]; then
  echo "cannot resolve the allocated physical GPU UUID" >&2
  exit 2
fi
if [[ -n "${T12_GPU_UUID:-}" && "$T12_GPU_UUID" != "$allocated_gpu_uuid" ]]; then
  echo "allocated GPU UUID differs from the optional external pin" >&2
  exit 2
fi
export T12_GPU_UUID="$allocated_gpu_uuid"

echo "python=$(command -v python)"
python --version
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv

mapfile -t t12_identities < <(
  python -c 'import secrets, uuid; print(uuid.uuid4()); print(secrets.token_hex(32)); print(uuid.uuid4()); print(secrets.token_hex(32))'
)
if [[ "${#t12_identities[@]}" -ne 4 ]]; then
  echo "failed to create fresh T12 canary identities" >&2
  exit 2
fi
uninterrupted_attempt="${t12_identities[0]}"
uninterrupted_token="${t12_identities[1]}"
resumable_attempt="${t12_identities[2]}"
resumable_token="${t12_identities[3]}"

common_args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --gpu-uuid "$T12_GPU_UUID"
  --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT"
  --t3-root "$TASTE_T3_ROOT"
  --official-root "$TASTE_OFFICIAL_GCF_ROOT"
  --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY"
)

python scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  "${common_args[@]}" \
  --mode uninterrupted \
  --output-root "$T12_CANARY_OUTPUT_BASE/uninterrupted" \
  --observation "$T12_CANARY_OUTPUT_BASE/uninterrupted.json" \
  --attempt-id "$uninterrupted_attempt" \
  --generation-token "$uninterrupted_token"

python scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  "${common_args[@]}" \
  --mode checkpoint \
  --output-root "$T12_CANARY_OUTPUT_BASE/resumable" \
  --attempt-id "$resumable_attempt" \
  --generation-token "$resumable_token"

checkpoint_manifest="$T12_CANARY_OUTPUT_BASE/resumable/checkpoints/checkpoint-00000008.manifest.json"
prefix_receipt="$T12_CANARY_OUTPUT_BASE/resumable/prefix_receipt.json"

python scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  "${common_args[@]}" \
  --mode resume \
  --output-root "$T12_CANARY_OUTPUT_BASE/resumable" \
  --observation "$T12_CANARY_OUTPUT_BASE/resumed.json" \
  --checkpoint-manifest "$checkpoint_manifest" \
  --attempt-id "$resumable_attempt" \
  --generation-token "$resumable_token"

python scripts/run_tastemolnet_gcf_replay_canary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --uninterrupted "$T12_CANARY_OUTPUT_BASE/uninterrupted.json" \
  --cross-process-resumed "$T12_CANARY_OUTPUT_BASE/resumed.json" \
  --checkpoint-prefix-receipt "$prefix_receipt" \
  --output "$T12_CANARY_OUTPUT_BASE/replay_gate.json"
