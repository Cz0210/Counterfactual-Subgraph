#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export TASTEMOLNET_T8_CONTROLLER_OWNED_GPU_SLOT=1
: "${T8_ATTEMPT_ID:?}" "${T8_T3_OUTPUT:?}" "${T8_T4_OUTPUT:?}" "${T8_GNN_CHECKPOINT:?}"
: "${T8_TRAIN_CSV:?}" "${T8_OFFICIAL_ROOT:?}" "${T8_STATE_DIR:?}" "${T8_OUTPUT_DIR:?}"
: "${T8_GSPAN_SCRATCH_ROOT:?}"
T8_ZERO_CANDIDATE_RECOVERY="${T8_ZERO_CANDIDATE_RECOVERY:-0}"
RECOVERY_ARGS=()
case "$T8_ZERO_CANDIDATE_RECOVERY" in
  0) ;;
  1)
    : "${T8_RECOVERY_SOURCE_ATTEMPT_ID:?}"
    RECOVERY_ARGS+=(
      --zero-candidate-recovery
      --recovery-source-attempt-id "$T8_RECOVERY_SOURCE_ATTEMPT_ID"
    )
    ;;
  *)
    echo "T8_ZERO_CANDIDATE_RECOVERY must be 0 or 1" >&2
    exit 64
    ;;
esac
echo "python=$(command -v python)"
echo "checkpoint_evidence=post_callback_atomic_seal_v1"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python -I -B scripts/autodl/run_tastemolnet_t8_deadline.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --attempt-id "$T8_ATTEMPT_ID" --t3-output "$T8_T3_OUTPUT" --t4-output "$T8_T4_OUTPUT" \
  --gnn-checkpoint "$T8_GNN_CHECKPOINT" --train-csv "$T8_TRAIN_CSV" \
  --official-root "$T8_OFFICIAL_ROOT" --gspan-scratch-root "$T8_GSPAN_SCRATCH_ROOT" \
  --state-dir "$T8_STATE_DIR" --output-dir "$T8_OUTPUT_DIR" \
  "${RECOVERY_ARGS[@]}"
