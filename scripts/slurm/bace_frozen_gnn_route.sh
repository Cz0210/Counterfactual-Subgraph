#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

ACTION="${ACTION:?Set ACTION=scoring-preflight or ACTION=stage-blocker}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the frozen B4 GNN bundle}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh stage directory}"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

case "$ACTION" in
  scoring-preflight)
    ORACLE_SMOKE_DIR="${ORACLE_SMOKE_DIR:?Set ORACLE_SMOKE_DIR to the passing B5 output}"
    python scripts/autodl/bace_frozen_gnn_route.py \
      --config configs/hpc.yaml \
      scoring-preflight \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --oracle-smoke-dir "$ORACLE_SMOKE_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --device cuda:0
    ;;
  stage-blocker)
    BACE_STAGE="${BACE_STAGE:?Set BACE_STAGE to one of B7_PPO_FULL through B14_FROZEN}"
    PREDECESSOR_OUTPUT="${PREDECESSOR_OUTPUT:?Set PREDECESSOR_OUTPUT to the prior PASS output}"
    EXTRA_ARGS=()
    [[ -n "${BASE_POOL_OUTPUT:-}" ]] && EXTRA_ARGS+=(--base-pool-output "$BASE_POOL_OUTPUT")
    [[ -n "${MOLCLR_CHECKPOINT:-}" ]] && EXTRA_ARGS+=(--molclr-checkpoint "$MOLCLR_CHECKPOINT")
    [[ -n "${TEST_CSV:-}" ]] && EXTRA_ARGS+=(--test-csv "$TEST_CSV")
    python scripts/autodl/bace_frozen_gnn_route.py \
      --config configs/hpc.yaml \
      stage-blocker \
      --stage "$BACE_STAGE" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --predecessor-output "$PREDECESSOR_OUTPUT" \
      --output-dir "$OUTPUT_DIR" \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unsupported ACTION: $ACTION" >&2
    exit 2
    ;;
esac
