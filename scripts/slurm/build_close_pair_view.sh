#!/bin/bash
#SBATCH --job-name=comrecgc_close_view
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'

: "${PAIR_SEMANTICS_CONTRACT:?PAIR_SEMANTICS_CONTRACT is required}"
: "${PHYSICAL_VECTORS:?PHYSICAL_VECTORS is required}"
: "${NORMALIZED_DISTANCES:?NORMALIZED_DISTANCES is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
ALL_PAIRS_CLOSE_CERTIFICATE="${ALL_PAIRS_CLOSE_CERTIFICATE:-}"
MAX_COMPACT_GB="${MAX_COMPACT_GB:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-1000000}"
RESUME="${RESUME:-false}"

ARGS=(
  --config configs/hpc.yaml
  --pair-semantics-contract "$PAIR_SEMANTICS_CONTRACT"
  --physical-vectors "$PHYSICAL_VECTORS"
  --normalized-distances "$NORMALIZED_DISTANCES"
  --output-dir "$OUTPUT_DIR"
  --max-compact-gb "$MAX_COMPACT_GB"
  --block-size "$BLOCK_SIZE"
)
[[ -z "$ALL_PAIRS_CLOSE_CERTIFICATE" ]] || ARGS+=(--all-pairs-close-certificate "$ALL_PAIRS_CLOSE_CERTIFICATE")
[[ "$RESUME" != "true" ]] || ARGS+=(--resume)
python scripts/baselines/comrecgc/build_close_pair_view.py "${ARGS[@]}"
