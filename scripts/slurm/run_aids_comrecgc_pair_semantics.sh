#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=192G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=aids-pair-semantics

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${COMRECGC_UPSTREAM_ROOT:?set COMRECGC_UPSTREAM_ROOT}"
: "${COMRECGC_DATASET_DIR:?set COMRECGC_DATASET_DIR}"
: "${COMRECGC_SOURCE_CSV:?set COMRECGC_SOURCE_CSV}"
: "${COMRECGC_GENERATION_DIR:?set COMRECGC_GENERATION_DIR}"
: "${COMRECGC_DISTANCE_CHECKPOINT:?set COMRECGC_DISTANCE_CHECKPOINT}"
: "${COMRECGC_PAIR_STORE_MANIFEST:?set COMRECGC_PAIR_STORE_MANIFEST}"
: "${COMRECGC_PAIR_STORE_MANIFEST_SHA256:?set COMRECGC_PAIR_STORE_MANIFEST_SHA256}"
: "${COMRECGC_PAIR_SEMANTICS_OUTPUT:?set COMRECGC_PAIR_SEMANTICS_OUTPUT}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "scientific_device=cpu"

extra_args=()
if [[ "${COMRECGC_RESUME:-0}" == "1" ]]; then
  extra_args+=(--resume)
fi
if [[ -n "${COMRECGC_MAX_CHUNKS:-}" ]]; then
  extra_args+=(--max-chunks "${COMRECGC_MAX_CHUNKS}")
fi
if [[ "${COMRECGC_SKIP_SOURCE_ARRAY_HASHES:-0}" == "1" ]]; then
  extra_args+=(--skip-source-array-hash-verification)
fi

python scripts/autodl/run_aids_comrecgc_pair_semantics.py \
  --config configs/hpc.yaml \
  --project-root "$PWD" \
  --upstream-root "$COMRECGC_UPSTREAM_ROOT" \
  --dataset-dir "$COMRECGC_DATASET_DIR" \
  --source-csv "$COMRECGC_SOURCE_CSV" \
  --generation-dir "$COMRECGC_GENERATION_DIR" \
  --distance-checkpoint "$COMRECGC_DISTANCE_CHECKPOINT" \
  --pair-store-manifest "$COMRECGC_PAIR_STORE_MANIFEST" \
  --expected-pair-store-manifest-sha256 \
    "$COMRECGC_PAIR_STORE_MANIFEST_SHA256" \
  --output-dir "$COMRECGC_PAIR_SEMANTICS_OUTPUT" \
  --parent-limit 1283 \
  --theta 0.1 \
  --device cpu \
  --distance-batch-size "${COMRECGC_DISTANCE_BATCH_SIZE:-128}" \
  "${extra_args[@]}"
