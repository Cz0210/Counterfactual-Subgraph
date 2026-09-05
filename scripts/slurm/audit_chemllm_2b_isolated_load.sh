#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=chemllm-2b-isolated-load

# The project Slurm baseline requires an A800 allocation.  The audit itself
# deliberately hides CUDA and never acquires a project GPU lock; submit it only
# when the project scheduler permits this non-science reservation.
set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118
set -u

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "python=$(command -v python)"
python --version
python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES=", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

: "${CHEMLLM_2B_SNAPSHOT_ROOT:?set exact physical 2B snapshot root}"
: "${CHEMLLM_2B_SNAPSHOT_MANIFEST:?set exact snapshot_manifest.json path}"
: "${CHEMLLM_2B_SNAPSHOT_MANIFEST_SHA256:?set snapshot manifest byte SHA256}"
: "${CHEMLLM_2B_ISOLATED_OUTPUT_ROOT:?set one fresh non-main output root}"

MODE=${CHEMLLM_2B_ISOLATED_MODE:-cpu-load}
TINY_FORWARD_ARGS=()
# Includes native model.build_inputs and one at-most-four-token greedy probe.
if [[ "${CHEMLLM_2B_TINY_FORWARD:-0}" == "1" ]]; then
  TINY_FORWARD_ARGS+=(--tiny-forward)
fi

exec nice -n 10 ionice -c2 -n7 \
  python scripts/ablations/llm/audit_chemllm_2b_isolated_load.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --snapshot-root "$CHEMLLM_2B_SNAPSHOT_ROOT" \
    --snapshot-manifest "$CHEMLLM_2B_SNAPSHOT_MANIFEST" \
    --snapshot-manifest-sha256 "$CHEMLLM_2B_SNAPSHOT_MANIFEST_SHA256" \
    --output-root "$CHEMLLM_2B_ISOLATED_OUTPUT_ROOT" \
    --mode "$MODE" \
    "${TINY_FORWARD_ARGS[@]}"
