#!/bin/bash
#SBATCH --job-name=mut-first-divergence
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${MUT_LEGACY_ROOT:?MUT_LEGACY_ROOT is required}"
: "${MUT_INSTRUMENTED_ROOT:?MUT_INSTRUMENTED_ROOT is required}"
: "${MUT_TASK_SPEC:?MUT_TASK_SPEC is required}"
: "${MUT_DATASET_SUMMARY:?MUT_DATASET_SUMMARY is required}"
: "${MUT_DIVERGENCE_OUTPUT:?MUT_DIVERGENCE_OUTPUT is required}"

# The audit itself is JSON-only and never uses the allocated GPU.  This paired
# wrapper exists for repository workflow compatibility; AutoDL production uses
# the one-shot CPU owner instead of submitting this Slurm script.
python scripts/autodl/audit_mut_first_divergence_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --legacy-root "$MUT_LEGACY_ROOT" \
  --instrumented-root "$MUT_INSTRUMENTED_ROOT" \
  --task-spec "$MUT_TASK_SPEC" \
  --dataset-summary "$MUT_DATASET_SUMMARY" \
  --output-dir "$MUT_DIVERGENCE_OUTPUT" \
  --timebox-seconds 3600
