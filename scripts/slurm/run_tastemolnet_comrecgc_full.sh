#!/usr/bin/env bash
#SBATCH --job-name=taste-t14-comrecgc
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
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
echo "T14 uses the AutoDL retained-input launcher; direct Slurm execution is disabled" >&2
exit 64

# Required CLI parity (documentation only):
# python scripts/run_tastemolnet_comrecgc_full.py --config configs/hpc.yaml \
#   --output-dir "$TASTEMOLNET_T14_OUTPUT" --run-id "$TASTEMOLNET_T14_RUN_ID" \
#   --gpu-uuid "$TASTEMOLNET_GPU_UUID" \
#   --physical-gpu-index "$TASTEMOLNET_T14_GPU_INDEX" \
#   --route-c-spec "$TASTEMOLNET_T14_ROUTE_C_SPEC" \
#   --route-c-storage "$TASTEMOLNET_T14_ROUTE_C_STORAGE" \
#   --checkpoint-only-step "$TASTEMOLNET_T14_CHECKPOINT_ONLY_STEP" \
#   --set inference.fallback_to_heuristic=false
# Add --resume only after the fresh route has atomically published
# "$TASTEMOLNET_T14_OUTPUT/checkpoints/LATEST"; the AutoDL wrapper exposes this
# as TASTEMOLNET_T14_RESUME=1.
# For convergence finalization, omit --checkpoint-only-step and additionally pass
# --resume --convergence-receipt "$TASTEMOLNET_T14_CONVERGENCE_RECEIPT".
