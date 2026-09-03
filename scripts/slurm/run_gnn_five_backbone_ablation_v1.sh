#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=gnn-five-backbone-v1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available(), "gpu_count=", torch.cuda.device_count())'

: "${GNN_FIVE_BACKBONE_STATUS:?set byte-pinned allowed status JSON}"
: "${GNN_FIVE_BACKBONE_STATUS_SHA256:?set status byte SHA256}"
: "${GNN_FIVE_BACKBONE_RUN_SPEC:?set complete self-hashed science run spec}"
: "${GNN_FIVE_BACKBONE_RUN_SPEC_SHA256:?set run-spec byte SHA256}"
: "${GNN_FIVE_BACKBONE_OUTPUT_ROOT:?set fresh UUID-bearing output root}"
: "${MAIN_READY_GPU_TASKS_RECEIPT:?set live main READY_GPU queue JSON}"

args=(
  python scripts/autodl/run_gnn_five_backbone_ablation_v1.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --status "$GNN_FIVE_BACKBONE_STATUS"
  --status-sha256 "$GNN_FIVE_BACKBONE_STATUS_SHA256"
  --run-spec "$GNN_FIVE_BACKBONE_RUN_SPEC"
  --run-spec-sha256 "$GNN_FIVE_BACKBONE_RUN_SPEC_SHA256"
  --output-root "$GNN_FIVE_BACKBONE_OUTPUT_ROOT"
  --main-ready-gpu-tasks "$MAIN_READY_GPU_TASKS_RECEIPT"
  --poll-seconds "${GNN_FIVE_BACKBONE_POLL_SECONDS:-5}"
)
[[ "${GNN_FIVE_BACKBONE_RESUME:-0}" == "1" ]] && args+=(--resume)

exec nice -n 10 "${args[@]}"
