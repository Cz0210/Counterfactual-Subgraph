#!/bin/bash
#SBATCH --job-name=bace-llm-common
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
if [[ -n "${LLM_EXECUTION_WORKTREE:-}" ]]; then
  cd "$LLM_EXECUTION_WORKTREE"
  export PYTHONPATH=$PWD
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
: "${LLM_TASK_SPEC:?native task spec required}"
: "${LLM_CANDIDATE_ROOT:?completed candidate root required}"
: "${GNN_INPUT_BUNDLE:?frozen GNN input bundle required}"
: "${GNN_VERIFIED_ARCHIVE:?independently verified archive required}"
: "${GNN_VERIFIED_SHA256:?archive transport SHA required}"
: "${LLM_REGISTRY_ROOT:?independent LLM registry required}"
: "${LLM_DOWNSTREAM_ROOT:?fresh downstream output required}"
echo "Python: $(command -v python)"
python -c 'import sys,torch; print(sys.version); print("CUDA available:",torch.cuda.is_available())'
extra=()
if [[ "${LLM_RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
# Launcher obtains its own authorized lease; this script never acquires a main lease.
exec python scripts/ablations/llm/run_bace_common_downstream.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --task-spec "$LLM_TASK_SPEC" --candidate-root "$LLM_CANDIDATE_ROOT" \
  --gnn-input-bundle "$GNN_INPUT_BUNDLE" \
  --gnn-verified-archive "$GNN_VERIFIED_ARCHIVE" --gnn-verified-sha256 "$GNN_VERIFIED_SHA256" \
  --registry-root "$LLM_REGISTRY_ROOT" --output-root "$LLM_DOWNSTREAM_ROOT" \
  --device "${LLM_DOWNSTREAM_DEVICE:-cuda:0}" --cpu-threads 2 "${extra[@]}"
