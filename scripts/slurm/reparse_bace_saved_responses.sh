#!/bin/bash
# Parser-only audit of saved strings: explicit CPU-only exception; no model/OT.
#SBATCH --job-name=bace-raw-reparse
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
if [[ -n "${LLM_EXECUTION_WORKTREE:-}" ]]; then
  cd "$LLM_EXECUTION_WORKTREE"
  export PYTHONPATH=$PWD
fi
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${LLM_SCORED_JSONL:?saved completed train response file required}"
: "${LLM_PARSER_AUDIT_ROOT:?fresh output root required}"
echo "Python: $(command -v python)"
python -c 'import sys; print(sys.version); print("CPU-only parser audit; CUDA disabled")'
exec python -I -B scripts/ablations/llm/reparse_bace_saved_responses.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --scored-jsonl "$LLM_SCORED_JSONL" --output-root "$LLM_PARSER_AUDIT_ROOT"
