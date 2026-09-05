#!/usr/bin/env bash
# Explicit project-owner override: L0 common evaluation is intel CPU-only.
# Submit once with --dependency=afterok:<corrected-package-job>; no GPU request.
#SBATCH --partition=intel
#SBATCH --job-name=bace-l0-corrected-core
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
if [[ -n "${LLM_EXECUTION_WORKTREE:-}" ]]; then cd "$LLM_EXECUTION_WORKTREE"; export PYTHONPATH=$PWD; fi
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
: "${L0_PORTABLE_INPUT_BUNDLE:?portable L0 bytes required}"
: "${GNN_INPUT_BUNDLE:?existing same-SHA GNN input bundle required}"
: "${GNN_CORRECTED_ARCHIVE:?new corrected package required}"
: "${GNN_CORRECTED_PACKAGE_RECEIPT:?new package receipt required}"
: "${LLM_REGISTRY_ROOT:?LLM-only registry required}"
: "${L0_OUTPUT_ROOT:?fresh/resumable CPU output root required}"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("CPU-only L0")'
extra=()
[[ "${L0_RESUME:-0}" == 1 ]] && extra+=(--resume)
exec python -I -B scripts/hpc/llm/run_bace_l0_cpu.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false run \
  --portable-input-bundle "$L0_PORTABLE_INPUT_BUNDLE" --gnn-input-bundle "$GNN_INPUT_BUNDLE" \
  --corrected-gnn-archive "$GNN_CORRECTED_ARCHIVE" \
  --corrected-package-receipt "$GNN_CORRECTED_PACKAGE_RECEIPT" \
  --registry-root "$LLM_REGISTRY_ROOT" --output-root "$L0_OUTPUT_ROOT" \
  --cpu-threads 2 "${extra[@]}"
