#!/usr/bin/env bash
# CPU-only exception explicitly authorized for BACE proposal-fixed ablation.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
# Site bashrc/Conda scripts read optional unset variables during initialization.
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?exact immutable execution worktree required}"
: "${GNN_INPUT_BUNDLE:?frozen BACE bundle required}"
: "${GNN_MODEL_ROOTS_JSON:?five-classifier bundle map required}"
: "${GNN_EVALUATION_ROOT:?dedicated fresh ablation root required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
echo "CPU-only proposal-fixed evaluation; no GPU request"
extra=()
if [[ ${GNN_EVALUATION_RESUME:-0} == 1 ]]; then extra+=(--resume); fi
exec nice -n 10 python scripts/hpc/gnn/evaluate_bace_gnn_seed7.py \
  --config configs/hpc.yaml --bundle-root "$GNN_INPUT_BUNDLE" \
  --model-roots-json "$GNN_MODEL_ROOTS_JSON" --output-root "$GNN_EVALUATION_ROOT" \
  --cpu-threads "$OMP_NUM_THREADS" --require-cpu-admission "${extra[@]}"
