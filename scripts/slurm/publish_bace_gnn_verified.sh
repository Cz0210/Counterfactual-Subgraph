#!/usr/bin/env bash
# Authorized CPU-only offline audit: no GNN inference, training, or OT rerun.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable verifier worktree required}"
: "${GNN_INPUT_BUNDLE:?exact frozen input required}"
: "${GNN_EVALUATION_ROOT:?completed evaluation required}"
: "${GNN_PUBLICATION_ROOT:?fresh publication overlay required}"
: "${GNN_ENVIRONMENT_MANIFEST:?environment manifest required}"
: "${GNN_PUBLICATION_DRIVER_COMMIT:?exact verifier commit required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
echo "Offline artifact replay; no model inference or heuristic fallback"
args=()
if [[ -n ${GNN_CALIBRATION_PREDICTION_ROOT:-} ]]; then
  args+=(--calibration-prediction-root "$GNN_CALIBRATION_PREDICTION_ROOT")
fi
exec nice -n 10 python scripts/hpc/gnn/publish_bace_gnn_verified.py \
  --config configs/hpc.yaml --bundle-root "$GNN_INPUT_BUNDLE" \
  --evaluation-root "$GNN_EVALUATION_ROOT" --output-root "$GNN_PUBLICATION_ROOT" \
  --environment-manifest "$GNN_ENVIRONMENT_MANIFEST" \
  --driver-commit "$GNN_PUBLICATION_DRIVER_COMMIT" "${args[@]}"
