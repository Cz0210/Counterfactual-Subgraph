#!/bin/bash
#SBATCH --job-name=bace_conn_theta_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

MATRIX_DIR=${MATRIX_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_residual_v3/calibration_matrix}
COMMON_ROOT=${COMMON_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3}
OUTPUT_PATH=${OUTPUT_PATH:-$COMMON_ROOT/thresholds.json}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$MATRIX_DIR/matrix_manifest.json" "$MATRIX_DIR/matrix_audit.json" "$MATRIX_DIR/pair_matrix.jsonl"; do
  test -s "$path" || { echo "[BACE_CONNECTED_THRESHOLD_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
args=(
  python scripts/freeze_bace_connected_thresholds.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --calibration-matrix-dir "$MATRIX_DIR"
  --output-path "$OUTPUT_PATH"
)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_CONNECTED_THRESHOLD_V3_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_PATH" || { echo "[BACE_CONNECTED_THRESHOLD_COLLISION] $OUTPUT_PATH" >&2; exit 2; }
mkdir -p "$COMMON_ROOT"
"${args[@]}"
test -s "$OUTPUT_PATH"
echo "[BACE_CONNECTED_THRESHOLD_V3_SUCCESS]"
