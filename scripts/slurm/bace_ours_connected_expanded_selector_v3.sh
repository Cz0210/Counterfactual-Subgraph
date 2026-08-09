#!/bin/bash
#SBATCH --job-name=bace_ours_conn_select_x_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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

MATRIX_RUN_DIR=${MATRIX_RUN_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_residual_v3_expanded/calibration_matrix}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3_expanded/thresholds.json}
CURRENT_SELECTED_CSV=${CURRENT_SELECTED_CSV:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_residual_v3/selected_top20.csv}
EXPANSION_MANIFEST=${EXPANSION_MANIFEST:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_multiseed_v3/merged/candidate_pool_audit.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_residual_v3_expanded}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$MATRIX_RUN_DIR/pair_matrix.jsonl" "$MATRIX_RUN_DIR/matrix_audit.json" "$THRESHOLDS_JSON" "$CURRENT_SELECTED_CSV" "$EXPANSION_MANIFEST"; do
  test -s "$path" || { echo "[BACE_CONNECTED_EXPANDED_SELECTOR_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
case "$MATRIX_RUN_DIR $THRESHOLDS_JSON $CURRENT_SELECTED_CSV $EXPANSION_MANIFEST" in
  *gcf*|*GCF*|*test*) echo "[BACE_CONNECTED_EXPANDED_SELECTOR_LEAKAGE] forbidden input" >&2; exit 2 ;;
esac
args=(python scripts/select_wnode_prefix_actions.py
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --dataset BACE --split calibration --matrix-run-dir "$MATRIX_RUN_DIR"
  --thresholds-json "$THRESHOLDS_JSON" --current-selected-csv "$CURRENT_SELECTED_CSV"
  --expansion-manifest "$EXPANSION_MANIFEST" --output-dir "$OUTPUT_DIR"
  --fold-count 5 --local-swap-passes 2 --forbid-test --require-connected)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_CONNECTED_EXPANDED_SELECTOR_V3_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_CONNECTED_EXPANDED_SELECTOR_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
test -s "$OUTPUT_DIR/frozen_selection.json"
echo "[BACE_CONNECTED_EXPANDED_SELECTOR_V3_SUCCESS]"
