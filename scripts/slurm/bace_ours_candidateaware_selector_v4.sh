#!/bin/bash
#SBATCH --job-name=bace_ours_selector_v4
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
MATRIX_ROOT=${MATRIX_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_candidateaware_v4/final_calibration_matrix}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol/thresholds.json}
CURRENT_SELECTED_CSV=${CURRENT_SELECTED_CSV:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_residual_v3_expanded/selected_top20.csv}
EXPANSION_MANIFEST=${EXPANSION_MANIFEST:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/merged/candidate_pool_audit.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_candidateaware_v4}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$MATRIX_ROOT/pair_matrix.jsonl" "$MATRIX_ROOT/matrix_audit.json" "$THRESHOLDS_JSON" "$CURRENT_SELECTED_CSV" "$EXPANSION_MANIFEST"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
case "$MATRIX_ROOT $THRESHOLDS_JSON $CURRENT_SELECTED_CSV $EXPANSION_MANIFEST" in *test*|*gcf*|*GCF*) echo "forbidden selector input" >&2; exit 2;; esac
args=(python scripts/select_wnode_prefix_actions.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --dataset BACE --split calibration --matrix-run-dir "$MATRIX_ROOT" --thresholds-json "$THRESHOLDS_JSON" --current-selected-csv "$CURRENT_SELECTED_CSV" --expansion-manifest "$EXPANSION_MANIFEST" --output-dir "$OUTPUT_DIR" --fold-count 5 --local-swap-passes 5 --forbid-test --require-connected)
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_OURS_SELECTOR_V4_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "collision $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python - "$OUTPUT_DIR/frozen_selection.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["selection_frozen"] and p["selection_split"]=="calibration" and p["test_used"] is False and p["gcf_result_used"] is False and p["ranks"]==list(range(1,21))
PY
echo '[BACE_OURS_SELECTOR_V4_SUCCESS]'
