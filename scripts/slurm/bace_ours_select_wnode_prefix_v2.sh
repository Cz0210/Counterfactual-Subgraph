#!/bin/bash
#SBATCH --job-name=bace_ours_select_v2
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

MATRIX_RUN_DIR=${MATRIX_RUN_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_wnode_prefix_v2/calibration_matrix}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
CURRENT_SELECTED_CSV=${CURRENT_SELECTED_CSV:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_top20/selected_subgraphs.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_wnode_prefix_v2}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$MATRIX_RUN_DIR/pair_matrix.jsonl" "$THRESHOLDS_JSON" "$CURRENT_SELECTED_CSV"; do
  test -s "$path" || { echo "[BACE_V2_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
case "$MATRIX_RUN_DIR $THRESHOLDS_JSON $CURRENT_SELECTED_CSV" in
  *gcf*|*GCF*|*test*) echo "[BACE_V2_LEAKAGE_ERROR] forbidden selector input" >&2; exit 2 ;;
esac

args=(
  python scripts/select_wnode_prefix_actions.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --dataset BACE
  --split calibration
  --matrix-run-dir "$MATRIX_RUN_DIR"
  --thresholds-json "$THRESHOLDS_JSON"
  --current-selected-csv "$CURRENT_SELECTED_CSV"
  --output-dir "$OUTPUT_DIR"
  --fold-count 5
  --local-swap-passes 2
  --forbid-test
)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_OURS_SELECTOR_V2_VALIDATE_OK]"
  exit 0
fi
"${args[@]}"
python scripts/audit_bace_ours_wnode_prefix_v2.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --mode selector --root "$OUTPUT_DIR" \
  --allow-provisional \
  --output-json "$OUTPUT_DIR/selector_audit.json"
echo "[BACE_OURS_SELECTOR_V2_SUCCESS]"
