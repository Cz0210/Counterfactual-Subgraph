#!/bin/bash
#SBATCH --job-name=bace_ours_conn_test_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
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

SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_residual_v3}
SELECTION_MANIFEST=${SELECTION_MANIFEST:-$SELECTOR_ROOT/frozen_selection.json}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3/thresholds.json}
WORK_DIR=${WORK_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_connected_residual_v3_work}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3/ours}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$SELECTOR_ROOT/connected_selection_gate.json" "$SELECTION_MANIFEST" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV" "$THRESHOLDS_JSON"; do
  test -s "$path" || { echo "[BACE_CONNECTED_TEST_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
python scripts/audit_bace_ours_wnode_prefix_v2.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --mode selector --root "$SELECTOR_ROOT" --require-connected \
  --output-json "/tmp/bace_connected_selector_${SLURM_JOB_ID:-validate}.json"
test "$(awk 'END {print NR-1}' "$TEST_CSV")" -eq 116
args=(
  python scripts/evaluate_bace_method.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --method ours
  --candidate-path "$SELECTOR_ROOT"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --test-csv "$TEST_CSV"
  --thresholds-json "$THRESHOLDS_JSON"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --expected-test-parents 116
  --selection-manifest "$SELECTION_MANIFEST"
  --test-evaluation-count 1
  --action-semantics-version connected_sanitized_residual_v1
  --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1
  --wnode-cache-db "$WNODE_CACHE_DB"
)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_CONNECTED_TEST_V3_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_CONNECTED_TEST_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
test ! -e "$WORK_DIR/test" || { echo "[BACE_CONNECTED_TEST_WORK_COLLISION] $WORK_DIR/test" >&2; exit 2; }
"${args[@]}"
python scripts/audit_bace_ours_wnode_prefix_v2.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --mode final --root "$OUTPUT_DIR" --selector-root "$SELECTOR_ROOT" \
  --require-connected --output-json "$OUTPUT_DIR/connected_final_gate.json"
echo "[BACE_CONNECTED_TEST_V3_SUCCESS]"
