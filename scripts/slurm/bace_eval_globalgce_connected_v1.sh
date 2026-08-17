#!/bin/bash
#SBATCH --job-name=bace_globalgce_eval
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
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

SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_globalgce_frequency_top20_connected_v1}
CANDIDATE_PATH=${CANDIDATE_PATH:-$SELECTOR_ROOT/selected_top20_for_eval.csv}
SELECTION_MANIFEST=${SELECTION_MANIFEST:-$SELECTOR_ROOT/frozen_selection.json}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol/thresholds.json}
REFERENCE_ROOT=${REFERENCE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/ours}
WORK_DIR=${WORK_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_common4_connected_residual_v1_work/globalgce}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_residual_v1/globalgce}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$CANDIDATE_PATH" "$SELECTION_MANIFEST" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV" "$THRESHOLDS_JSON" "$REFERENCE_ROOT/final_artifact_audit.json"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done

args=(
  python scripts/evaluate_bace_method.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --method globalgce
  --candidate-path "$CANDIDATE_PATH"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --test-csv "$TEST_CSV"
  --thresholds-json "$THRESHOLDS_JSON"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --expected-test-parents 116
  --device cpu
  --selection-manifest "$SELECTION_MANIFEST"
  --test-evaluation-count 1
  --reference-artifact-root "$REFERENCE_ROOT"
  --action-semantics-version connected_sanitized_residual_v1
  --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1
  --wnode-cache-db "$WNODE_CACHE_DB"
)
echo "hostname=$(hostname)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then
  echo '[BACE_GLOBALGCE_EVAL_VALIDATE_OK]'
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
test -s "$OUTPUT_DIR/final_artifact_audit.json"
echo '[BACE_GLOBALGCE_EVAL_SUCCESS]'
