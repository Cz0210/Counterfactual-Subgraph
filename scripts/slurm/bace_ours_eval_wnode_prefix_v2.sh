#!/bin/bash
#SBATCH --job-name=bace_ours_eval_v2
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

SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_wnode_prefix_v2}
CANDIDATE_PATH=${CANDIDATE_PATH:-$SELECTOR_ROOT}
SELECTION_MANIFEST=${SELECTION_MANIFEST:-$SELECTOR_ROOT/frozen_selection.json}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
WORK_DIR=${WORK_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_prefix_v2_work/test_run}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_ours_wnode_prefix_v2}
REFERENCE_ARTIFACT_ROOT=${REFERENCE_ARTIFACT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_ours_wnode}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
RESUME_EXISTING_TEST_RUN=${RESUME_EXISTING_TEST_RUN:-0}

for path in "$SELECTOR_ROOT/selector_audit.json" "$SELECTION_MANIFEST" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV" "$THRESHOLDS_JSON" "$REFERENCE_ARTIFACT_ROOT/run_manifest.json"; do
  test -s "$path" || { echo "[BACE_V2_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
python scripts/audit_bace_ours_wnode_prefix_v2.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --mode selector --root "$SELECTOR_ROOT" \
  --output-json "/tmp/bace_ours_selector_audit_${SLURM_JOB_ID:-validate}.json"
EXPECTED_TEST_PARENTS=$(awk 'END {print NR-1}' "$TEST_CSV")
test "$EXPECTED_TEST_PARENTS" -eq 116
args=(
  python scripts/evaluate_bace_method.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --method ours
  --candidate-path "$CANDIDATE_PATH"
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
  --reference-artifact-root "$REFERENCE_ARTIFACT_ROOT"
)
if [ "$RESUME_EXISTING_TEST_RUN" = 1 ]; then
  args+=(--resume)
fi
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_OURS_EVAL_V2_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_V2_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
echo "[BACE_OURS_EVAL_V2_SUCCESS]"
