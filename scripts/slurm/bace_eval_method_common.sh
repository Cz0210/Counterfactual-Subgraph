#!/bin/bash

set -euo pipefail

: "${BACE_METHOD:?BACE_METHOD is required}"
: "${CANDIDATE_PATH:?CANDIDATE_PATH is required}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
DATA_DIR=${DATA_DIR:-data/processed/BACE}
TEACHER_ROOT=${TEACHER_ROOT:-outputs/hpc/oracle/bace}
TEACHER_PATH=${TEACHER_PATH:-$TEACHER_ROOT/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}
PAPER_ROOT=${PAPER_ROOT:-outputs/hpc/eval/paper/bace_common3_standardized_v1}
WORK_ROOT=${WORK_ROOT:-outputs/hpc/eval/bace_wnode_v1}
TEST_CSV=${TEST_CSV:-$TEACHER_ROOT/teacher_consistent/test_source_label1_teacher_correct.csv}
CALIBRATION_CSV=${CALIBRATION_CSV:-$TEACHER_ROOT/teacher_consistent/calibration_source_label1_teacher_correct.csv}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$PAPER_ROOT/thresholds.json}
OUTPUT_DIR=${OUTPUT_DIR:-$PAPER_ROOT/$BACE_METHOD}
WORK_DIR=${WORK_DIR:-$WORK_ROOT/$BACE_METHOD}
RESUME=${RESUME:-false}

cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

for required in "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV"; do
  if [ ! -s "$required" ]; then
    echo "[BACE_CONFIG_ERROR] missing required file: $required" >&2
    exit 2
  fi
done
if [ ! -e "$CANDIDATE_PATH" ]; then
  echo "[BACE_CONFIG_ERROR] missing frozen candidate input: $CANDIDATE_PATH" >&2
  exit 2
fi
if [ "$BACE_METHOD" != "ours" ] && [ ! -s "$THRESHOLDS_JSON" ]; then
  echo "[BACE_CONFIG_ERROR] run Ours calibration first; missing: $THRESHOLDS_JSON" >&2
  exit 2
fi
if [ "$RESUME" != "true" ] && { [ -d "$OUTPUT_DIR" ] || [ -d "$WORK_DIR" ]; }; then
  echo "[BACE_CONFIG_ERROR] output/work path exists while RESUME=false" >&2
  echo "output_dir=$OUTPUT_DIR" >&2
  echo "work_dir=$WORK_DIR" >&2
  exit 2
fi

EXPECTED_TEST_PARENTS=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$TEST_CSV")
if [ "$EXPECTED_TEST_PARENTS" -le 0 ]; then
  echo "[BACE_CONFIG_ERROR] empty test source cohort: $TEST_CSV" >&2
  exit 2
fi

echo "hostname=$(hostname)"
echo "date=$(date -Is)"
echo "pwd=$(pwd)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "method=$BACE_METHOD"
echo "candidate_path=$CANDIDATE_PATH"
echo "teacher_path=$TEACHER_PATH"
echo "test_csv=$TEST_CSV"
echo "expected_test_parents=$EXPECTED_TEST_PARENTS"
echo "thresholds_json=$THRESHOLDS_JSON"
echo "output_dir=$OUTPUT_DIR"

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --method "$BACE_METHOD"
  --candidate-path "$CANDIDATE_PATH"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --test-csv "$TEST_CSV"
  --thresholds-json "$THRESHOLDS_JSON"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --expected-test-parents "$EXPECTED_TEST_PARENTS"
)
if [ "$BACE_METHOD" = "ours" ]; then
  if [ ! -s "$CALIBRATION_CSV" ]; then
    echo "[BACE_CONFIG_ERROR] missing Ours calibration cohort: $CALIBRATION_CSV" >&2
    exit 2
  fi
  args+=(--calibrate-thresholds --calibration-csv "$CALIBRATION_CSV")
fi
if [ "$RESUME" = "true" ]; then
  args+=(--resume)
fi

python scripts/evaluate_bace_method.py "${args[@]}"

test -s "$OUTPUT_DIR/figure3_coverage_vs_k.csv"
test -s "$OUTPUT_DIR/figure4_coverage_vs_threshold.csv"
test -s "$OUTPUT_DIR/table2_${BACE_METHOD}_k10.csv"
test -s "$OUTPUT_DIR/summary.json"
test -s "$OUTPUT_DIR/run_manifest.json"
test -s "$OUTPUT_DIR/final_artifact_audit.json"
echo "[BACE_METHOD_EVALUATION_SUCCESS] method=$BACE_METHOD"
