#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"

: "${DATASET:?DATASET must be aids or mutagenicity}"
: "${SOURCE_GENERATION_ROOT:?SOURCE_GENERATION_ROOT is required}"
: "${COMRECGC_UPSTREAM_ROOT:?COMRECGC_UPSTREAM_ROOT is required}"
: "${DATASET_DIR:?DATASET_DIR is required}"
: "${DISTANCE_CHECKPOINT:?DISTANCE_CHECKPOINT is required}"
: "${DATASET_CSV:?DATASET_CSV is required}"
: "${TEACHER_PATH:?TEACHER_PATH is required}"
: "${MOLCLR_ROOT:?MOLCLR_ROOT is required}"
: "${MOLCLR_CHECKPOINT:?MOLCLR_CHECKPOINT is required}"
: "${THRESHOLDS_PATH:?THRESHOLDS_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT fresh persistent path is required}"

[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || exit 64
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 66; }
for path in "$SOURCE_GENERATION_ROOT" "$COMRECGC_UPSTREAM_ROOT" "$DATASET_DIR" \
  "$DISTANCE_CHECKPOINT" "$DATASET_CSV" "$TEACHER_PATH" "$MOLCLR_ROOT" \
  "$MOLCLR_CHECKPOINT" "$THRESHOLDS_PATH" "$OUTPUT_ROOT"; do
  [[ "$path" == /* ]] || { echo "absolute path required: $path" >&2; exit 64; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "fresh OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2; exit 73; }

args=(
  --dataset "$DATASET"
  --source-generation-root "$SOURCE_GENERATION_ROOT"
  --upstream-root "$COMRECGC_UPSTREAM_ROOT"
  --dataset-dir "$DATASET_DIR"
  --distance-checkpoint "$DISTANCE_CHECKPOINT"
  --dataset-csv "$DATASET_CSV"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --thresholds-path "$THRESHOLDS_PATH"
  --output-root "$OUTPUT_ROOT"
  --device "${DEVICE:-cuda:0}"
)
if [[ "$DATASET" == "aids" ]]; then
  : "${SOURCE_CSV:?SOURCE_CSV is required for AIDS}"
  args+=(--source-csv "$SOURCE_CSV" --theta-star "${THETA_STAR:-0.05}" --cost-cap "${COST_CAP:-0.0535}")
fi
if [[ -n "${COMMON_RECOURSE_ENGINE:-}" ]]; then
  args+=(--common-recourse-engine "$COMMON_RECOURSE_ENGINE")
fi
if [[ "${COMMON_RECOURSE_ENGINE:-}" == "external_memory_exact_v1" ]]; then
  args+=(
    --external-max-rss-gb "${COMRECGC_EXTERNAL_MAX_RSS_GB:-96}"
    --external-query-block-size "${COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE:-8}"
    --external-checkpoint-interval-blocks "${COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS:-1}"
    --expected-sklearn-version "${COMRECGC_EXPECTED_SKLEARN_VERSION:-1.7.2}"
  )
  if [[ "${COMRECGC_COMMON_RECOURSE_RESUME:-0}" == "1" ]]; then
    args+=(--common-recourse-resume)
  fi
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "[COMRECGC_STANDARDIZED_CONTINUATION_START] dataset=$DATASET output=$OUTPUT_ROOT"
exec "$PYTHON" "$PROJECT_ROOT/scripts/autodl/run_comrecgc_standardized_continuation.py" "${args[@]}"
