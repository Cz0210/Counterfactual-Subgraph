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
if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ "${COMMON_RECOURSE_ENGINE:-}" != "external_memory_exact_v1" \
        || "${COMRECGC_COMMON_RECOURSE_RESUME:-0}" != "1" \
        || "$DATASET" != "aids" \
        || ! -d "$OUTPUT_ROOT" \
        || -L "$OUTPUT_ROOT" \
        || ! -s "$OUTPUT_ROOT/continuation_resume_contract.json" \
        || -e "$OUTPUT_ROOT/PASS" ]]; then
    echo "existing OUTPUT_ROOT is not an eligible exact resume: $OUTPUT_ROOT" >&2
    exit 73
  fi
  echo "[COMRECGC_STANDARDIZED_CONTINUATION_RESUME] dataset=$DATASET output=$OUTPUT_ROOT"
fi

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
    --external-dbscan-shortcut-mode "${COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE:-disabled}"
    --external-shortcut-seed-count "${COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT:-3}"
    --external-shortcut-failure-cap "${COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP:-4096}"
    --external-shortcut-query-block-size "${COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE:-65536}"
    --external-exact-fallback-max-samples "${COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES:-100000}"
    --external-summary-block-size "${COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE:-65536}"
    --expected-sklearn-version "${COMRECGC_EXPECTED_SKLEARN_VERSION:-1.7.2}"
  )

  terminal_source="${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST:-}"
  chunk_checkpoint="${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT:-}"
  close_pair_view_manifest="${COMRECGC_EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST:-}"
  source_owner="${COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT:-}"
  vector_cache_root="${COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT:-}"
  vector_cache_lock="${COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK:-}"
  vector_cache_route_lock="${COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK:-}"
  auto_pair_root="${COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT:-}"
  require_promoted_final="${COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL:-0}"
  [[ "$require_promoted_final" == "0" || "$require_promoted_final" == "1" ]] || { echo "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL must be 0 or 1" >&2; exit 64; }
  [[ "$require_promoted_final" == "0" || -n "$auto_pair_root" ]] || { echo "promoted-final requirement needs automatic pair-store root" >&2; exit 64; }
  if [[ -n "$auto_pair_root" ]]; then
    [[ "$auto_pair_root" == /* && -d "$auto_pair_root" && ! -L "$auto_pair_root" ]] || { echo "invalid automatic pair-store root: $auto_pair_root" >&2; exit 64; }
    if [[ -e "$auto_pair_root/run_manifest.json" || -L "$auto_pair_root/run_manifest.json" ]]; then
      [[ -f "$auto_pair_root/run_manifest.json" && -s "$auto_pair_root/run_manifest.json" && ! -L "$auto_pair_root/run_manifest.json" ]] || { echo "invalid promoted pair-store manifest: $auto_pair_root/run_manifest.json" >&2; exit 64; }
      terminal_source="$auto_pair_root/run_manifest.json"
      chunk_checkpoint=""
      vector_cache_root=""
      vector_cache_lock=""
      vector_cache_route_lock=""
      echo "[COMRECGC_PAIR_SOURCE_SELECTED] mode=promoted_final manifest=$terminal_source"
    else
      [[ "$require_promoted_final" == "0" ]] || { echo "required promoted pair-store manifest is absent: $auto_pair_root/run_manifest.json" >&2; exit 75; }
      echo "[COMRECGC_PAIR_SOURCE_SELECTED] mode=closed_chunks checkpoint=$chunk_checkpoint"
    fi
  fi
  if [[ -n "$terminal_source" ]]; then
    [[ -n "$source_owner" ]] || { echo "terminal pair source requires owner root" >&2; exit 64; }
    [[ -z "$chunk_checkpoint$vector_cache_root$vector_cache_lock$vector_cache_route_lock" ]] || { echo "terminal and chunk pair sources are mutually exclusive" >&2; exit 64; }
    args+=(
      --external-pair-store-source-manifest "$terminal_source"
      --external-pair-store-source-owner-root "$source_owner"
    )
    if [[ -n "$close_pair_view_manifest" ]]; then
      [[ -f "$close_pair_view_manifest" ]] || { echo "invalid close-pair view manifest: $close_pair_view_manifest" >&2; exit 64; }
      args+=(--external-close-pair-view-manifest "$close_pair_view_manifest")
    fi
  elif [[ -n "$chunk_checkpoint$close_pair_view_manifest$vector_cache_root$vector_cache_lock$vector_cache_route_lock$source_owner" ]]; then
    [[ -n "$chunk_checkpoint" && -n "$close_pair_view_manifest" && -f "$close_pair_view_manifest" && -n "$source_owner" && -n "$vector_cache_root" && -n "$vector_cache_lock" && -n "$vector_cache_route_lock" ]] || { echo "chunk-source/cache/close-view environment is incomplete" >&2; exit 64; }
    args+=(
      --external-pair-store-source-checkpoint "$chunk_checkpoint"
      --external-pair-store-source-owner-root "$source_owner"
      --external-close-pair-view-manifest "$close_pair_view_manifest"
      --external-vector-cache-root "$vector_cache_root"
      --external-vector-cache-lock "$vector_cache_lock"
      --external-vector-cache-route-lock "$vector_cache_route_lock"
      --external-vector-cache-min-free-gb "${COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB:-3}"
      --external-vector-cache-proc-root "${COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT:-/proc}"
    )
  fi
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
