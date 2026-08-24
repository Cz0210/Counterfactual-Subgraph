#!/bin/bash
#SBATCH --job-name=comrecgc_recourse
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'

DATASET="${DATASET:-}"
MODE="${MODE:-smoke}"
RESUME="${RESUME:-false}"
ENGINE="${ENGINE:-legacy_in_memory}"
EXTERNAL_MAX_RSS_GB="${EXTERNAL_MAX_RSS_GB:-96}"
EXTERNAL_QUERY_BLOCK_SIZE="${EXTERNAL_QUERY_BLOCK_SIZE:-8}"
EXTERNAL_DBSCAN_SHORTCUT_MODE="${EXTERNAL_DBSCAN_SHORTCUT_MODE:-disabled}"
EXTERNAL_SHORTCUT_SEED_COUNT="${EXTERNAL_SHORTCUT_SEED_COUNT:-3}"
EXTERNAL_SHORTCUT_FAILURE_CAP="${EXTERNAL_SHORTCUT_FAILURE_CAP:-4096}"
EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE="${EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE:-65536}"
EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES="${EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES:-100000}"
EXTERNAL_SUMMARY_BLOCK_SIZE="${EXTERNAL_SUMMARY_BLOCK_SIZE:-65536}"
EXTERNAL_PAIR_STORE_SOURCE_MANIFEST="${EXTERNAL_PAIR_STORE_SOURCE_MANIFEST:-}"
EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT="${EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT:-}"
EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT="${EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT:-}"
EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST="${EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST:-}"
EXTERNAL_PAIR_STORE_AUTO_ROOT="${EXTERNAL_PAIR_STORE_AUTO_ROOT:-}"
COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL="${COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL:-0}"
EXTERNAL_VECTOR_CACHE_ROOT="${EXTERNAL_VECTOR_CACHE_ROOT:-}"
EXTERNAL_VECTOR_CACHE_LOCK="${EXTERNAL_VECTOR_CACHE_LOCK:-}"
EXTERNAL_VECTOR_CACHE_ROUTE_LOCK="${EXTERNAL_VECTOR_CACHE_ROUTE_LOCK:-}"
EXTERNAL_VECTOR_CACHE_MIN_FREE_GB="${EXTERNAL_VECTOR_CACHE_MIN_FREE_GB:-3}"
EXTERNAL_VECTOR_CACHE_PROC_ROOT="${EXTERNAL_VECTOR_CACHE_PROC_ROOT:-/proc}"
EXPECTED_SKLEARN_VERSION="${EXPECTED_SKLEARN_VERSION:-1.7.2}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || exit 2
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || exit 2
if [[ "$MODE" == "smoke" ]]; then PARENT_LIMIT_EXPECTED=64; elif [[ "$DATASET" == "aids" ]]; then PARENT_LIMIT_EXPECTED=1283; else PARENT_LIMIT_EXPECTED=1448; fi
PARENT_LIMIT="${PARENT_LIMIT:-$PARENT_LIMIT_EXPECTED}"
[[ "$PARENT_LIMIT" == "$PARENT_LIMIT_EXPECTED" ]] || { echo "[COMRECGC_CONFIG_ERROR] parent_limit=$PARENT_LIMIT expected=$PARENT_LIMIT_EXPECTED" >&2; exit 2; }
BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/$DATASET/${MODE}_v1}"
GENERATION_DIR="${GENERATION_DIR:-$BASE_ROOT/generation}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/common_recourse}"
COMRECGC_EXPECTED_COMMIT="${COMRECGC_EXPECTED_COMMIT:-122f9341a360e9f06bb58a2f5823bb596021f6bf}"
COMRECGC_ROOT="${COMRECGC_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/$COMRECGC_EXPECTED_COMMIT}"
if [[ "$RESUME" != "true" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[COMRECGC_CONFIG_ERROR] non-empty output with RESUME=false: $OUTPUT_DIR" >&2; exit 2
fi
if [[ "$DATASET" == "aids" ]]; then
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/gcfexplainer_hiv_csv/dataset}"
  SOURCE_CSV="${SOURCE_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt}"
  SOURCE_ARGS=(--source-csv "$SOURCE_CSV")
else
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
  DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}"
  SOURCE_ARGS=()
fi
[[ -s "$GENERATION_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] generation incomplete" >&2; exit 2; }
python scripts/verify_comrecgc_checkout.py --config configs/hpc.yaml \
  --root "$COMRECGC_ROOT" --expected-commit "$COMRECGC_EXPECTED_COMMIT" \
  --validate-imports
echo "[COMRECGC_STAGE_CONFIG] stage=common_recourse dataset=$DATASET mode=$MODE output_dir=$OUTPUT_DIR"
RESUME_ARGS=(); [[ "$RESUME" == "true" ]] && RESUME_ARGS=(--resume)
ENGINE_ARGS=(--engine "$ENGINE")
if [[ "$ENGINE" == "external_memory_exact_v1" ]]; then
  [[ "$DATASET" == "aids" ]] || { echo "[COMRECGC_CONFIG_ERROR] external engine is AIDS-only" >&2; exit 2; }
  DEVICE="${DEVICE:-cpu}"
  [[ "$DEVICE" == "cpu" ]] || { echo "[COMRECGC_CONFIG_ERROR] AIDS external engine is CPU-only" >&2; exit 2; }
  ENGINE_ARGS+=(--external-max-rss-gb "$EXTERNAL_MAX_RSS_GB" --external-query-block-size "$EXTERNAL_QUERY_BLOCK_SIZE" --external-dbscan-shortcut-mode "$EXTERNAL_DBSCAN_SHORTCUT_MODE" --external-shortcut-seed-count "$EXTERNAL_SHORTCUT_SEED_COUNT" --external-shortcut-failure-cap "$EXTERNAL_SHORTCUT_FAILURE_CAP" --external-shortcut-query-block-size "$EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE" --external-exact-fallback-max-samples "$EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES" --external-summary-block-size "$EXTERNAL_SUMMARY_BLOCK_SIZE" --expected-sklearn-version "$EXPECTED_SKLEARN_VERSION")
  [[ "$COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL" == "0" || "$COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL" == "1" ]] || { echo "[COMRECGC_CONFIG_ERROR] promoted-final requirement must be 0 or 1" >&2; exit 2; }
  [[ "$COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL" == "0" || -n "$EXTERNAL_PAIR_STORE_AUTO_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] promoted-final requirement needs automatic root" >&2; exit 2; }
  if [[ -n "$EXTERNAL_PAIR_STORE_AUTO_ROOT" ]]; then
    [[ "$EXTERNAL_PAIR_STORE_AUTO_ROOT" == /* && -d "$EXTERNAL_PAIR_STORE_AUTO_ROOT" && ! -L "$EXTERNAL_PAIR_STORE_AUTO_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] invalid automatic pair-store root" >&2; exit 2; }
    if [[ -e "$EXTERNAL_PAIR_STORE_AUTO_ROOT/run_manifest.json" || -L "$EXTERNAL_PAIR_STORE_AUTO_ROOT/run_manifest.json" ]]; then
      [[ -f "$EXTERNAL_PAIR_STORE_AUTO_ROOT/run_manifest.json" && -s "$EXTERNAL_PAIR_STORE_AUTO_ROOT/run_manifest.json" && ! -L "$EXTERNAL_PAIR_STORE_AUTO_ROOT/run_manifest.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] invalid promoted pair-store manifest" >&2; exit 2; }
      EXTERNAL_PAIR_STORE_SOURCE_MANIFEST="$EXTERNAL_PAIR_STORE_AUTO_ROOT/run_manifest.json"
      EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT=""
      EXTERNAL_VECTOR_CACHE_ROOT=""
      EXTERNAL_VECTOR_CACHE_LOCK=""
      EXTERNAL_VECTOR_CACHE_ROUTE_LOCK=""
      echo "[COMRECGC_PAIR_SOURCE_SELECTED] mode=promoted_final manifest=$EXTERNAL_PAIR_STORE_SOURCE_MANIFEST"
    else
      [[ "$COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL" == "0" ]] || { echo "[COMRECGC_CONFIG_ERROR] required promoted pair-store manifest is absent" >&2; exit 75; }
      echo "[COMRECGC_PAIR_SOURCE_SELECTED] mode=closed_chunks checkpoint=$EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT"
    fi
  fi
  CHUNK_SOURCE_COUNT=0
  [[ -n "$EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT" ]] && CHUNK_SOURCE_COUNT=$((CHUNK_SOURCE_COUNT + 1))
  [[ -n "$EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" ]] && CHUNK_SOURCE_COUNT=$((CHUNK_SOURCE_COUNT + 1))
  [[ -n "$EXTERNAL_VECTOR_CACHE_ROOT" ]] && CHUNK_SOURCE_COUNT=$((CHUNK_SOURCE_COUNT + 1))
  [[ -n "$EXTERNAL_VECTOR_CACHE_LOCK" ]] && CHUNK_SOURCE_COUNT=$((CHUNK_SOURCE_COUNT + 1))
  [[ -n "$EXTERNAL_VECTOR_CACHE_ROUTE_LOCK" ]] && CHUNK_SOURCE_COUNT=$((CHUNK_SOURCE_COUNT + 1))
  if [[ -n "$EXTERNAL_PAIR_STORE_SOURCE_MANIFEST" ]]; then
    [[ -n "$EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] terminal source requires owner root" >&2; exit 2; }
    [[ "$CHUNK_SOURCE_COUNT" -eq 1 ]] || { echo "[COMRECGC_CONFIG_ERROR] terminal and chunk sources are mutually exclusive" >&2; exit 2; }
    ENGINE_ARGS+=(--external-pair-store-source-manifest "$EXTERNAL_PAIR_STORE_SOURCE_MANIFEST" --external-pair-store-source-owner-root "$EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT")
    if [[ -n "$EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST" ]]; then
      [[ -s "$EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST" ]] || { echo "[COMRECGC_CONFIG_ERROR] invalid close-pair view manifest" >&2; exit 2; }
      ENGINE_ARGS+=(--external-close-pair-view-manifest "$EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST")
    fi
  else
    [[ "$CHUNK_SOURCE_COUNT" -eq 0 || "$CHUNK_SOURCE_COUNT" -eq 5 ]] || { echo "[COMRECGC_CONFIG_ERROR] chunk source/cache arguments are all-or-none" >&2; exit 2; }
  fi
  if [[ -z "$EXTERNAL_PAIR_STORE_SOURCE_MANIFEST" && "$CHUNK_SOURCE_COUNT" -eq 5 ]]; then
    [[ -s "$EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST" ]] || { echo "[COMRECGC_CONFIG_ERROR] Cartesian snapshot requires a validated theta-close view manifest" >&2; exit 2; }
    ENGINE_ARGS+=(--external-pair-store-source-checkpoint "$EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT" --external-pair-store-source-owner-root "$EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT" --external-close-pair-view-manifest "$EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST" --external-vector-cache-root "$EXTERNAL_VECTOR_CACHE_ROOT" --external-vector-cache-lock "$EXTERNAL_VECTOR_CACHE_LOCK" --external-vector-cache-route-lock "$EXTERNAL_VECTOR_CACHE_ROUTE_LOCK" --external-vector-cache-min-free-gb "$EXTERNAL_VECTOR_CACHE_MIN_FREE_GB" --external-vector-cache-proc-root "$EXTERNAL_VECTOR_CACHE_PROC_ROOT")
  fi
else
  DEVICE="${DEVICE:-cuda:0}"
fi
python scripts/baselines/comrecgc/run_common_recourse.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --dataset "$DATASET" --mode "$MODE" --upstream-root "$COMRECGC_ROOT" \
  --dataset-dir "$DATASET_DIR" "${SOURCE_ARGS[@]}" --generation-dir "$GENERATION_DIR" \
  --distance-checkpoint "$DISTANCE_CHECKPOINT" --output-dir "$OUTPUT_DIR" \
  --parent-limit "$PARENT_LIMIT" --device "$DEVICE" "${ENGINE_ARGS[@]}" "${RESUME_ARGS[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_COMMON_RECOURSE_SUCCESS]"
