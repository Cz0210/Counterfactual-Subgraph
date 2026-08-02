#!/bin/bash
# Submit only through scripts/exp_sbatch.sh; full train-cohort generation from explicit checkpoints.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=mut_clear_genfull
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[MUTAGENICITY_CLEAR_FULL_GENERATION_CONFIG_ERROR] Could not determine PROJECT_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

config_error() {
  echo "[MUTAGENICITY_CLEAR_FULL_GENERATION_CONFIG_ERROR] $*" >&2
  exit 2
}

resolve_from_project_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

PHASE_A_ROOT="${PHASE_A_ROOT:-outputs/hpc/mutagenicity/final/clear_phase_a_dataset_codec_best}"
GENERATION_CSV="${GENERATION_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/train_source_label1_teacher_correct.csv}"
TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-baselines/clear_official}"
[[ -n "${SOURCE_RUN_ROOT:-}" ]] || config_error "SOURCE_RUN_ROOT must be explicit"
[[ -n "${GRAPHPRED_CHECKPOINT:-}" ]] || config_error "GRAPHPRED_CHECKPOINT must be explicit"
[[ -n "${GRAPHCFE_CHECKPOINT:-}" ]] || config_error "GRAPHCFE_CHECKPOINT must be explicit"
[[ -n "${OUTPUT_DIR:-}" ]] || config_error "OUTPUT_DIR must be an explicit new full output directory"

PHASE_A_ROOT="$(resolve_from_project_root "$PHASE_A_ROOT")"
GENERATION_CSV="$(resolve_from_project_root "$GENERATION_CSV")"
TEACHER_PATH="$(resolve_from_project_root "$TEACHER_PATH")"
OFFICIAL_ROOT="$(resolve_from_project_root "$OFFICIAL_ROOT")"
SOURCE_RUN_ROOT="$(resolve_from_project_root "$SOURCE_RUN_ROOT")"
GRAPHPRED_CHECKPOINT="$(resolve_from_project_root "$GRAPHPRED_CHECKPOINT")"
GRAPHCFE_CHECKPOINT="$(resolve_from_project_root "$GRAPHCFE_CHECKPOINT")"
OUTPUT_DIR="$(resolve_from_project_root "$OUTPUT_DIR")"

PARENT_LIMIT="${PARENT_LIMIT:-1448}"
GENERATION_CHUNK_SIZE="${GENERATION_CHUNK_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-13}"
RESUME="${RESUME:-true}"
DEVICE="${DEVICE:-cuda}"
MODEL_TRAINING_PERFORMED=false
CALIBRATION_LOADED=false
TEST_LOADED=false

if [[ "$PARENT_LIMIT" -ne 1448 || "$GENERATION_CHUNK_SIZE" -ne 16 || "$SEED" -ne 13 ]]; then
  config_error "Full generation requires parent_limit=1448, chunk_size=16, seed=13"
fi
if [[ "$NUM_WORKERS" -ne 0 ]]; then
  config_error "NUM_WORKERS must remain 0"
fi

case "$(printf '%s' "$RESUME" | tr '[:upper:]' '[:lower:]')" in
  true|1|yes|on)
    RESUME_ENABLED=true
    RESUME_FLAG="--resume"
    ;;
  false|0|no|off)
    RESUME_ENABLED=false
    RESUME_FLAG="--no-resume"
    ;;
  *) config_error "RESUME must be true or false" ;;
esac

if [[ "$OUTPUT_DIR" == "$SOURCE_RUN_ROOT" ]]; then
  config_error "OUTPUT_DIR must not overwrite SOURCE_RUN_ROOT"
fi
for required in "$GENERATION_CSV" "$TEACHER_PATH" "$GRAPHPRED_CHECKPOINT" "$GRAPHCFE_CHECKPOINT"; do
  if [[ ! -s "$required" ]]; then
    config_error "Missing or empty required input: $required"
  fi
done
for required_dir in "$PHASE_A_ROOT" "$OFFICIAL_ROOT" "$SOURCE_RUN_ROOT"; do
  if [[ ! -d "$required_dir" ]]; then
    config_error "Missing required directory: $required_dir"
  fi
done
if [[ -e "$OUTPUT_DIR/_RUN_COMPLETE.json" || -e "$OUTPUT_DIR/_FINALIZED.json" ]]; then
  config_error "Completed or finalized output cannot be rerun: $OUTPUT_DIR"
fi
if [[ -e "$OUTPUT_DIR/_RUN_FAILED.json" && "$RESUME_ENABLED" != true ]]; then
  config_error "Failed output requires RESUME=true: $OUTPUT_DIR"
fi

mkdir -p "$PROJECT_ROOT/logs"

echo "===== MUTAGENICITY CLEAR GENERATION-ONLY FULL ====="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "python=$(command -v python)"
echo "git_commit=$(git rev-parse HEAD)"
echo "SOURCE_RUN_ROOT=$SOURCE_RUN_ROOT"
echo "GRAPHPRED_CHECKPOINT=$GRAPHPRED_CHECKPOINT"
echo "GRAPHCFE_CHECKPOINT=$GRAPHCFE_CHECKPOINT"
echo "GENERATION_CSV=$GENERATION_CSV"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PARENT_LIMIT=$PARENT_LIMIT"
echo "GENERATION_CHUNK_SIZE=$GENERATION_CHUNK_SIZE"
echo "SEED=$SEED"
echo "MODEL_TRAINING_PERFORMED=$MODEL_TRAINING_PERFORMED"
echo "CALIBRATION_LOADED=$CALIBRATION_LOADED"
echo "TEST_LOADED=$TEST_LOADED"
python --version
nvidia-smi || true

python scripts/baselines/clear/build_mutagenicity_train_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --phase-a-root "$PHASE_A_ROOT" \
  --generation-csv "$GENERATION_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --official-root "$OFFICIAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --generation-only \
  --generation-profile full \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --graphpred-checkpoint "$GRAPHPRED_CHECKPOINT" \
  --graphcfe-checkpoint "$GRAPHCFE_CHECKPOINT" \
  --parent-limit "$PARENT_LIMIT" \
  --generation-chunk-size "$GENERATION_CHUNK_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "$RESUME_FLAG" \
  --forbid-calibration-test

python scripts/baselines/clear/audit_mutagenicity_train_pool.py \
  --config configs/hpc.yaml \
  --run-dir "$OUTPUT_DIR" \
  --generation-csv "$GENERATION_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --expected-model-train-rows 2885 \
  --expected-model-val-rows 355 \
  --expected-generation-parent-rows 1448 \
  --expected-selected-parents 1448 \
  --expected-generation-profile full \
  --require-generation-only \
  --require-target-label-zero \
  --require-unique-universe \
  --forbid-calibration-test \
  --require-complete

test -s "$OUTPUT_DIR/feature_decoding_summary.json"
test -s "$OUTPUT_DIR/train_pool_audit.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_CLEAR_FULL_GENERATION_OK]"
