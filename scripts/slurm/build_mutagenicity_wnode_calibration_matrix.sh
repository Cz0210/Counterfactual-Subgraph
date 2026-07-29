#!/bin/bash
# Build the Mutagenicity calibration parent-candidate WNode action matrix.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=mut_wnode_mat
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] Could not determine PROJECT_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

: "${CANDIDATE_POOL:?CANDIDATE_POOL must be explicitly provided}"
: "${CALIBRATION_CSV:?CALIBRATION_CSV must be explicitly provided}"
: "${TEACHER_PATH:?TEACHER_PATH must be explicitly provided}"
: "${MOLCLR_ROOT:?MOLCLR_ROOT must be explicitly provided}"
: "${MOLCLR_CHECKPOINT:?MOLCLR_CHECKPOINT must be explicitly provided}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be explicitly provided}"
: "${WNODE_CACHE_DB:?WNODE_CACHE_DB must be explicitly provided}"

resolve_from_project_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

CANDIDATE_POOL="$(resolve_from_project_root "$CANDIDATE_POOL")"
CALIBRATION_CSV="$(resolve_from_project_root "$CALIBRATION_CSV")"
TEACHER_PATH="$(resolve_from_project_root "$TEACHER_PATH")"
MOLCLR_ROOT="$(resolve_from_project_root "$MOLCLR_ROOT")"
MOLCLR_CHECKPOINT="$(resolve_from_project_root "$MOLCLR_CHECKPOINT")"
OUTPUT_DIR="$(resolve_from_project_root "$OUTPUT_DIR")"
WNODE_CACHE_DB="$(resolve_from_project_root "$WNODE_CACHE_DB")"

PARENT_LIMIT="${PARENT_LIMIT:-0}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
EXPECTED_PARENT_COUNT="${EXPECTED_PARENT_COUNT:-235}"
EXPECTED_CANDIDATE_COUNT="${EXPECTED_CANDIDATE_COUNT:-0}"
CANDIDATE_ORDER="${CANDIDATE_ORDER:-source_support_desc}"
FLUSH_EVERY="${FLUSH_EVERY:-100}"
RESUME="${RESUME:-true}"
WNODE_SIZE_PENALTY_BETA="${WNODE_SIZE_PENALTY_BETA:-0.0}"
EXPECTED_SOURCE_ELIGIBLE_ROWS="${EXPECTED_SOURCE_ELIGIBLE_ROWS:-1961}"
EXPECTED_SOURCE_ELIGIBLE_RAW_UNIQUE="${EXPECTED_SOURCE_ELIGIBLE_RAW_UNIQUE:-683}"

for path in \
  "$CANDIDATE_POOL" \
  "$CALIBRATION_CSV" \
  "$TEACHER_PATH" \
  "$MOLCLR_ROOT" \
  "$MOLCLR_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Required input does not exist: $path" >&2
    exit 2
  fi
done

RESUME_NORMALIZED="$(printf '%s' "$RESUME" | tr '[:upper:]' '[:lower:]')"
case "$RESUME_NORMALIZED" in
  true|1|yes|on) RESUME_FLAG="--resume" ;;
  false|0|no|off) RESUME_FLAG="--no-resume" ;;
  *)
    echo "[ERROR] RESUME must be a boolean value" >&2
    exit 2
    ;;
esac

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  if [[ "$RESUME_FLAG" == "--no-resume" ]]; then
    echo "[ERROR] OUTPUT_DIR is non-empty and RESUME=false: $OUTPUT_DIR" >&2
    exit 2
  fi
  for required in run_manifest.json resume_checkpoint.json; do
    if [[ ! -s "$OUTPUT_DIR/$required" ]]; then
      echo "[ERROR] Resume output is missing a valid $required" >&2
      exit 2
    fi
  done
fi
mkdir -p "$OUTPUT_DIR" "$(dirname "$WNODE_CACHE_DB")" "$PROJECT_ROOT/logs"

EXPECTED_PAIR_COUNT=0
if [[ "$EXPECTED_CANDIDATE_COUNT" -gt 0 ]]; then
  EXPECTED_PAIR_COUNT=$((EXPECTED_PARENT_COUNT * EXPECTED_CANDIDATE_COUNT))
fi

echo "===== MUTAGENICITY WNODE CALIBRATION MATRIX ====="
echo "host=$(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "python=$(which python)"
echo "git_commit=$(git rev-parse HEAD)"
echo "CANDIDATE_POOL=$CANDIDATE_POOL"
echo "CALIBRATION_CSV=$CALIBRATION_CSV"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "MOLCLR_ROOT=$MOLCLR_ROOT"
echo "MOLCLR_CHECKPOINT=$MOLCLR_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "WNODE_CACHE_DB=$WNODE_CACHE_DB"
echo "WNODE_SIZE_PENALTY_BETA=$WNODE_SIZE_PENALTY_BETA"
echo "PARENT_LIMIT=$PARENT_LIMIT"
echo "CANDIDATE_LIMIT=$CANDIDATE_LIMIT"
echo "EXPECTED_PARENT_COUNT=$EXPECTED_PARENT_COUNT"
echo "EXPECTED_CANDIDATE_COUNT=$EXPECTED_CANDIDATE_COUNT"
echo "CANDIDATE_ORDER=$CANDIDATE_ORDER"
echo "FLUSH_EVERY=$FLUSH_EVERY"
echo "RESUME=$RESUME"
python --version
nvidia-smi || true

python scripts/build_mutagenicity_wnode_calibration_matrix.py \
  --candidate-pool "$CANDIDATE_POOL" \
  --calibration-csv "$CALIBRATION_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --wnode-size-penalty-beta "$WNODE_SIZE_PENALTY_BETA" \
  --id-col molecule_id \
  --smiles-col smiles \
  --label-col label \
  --cohort-name calibration \
  --parent-limit "$PARENT_LIMIT" \
  --candidate-limit "$CANDIDATE_LIMIT" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --candidate-order "$CANDIDATE_ORDER" \
  --flush-every "$FLUSH_EVERY" \
  "$RESUME_FLAG" \
  --local-files-only

python scripts/audit_mutagenicity_wnode_calibration_matrix.py \
  --run-dir "$OUTPUT_DIR" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --expected-candidate-count "$EXPECTED_CANDIDATE_COUNT" \
  --expected-pair-count "$EXPECTED_PAIR_COUNT" \
  --expected-source-eligible-rows "$EXPECTED_SOURCE_ELIGIBLE_ROWS" \
  --expected-source-eligible-raw-unique "$EXPECTED_SOURCE_ELIGIBLE_RAW_UNIQUE" \
  --require-complete-cartesian \
  --require-strict-flip-pair \
  --forbid-test

[[ -s "$OUTPUT_DIR/_RUN_COMPLETE.json" ]]
[[ -s "$OUTPUT_DIR/matrix_audit.json" ]]
echo "[MUTAGENICITY_WNODE_CALIBRATION_MATRIX_OK]"
