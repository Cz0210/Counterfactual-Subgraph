#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=mut_wnode_test
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
if [[ ! -f "$PROJECT_ROOT/scripts/evaluate_mutagenicity_wnode_frozen_test.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi

: "${FROZEN_SELECTOR_ROOT:?FROZEN_SELECTOR_ROOT must be provided}"
: "${TEST_CSV:?TEST_CSV must be provided}"
: "${TEACHER_PATH:?TEACHER_PATH must be provided}"
: "${MOLCLR_ROOT:?MOLCLR_ROOT must be provided}"
: "${MOLCLR_CHECKPOINT:?MOLCLR_CHECKPOINT must be provided}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be provided}"
: "${WNODE_CACHE_DB:?WNODE_CACHE_DB must be provided}"

resolve_from_root() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$PROJECT_ROOT/$value"
  fi
}

FROZEN_SELECTOR_ROOT="$(resolve_from_root "$FROZEN_SELECTOR_ROOT")"
TEST_CSV="$(resolve_from_root "$TEST_CSV")"
TEACHER_PATH="$(resolve_from_root "$TEACHER_PATH")"
MOLCLR_ROOT="$(resolve_from_root "$MOLCLR_ROOT")"
MOLCLR_CHECKPOINT="$(resolve_from_root "$MOLCLR_CHECKPOINT")"
OUTPUT_DIR="$(resolve_from_root "$OUTPUT_DIR")"
WNODE_CACHE_DB="$(resolve_from_root "$WNODE_CACHE_DB")"

EXPECTED_PARENT_COUNT="${EXPECTED_PARENT_COUNT:-217}"
EXPECTED_CANDIDATE_COUNT="${EXPECTED_CANDIDATE_COUNT:-20}"
EXPECTED_PAIR_COUNT="${EXPECTED_PAIR_COUNT:-4340}"
EXPECTED_TOP_K="${EXPECTED_TOP_K:-20}"
EXPECTED_TABLE_K="${EXPECTED_TABLE_K:-10}"
WNODE_SIZE_PENALTY_BETA="${WNODE_SIZE_PENALTY_BETA:-0.0}"
FLUSH_EVERY="${FLUSH_EVERY:-100}"
RESUME="${RESUME:-true}"

if [[ -s "$OUTPUT_DIR/_RUN_COMPLETE.json" ]]; then
  echo "[ERROR] Frozen test evaluation is already complete: $OUTPUT_DIR" >&2
  exit 3
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  if [[ "$RESUME" != "true" ]]; then
    echo "[ERROR] OUTPUT_DIR is non-empty and RESUME is false: $OUTPUT_DIR" >&2
    exit 3
  fi
  test -s "$OUTPUT_DIR/run_manifest.json"
  test -s "$OUTPUT_DIR/resume_checkpoint.json"
fi

for path in \
  "$FROZEN_SELECTOR_ROOT/frozen_selector_manifest.json" \
  "$TEST_CSV" \
  "$TEACHER_PATH" \
  "$MOLCLR_ROOT" \
  "$MOLCLR_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Required input does not exist: $path" >&2
    exit 4
  fi
done

mkdir -p "$PROJECT_ROOT/logs" "$(dirname "$WNODE_CACHE_DB")" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source ~/.bashrc
conda activate smiles_pip118

echo "===== MUTAGENICITY FROZEN WNODE TEST EVALUATION ====="
echo "hostname=$(hostname)"
echo "date=$(date -Iseconds)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "FROZEN_SELECTOR_ROOT=$FROZEN_SELECTOR_ROOT"
echo "TEST_CSV=$TEST_CSV"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "MOLCLR_ROOT=$MOLCLR_ROOT"
echo "MOLCLR_CHECKPOINT=$MOLCLR_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "WNODE_CACHE_DB=$WNODE_CACHE_DB"
echo "EXPECTED_PARENT_COUNT=$EXPECTED_PARENT_COUNT"
echo "EXPECTED_CANDIDATE_COUNT=$EXPECTED_CANDIDATE_COUNT"
echo "EXPECTED_PAIR_COUNT=$EXPECTED_PAIR_COUNT"
echo "EXPECTED_TOP_K=$EXPECTED_TOP_K"
echo "EXPECTED_TABLE_K=$EXPECTED_TABLE_K"
echo "WNODE_SIZE_PENALTY_BETA=$WNODE_SIZE_PENALTY_BETA"
echo "FLUSH_EVERY=$FLUSH_EVERY"
echo "RESUME=$RESUME"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
nvidia-smi

RESUME_FLAG="--resume"
if [[ "$RESUME" != "true" ]]; then
  RESUME_FLAG="--no-resume"
fi

python scripts/evaluate_mutagenicity_wnode_frozen_test.py \
  --frozen-selector-root "$FROZEN_SELECTOR_ROOT" \
  --test-csv "$TEST_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --id-col molecule_id \
  --smiles-col smiles \
  --label-col label \
  --cohort-name test \
  --wnode-size-penalty-beta "$WNODE_SIZE_PENALTY_BETA" \
  --flush-every "$FLUSH_EVERY" \
  "$RESUME_FLAG" \
  --local-files-only

python scripts/audit_mutagenicity_wnode_frozen_test.py \
  --run-dir "$OUTPUT_DIR" \
  --frozen-selector-root "$FROZEN_SELECTOR_ROOT" \
  --test-csv "$TEST_CSV" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --expected-candidate-count "$EXPECTED_CANDIDATE_COUNT" \
  --expected-pair-count "$EXPECTED_PAIR_COUNT" \
  --expected-top-k "$EXPECTED_TOP_K" \
  --expected-table-k "$EXPECTED_TABLE_K" \
  --require-complete-cartesian \
  --require-frozen-thresholds \
  --require-frozen-candidate-order \
  --require-monotonic-coverage \
  --require-nonincreasing-capped-cost

test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$OUTPUT_DIR/frozen_test_audit.json"
echo "[MUTAGENICITY_WNODE_FROZEN_TEST_EVAL_OK]"
