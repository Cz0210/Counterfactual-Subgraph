#!/bin/bash
# Select calibration-only Mutagenicity WNode-aware nested prefixes.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mut_wnode_sel
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

: "${MATRIX_RUN_DIR:?MATRIX_RUN_DIR must be explicitly provided}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be explicitly provided}"

resolve_from_project_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

MATRIX_RUN_DIR="$(resolve_from_project_root "$MATRIX_RUN_DIR")"
OUTPUT_DIR="$(resolve_from_project_root "$OUTPUT_DIR")"

TOP_K="${TOP_K:-20}"
TABLE_K="${TABLE_K:-10}"
THRESHOLD_QUANTILES="${THRESHOLD_QUANTILES:-0.05,0.10,0.20,0.30,0.50,0.70,0.90}"
THRESHOLD_WEIGHTS="${THRESHOLD_WEIGHTS:-4,4,3,3,2,1,1}"
THETA_STAR_QUANTILE="${THETA_STAR_QUANTILE:-0.30}"
COST_CAP_QUANTILE="${COST_CAP_QUANTILE:-0.90}"
PREFIX_WEIGHTS="${PREFIX_WEIGHTS:-1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}"
PARENT_LIMIT="${PARENT_LIMIT:-0}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
LOCAL_SWAP_PASSES="${LOCAL_SWAP_PASSES:-2}"
SEED="${SEED:-13}"
if [[ -z "${EXPECTED_PARENT_COUNT:-}" ]]; then
  if [[ "$PARENT_LIMIT" -gt 0 ]]; then
    EXPECTED_PARENT_COUNT="$PARENT_LIMIT"
  else
    EXPECTED_PARENT_COUNT=235
  fi
fi
if [[ -z "${EXPECTED_CANDIDATE_COUNT:-}" ]]; then
  if [[ "$CANDIDATE_LIMIT" -gt 0 ]]; then
    EXPECTED_CANDIDATE_COUNT="$CANDIDATE_LIMIT"
  else
    EXPECTED_CANDIDATE_COUNT=683
  fi
fi

if [[ ! -d "$MATRIX_RUN_DIR" ]]; then
  echo "[ERROR] MATRIX_RUN_DIR does not exist: $MATRIX_RUN_DIR" >&2
  exit 2
fi
for required in \
  pair_matrix.jsonl \
  selected_candidate_universe.jsonl \
  summary.json \
  run_manifest.json; do
  if [[ ! -s "$MATRIX_RUN_DIR/$required" ]]; then
    echo "[ERROR] Missing matrix artifact: $MATRIX_RUN_DIR/$required" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] OUTPUT_DIR exists and is non-empty: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"

echo "===== MUTAGENICITY WNODE PREFIX SELECTOR ====="
echo "hostname=$(hostname)"
echo "date=$(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "python=$(which python)"
echo "git_commit=$(git rev-parse HEAD)"
echo "MATRIX_RUN_DIR=$MATRIX_RUN_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TOP_K=$TOP_K"
echo "TABLE_K=$TABLE_K"
echo "THRESHOLD_QUANTILES=$THRESHOLD_QUANTILES"
echo "THRESHOLD_WEIGHTS=$THRESHOLD_WEIGHTS"
echo "THETA_STAR_QUANTILE=$THETA_STAR_QUANTILE"
echo "COST_CAP_QUANTILE=$COST_CAP_QUANTILE"
echo "PREFIX_WEIGHTS=$PREFIX_WEIGHTS"
echo "PARENT_LIMIT=$PARENT_LIMIT"
echo "CANDIDATE_LIMIT=$CANDIDATE_LIMIT"
echo "LOCAL_SWAP_PASSES=$LOCAL_SWAP_PASSES"
echo "SEED=$SEED"
python --version
nvidia-smi || true

python scripts/select_mutagenicity_wnode_prefix.py \
  --matrix-run-dir "$MATRIX_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  --table-k "$TABLE_K" \
  --threshold-quantiles "$THRESHOLD_QUANTILES" \
  --threshold-weights "$THRESHOLD_WEIGHTS" \
  --theta-star-quantile "$THETA_STAR_QUANTILE" \
  --cost-cap-quantile "$COST_CAP_QUANTILE" \
  --prefix-weights "$PREFIX_WEIGHTS" \
  --parent-limit "$PARENT_LIMIT" \
  --candidate-limit "$CANDIDATE_LIMIT" \
  --local-swap-passes "$LOCAL_SWAP_PASSES" \
  --seed "$SEED" \
  --forbid-test

python scripts/audit_mutagenicity_wnode_prefix.py \
  --run-dir "$OUTPUT_DIR" \
  --matrix-run-dir "$MATRIX_RUN_DIR" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --expected-candidate-count "$EXPECTED_CANDIDATE_COUNT" \
  --expected-top-k "$TOP_K" \
  --expected-table-k "$TABLE_K" \
  --require-all-variants \
  --require-nested-prefix \
  --require-monotonic-coverage \
  --require-nonincreasing-capped-cost \
  --forbid-test

[[ -s "$OUTPUT_DIR/_RUN_COMPLETE.json" ]]
[[ -s "$OUTPUT_DIR/selector_audit.json" ]]
echo "[MUTAGENICITY_WNODE_PREFIX_SELECTOR_OK]"
