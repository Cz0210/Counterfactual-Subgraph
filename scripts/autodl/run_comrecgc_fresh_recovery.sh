#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
DATASET="${DATASET:?DATASET must be mutagenicity or aids}"
SOURCE_GENERATION_DIR="${SOURCE_GENERATION_DIR:?SOURCE_GENERATION_DIR is required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT fresh persistent path is required}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-$OUTPUT_ROOT/fresh_recovery_audit.json}"
EXPECTED_STEPS="${EXPECTED_STEPS:-50000}"
EXPECTED_PROJECT_COMMIT="${EXPECTED_PROJECT_COMMIT:-}"
SOURCE_CSV="${SOURCE_CSV:-}"

[[ "$DATASET" == "mutagenicity" || "$DATASET" == "aids" ]] || exit 64
for path in "$SOURCE_GENERATION_DIR" "$DATASET_DIR" "$OUTPUT_ROOT" "$AUDIT_OUTPUT"; do
  [[ "$path" == /* ]] || { echo "absolute path required: $path" >&2; exit 64; }
done
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 64; }
[[ -d "$SOURCE_GENERATION_DIR" ]] || { echo "source generation missing" >&2; exit 66; }
[[ -d "$DATASET_DIR" ]] || { echo "dataset directory missing" >&2; exit 66; }
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "fresh OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2
  exit 73
fi
if [[ "$DATASET" == "aids" && -z "$SOURCE_CSV" ]]; then
  echo "SOURCE_CSV is required for AIDS" >&2
  exit 64
fi

mkdir -p "$(dirname "$OUTPUT_ROOT")"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

args=(
  --source-generation-dir "$SOURCE_GENERATION_DIR"
  --output-dir "$OUTPUT_ROOT"
  --audit-output "$AUDIT_OUTPUT"
  --dataset "$DATASET"
  --dataset-dir "$DATASET_DIR"
  --expected-steps "$EXPECTED_STEPS"
)
[[ -z "$SOURCE_CSV" ]] || args+=(--source-csv "$SOURCE_CSV")
[[ -z "$EXPECTED_PROJECT_COMMIT" ]] || args+=(--expected-project-commit "$EXPECTED_PROJECT_COMMIT")

echo "[COMRECGC_FRESH_RECOVERY_START] dataset=$DATASET output=$OUTPUT_ROOT"
exec env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "$PYTHON" "$PROJECT_ROOT/scripts/baselines/comrecgc/recover_completed_generation_freeze.py" \
  "${args[@]}"
