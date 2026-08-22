#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"

: "${TASTEMOLNET_PREPARED_ROOT:?TASTEMOLNET_PREPARED_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT fresh persistent path is required}"
[[ "$TASTEMOLNET_PREPARED_ROOT" == /* && "$OUTPUT_ROOT" == /* ]] || {
  echo "Taste license audit paths must be absolute" >&2
  exit 64
}
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "fresh OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2
  exit 73
}

args=(
  --config configs/hpc.yaml
  --prepared-root "$TASTEMOLNET_PREPARED_ROOT"
  --output-dir "$OUTPUT_ROOT"
  --audit-completion-mode
)
if [[ -n "${TASTEMOLNET_LICENSE_APPROVAL_FILE:-}" ]]; then
  args+=(--approval-file "$TASTEMOLNET_LICENSE_APPROVAL_FILE")
fi
if [[ -n "${TASTEMOLNET_UPSTREAM_CHECKOUT:-}" ]]; then
  args+=(--upstream-checkout "$TASTEMOLNET_UPSTREAM_CHECKOUT")
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
exec "$PYTHON" "$PROJECT_ROOT/scripts/audit_tastemolnet_license.py" "${args[@]}"
