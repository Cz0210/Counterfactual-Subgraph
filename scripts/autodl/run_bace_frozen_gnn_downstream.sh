#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

DRIVER="$PROJECT_ROOT/scripts/autodl/bace_frozen_gnn_downstream.py"
autodl_require_file "$DRIVER"

# This wrapper stays foreground.  The persistent controller owns process
# lifetime, GPU UUID locks, retries, logs, and registry entries.
exec "$AUTODL_PYTHON" "$DRIVER" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  "$@"
