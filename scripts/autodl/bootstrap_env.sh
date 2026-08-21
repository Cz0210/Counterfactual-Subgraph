#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ENV_ROOT="$AUTODL_CONTROL_ROOT/environment"
mkdir -p "$ENV_ROOT"

python -m pip freeze > "$ENV_ROOT/before_freeze.txt"
python "$SCRIPT_DIR/detect_runtime.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  --prepare \
  --output "$ENV_ROOT/runtime.json" > "$ENV_ROOT/runtime.stdout.json"

if [[ "${AUTODL_INSTALL_MISSING:-0}" == "1" ]]; then
  echo "AUTODL_INSTALL_MISSING=1 is intentionally fail-closed." >&2
  echo "Pin a torch/CUDA-compatible install command outside this launcher; no system-wide upgrade is performed." >&2
  exit 64
fi

python - <<'PY' > "$ENV_ROOT/cuda_report.txt"
import json
try:
    import torch
    print(json.dumps({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
PY

python - <<'PY' > "$ENV_ROOT/pyg_import_test.txt"
try:
    import torch_geometric
    print("PASS", torch_geometric.__version__)
except Exception as exc:
    print("FAIL", type(exc).__name__, str(exc))
PY

python - <<'PY' > "$ENV_ROOT/rdkit_import_test.txt"
try:
    import rdkit
    from rdkit import Chem
    assert Chem.MolFromSmiles("CCO") is not None
    print("PASS", rdkit.__version__)
except Exception as exc:
    print("FAIL", type(exc).__name__, str(exc))
PY

python -m pip freeze > "$ENV_ROOT/after_freeze.txt"
python "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  init-bace > "$ENV_ROOT/bace_stage_root.txt"

echo "[AUTODL_ENVIRONMENT_BOOTSTRAP_PASS]"
