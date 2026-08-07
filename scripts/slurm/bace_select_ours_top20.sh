#!/bin/bash
#SBATCH --job-name=bace_ours_select
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

POOL_ROOT=${POOL_ROOT:-outputs/hpc/candidate_pools/bace_ours}
POOL_JSONL=${POOL_JSONL:-$POOL_ROOT/candidate_pool.jsonl}
AUDIT_GATE=${AUDIT_GATE:-$POOL_ROOT/candidate_audit_gate.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/selectors/bace_ours_top20}

for path in "$POOL_JSONL" "$AUDIT_GATE" "$POOL_ROOT/_AUDIT_COMPLETE.json"; do
  if [ ! -s "$path" ]; then
    echo "[BACE_CONFIG_ERROR] missing selector input: $path" >&2
    exit 2
  fi
done
if [ -e "$OUTPUT_DIR" ]; then
  echo "[BACE_CONFIG_ERROR] selector output already exists: $OUTPUT_DIR" >&2
  exit 2
fi

python scripts/select_class_counterfactual_subgraphs.py \
  --config configs/hpc.yaml \
  --pool-jsonl "$POOL_JSONL" \
  --out-dir "$OUTPUT_DIR" \
  --label 1 \
  --top-k 20 \
  --alpha-cf 1.0 \
  --beta-coverage 1.0 \
  --gamma-redundancy 0.7 \
  --eta-size 0.3 \
  --min-cf-drop 0.2 \
  --require-cf-flip \
  --require-final-substructure \
  --dedup-by-final-fragment \
  --sim-metric morgan

export OUTPUT_DIR
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
summary = json.loads((root / "selector_summary.json").read_text())
if int(summary.get("selected_count", 0)) != 20:
    raise SystemExit(
        "[BACE_OURS_SELECTOR_FAIL] "
        f"selected_count={summary.get('selected_count')} expected=20"
    )
required = (
    root / "selected_subgraphs.json",
    root / "selected_subgraphs.csv",
    root / "selector_summary.json",
    root / "selector_report.txt",
)
if any(not path.is_file() or path.stat().st_size == 0 for path in required):
    raise SystemExit("[BACE_OURS_SELECTOR_FAIL] required output missing")
(root / "_RUN_COMPLETE.json").write_text(
    json.dumps({"status": "complete", "stage": "bace_ours_selector"}) + "\n",
    encoding="utf-8",
)
PY

cat "$OUTPUT_DIR/selector_report.txt"
echo "[BACE_OURS_SELECTOR_SUCCESS] selected_count=20"
