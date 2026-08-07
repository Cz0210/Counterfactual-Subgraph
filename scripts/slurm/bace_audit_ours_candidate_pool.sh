#!/bin/bash
#SBATCH --job-name=bace_ours_pool_audit
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
DATASET_PATH=${DATASET_PATH:-outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/bace/bace_teacher.pkl}
AUDIT_DIR=$POOL_ROOT/audit
FULL_AUDIT_DIR=$POOL_ROOT/full_audit
GATE_PATH=$POOL_ROOT/candidate_audit_gate.json
COMPLETE_MARKER=$POOL_ROOT/_AUDIT_COMPLETE.json

for path in "$POOL_JSONL" "$DATASET_PATH" "$TEACHER_PATH" "$POOL_ROOT/_RUN_COMPLETE.json"; do
  if [ ! -s "$path" ]; then
    echo "[BACE_CONFIG_ERROR] missing candidate-audit input: $path" >&2
    exit 2
  fi
done
if [ -e "$COMPLETE_MARKER" ]; then
  echo "[BACE_OURS_CANDIDATE_AUDIT_ADOPT_EXISTING]"
  exit 0
fi
if [ -d "$AUDIT_DIR" ] || [ -d "$FULL_AUDIT_DIR" ] || [ -e "$GATE_PATH" ]; then
  echo "[BACE_CONFIG_ERROR] candidate audit output already exists" >&2
  exit 2
fi
mkdir -p "$AUDIT_DIR" "$FULL_AUDIT_DIR"

python scripts/audit_candidate_pool.py \
  --config configs/hpc.yaml \
  --pool_jsonl "$POOL_JSONL" \
  --out_json "$AUDIT_DIR/audit_summary.json" \
  --out_txt "$AUDIT_DIR/audit_report.txt" \
  --group_by_label \
  --sim_sample_size 10000 \
  --topk_show 30

python scripts/export_candidate_pool_audit_artifacts.py \
  --config configs/hpc.yaml \
  --audit-json "$AUDIT_DIR/audit_summary.json" \
  --out-dir "$AUDIT_DIR" \
  --topk 30

python scripts/audit_full_candidate_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pool-jsonl "$POOL_JSONL" \
  --dataset-path "$DATASET_PATH" \
  --teacher-path "$TEACHER_PATH" \
  --out-dir "$FULL_AUDIT_DIR" \
  --label-col label \
  --smiles-col smiles \
  --target-label 1 \
  --sim-sample-size 10000 \
  --topk-show 30 \
  --coverage-parent-limit 0

export POOL_JSONL AUDIT_DIR FULL_AUDIT_DIR GATE_PATH
python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads((Path(os.environ["AUDIT_DIR"]) / "audit_summary.json").read_text())
overall = summary["overall"]
judgment = summary["judgment"]
rate = float(overall["final_substructure_rate"])
if rate <= 0.9:
    raise SystemExit(
        "[BACE_OURS_CANDIDATE_AUDIT_FAIL] "
        f"final_substructure_rate={rate} expected_gt=0.9"
    )
if judgment.get("recommend_start_selector") is not True:
    raise SystemExit("[BACE_OURS_CANDIDATE_AUDIT_FAIL] suitable_for_selector=no")
gate = {
    "passed": True,
    "stage": "bace_ours_candidate_pool_audit",
    "candidate_count": int(overall["num_total"]),
    "num_parents": int(overall["num_unique_parent"]),
    "parse_ok_rate": float(overall["parse_ok_rate"]),
    "valid_rate": float(overall["valid_rate"]),
    "final_substructure_rate": rate,
    "cf_flip_rate": float(overall["cf_flip_rate"]),
    "cf_drop_mean": overall.get("cf_drop_mean"),
    "atom_ratio_mean": overall.get("atom_ratio_mean"),
    "unique_fragment_rate": float(overall["unique_final_fragment_rate"]),
    "suitable_for_selector": True,
    "candidate_pool": str(Path(os.environ["POOL_JSONL"]).resolve()),
}
Path(os.environ["GATE_PATH"]).write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf '{"status":"complete","stage":"bace_ours_candidate_pool_audit"}\n' > "$COMPLETE_MARKER"
cat "$GATE_PATH"
echo "[BACE_OURS_CANDIDATE_AUDIT_SUCCESS]"
