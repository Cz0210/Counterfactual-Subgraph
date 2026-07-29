#!/bin/bash
# Losslessly merge and audit the versioned Mutagenicity Fresh-PPO pools.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=mut_pool_merge_v2
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

: "${BASE_POOL:?BASE_POOL must be explicitly provided}"
: "${HIGHTEMP_POOL:?HIGHTEMP_POOL must be explicitly provided}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be explicitly provided}"
: "${DATASET_PATH:?DATASET_PATH must be explicitly provided}"
: "${TEACHER_PATH:?TEACHER_PATH must be explicitly provided}"

resolve_from_project_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

BASE_POOL="$(resolve_from_project_root "$BASE_POOL")"
HIGHTEMP_POOL="$(resolve_from_project_root "$HIGHTEMP_POOL")"
OUTPUT_DIR="$(resolve_from_project_root "$OUTPUT_DIR")"
DATASET_PATH="$(resolve_from_project_root "$DATASET_PATH")"
TEACHER_PATH="$(resolve_from_project_root "$TEACHER_PATH")"

if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "[ERROR] OUTPUT_DIR exists and is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] OUTPUT_DIR already exists and is non-empty: $OUTPUT_DIR" >&2
  exit 2
fi

for path in "$BASE_POOL" "$HIGHTEMP_POOL" "$DATASET_PATH" "$TEACHER_PATH"; do
  if [[ ! -s "$path" ]]; then
    echo "[ERROR] Required input file is missing or empty: $path" >&2
    exit 2
  fi
done

for script in \
  scripts/merge_candidate_pools.py \
  scripts/audit_candidate_pool.py \
  scripts/export_candidate_pool_audit_artifacts.py \
  scripts/audit_full_candidate_pool.py; do
  if [[ ! -f "$script" ]]; then
    echo "[ERROR] Required script is missing: $script" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"
MERGED_POOL="$OUTPUT_DIR/candidate_pool.jsonl"
MERGE_SUMMARY="$OUTPUT_DIR/merge_summary.json"
SEMANTIC_AUDIT="$OUTPUT_DIR/merge_semantic_audit.json"
AUDIT_DIR="$OUTPUT_DIR/audit"
FULL_AUDIT_DIR="$OUTPUT_DIR/full_audit"
mkdir -p "$AUDIT_DIR"

EXPECTED_BASE_ROWS=5792
EXPECTED_HIGHTEMP_ROWS=5792
EXPECTED_INPUT_ROWS=11584
EXPECTED_ELIGIBLE_UNIQUE_KEYS=2773
EXPECTED_MERGED_ROWS=2773
EXPECTED_UNIQUE_PARENTS=1448

echo "===== MUTAGENICITY PPO POOL LOSSLESS MERGE V2 ====="
echo "host=$(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "python=$(which python)"
echo "git_commit=$(git rev-parse HEAD)"
echo "BASE_POOL=$BASE_POOL"
echo "HIGHTEMP_POOL=$HIGHTEMP_POOL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "dedup_key=final_fragment,parent_smiles"
echo "keep_best_by=reward_total"
python --version

python scripts/merge_candidate_pools.py \
  --config configs/hpc.yaml \
  --pool-jsonl "$BASE_POOL" \
  --pool-jsonl "$HIGHTEMP_POOL" \
  --out-jsonl "$MERGED_POOL" \
  --out-summary-json "$MERGE_SUMMARY" \
  --dedup-key final_fragment,parent_smiles \
  --keep-best-by reward_total

export BASE_POOL HIGHTEMP_POOL MERGED_POOL MERGE_SUMMARY SEMANTIC_AUDIT
export EXPECTED_BASE_ROWS EXPECTED_HIGHTEMP_ROWS EXPECTED_INPUT_ROWS
export EXPECTED_ELIGIBLE_UNIQUE_KEYS EXPECTED_MERGED_ROWS
export EXPECTED_UNIQUE_PARENTS
python - <<'PY'
import json
import os
from pathlib import Path
from typing import Any


DEDUP_KEY = ("final_fragment", "parent_smiles")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise AssertionError(
                    f"{path}:{line_number} is not a JSON object"
                )
            rows.append(payload)
    return rows


def dedup_key(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(  # type: ignore[return-value]
        str(row.get(field) or "").strip() for field in DEDUP_KEY
    )


base_rows = load_jsonl(Path(os.environ["BASE_POOL"]))
high_temp_rows = load_jsonl(Path(os.environ["HIGHTEMP_POOL"]))
output_rows = load_jsonl(Path(os.environ["MERGED_POOL"]))
input_rows = [*base_rows, *high_temp_rows]

eligible_input_keys = {
    key for row in input_rows if all(key := dedup_key(row))
}
output_keys = [dedup_key(row) for row in output_rows]
output_key_set = set(output_keys)
missing_eligible_keys = eligible_input_keys - output_key_set
unexpected_keys = output_key_set - eligible_input_keys
remaining_duplicate_keys = len(output_keys) - len(output_key_set)
unique_parents = {
    str(row.get("parent_smiles") or "").strip()
    for row in output_rows
    if str(row.get("parent_smiles") or "").strip()
}
merge_summary = json.loads(
    Path(os.environ["MERGE_SUMMARY"]).read_text(encoding="utf-8")
)

observed = {
    "base_rows": len(base_rows),
    "high_temp_rows": len(high_temp_rows),
    "input_rows": len(input_rows),
    "eligible_unique_keys": len(eligible_input_keys),
    "merged_rows": len(output_rows),
    "unique_parents": len(unique_parents),
    "missing_eligible_keys": len(missing_eligible_keys),
    "unexpected_keys": len(unexpected_keys),
    "remaining_duplicate_keys": remaining_duplicate_keys,
}
expected = {
    "base_rows": int(os.environ["EXPECTED_BASE_ROWS"]),
    "high_temp_rows": int(os.environ["EXPECTED_HIGHTEMP_ROWS"]),
    "input_rows": int(os.environ["EXPECTED_INPUT_ROWS"]),
    "eligible_unique_keys": int(
        os.environ["EXPECTED_ELIGIBLE_UNIQUE_KEYS"]
    ),
    "merged_rows": int(os.environ["EXPECTED_MERGED_ROWS"]),
    "unique_parents": int(os.environ["EXPECTED_UNIQUE_PARENTS"]),
    "missing_eligible_keys": 0,
    "unexpected_keys": 0,
    "remaining_duplicate_keys": 0,
}
summary_checks = {
    "input_rows": merge_summary.get("input_rows") == observed["input_rows"],
    "eligible_unique_key_count": (
        merge_summary.get("eligible_unique_key_count")
        == observed["eligible_unique_keys"]
    ),
    "merged_count_after_dedup": (
        merge_summary.get("merged_count_after_dedup")
        == observed["merged_rows"]
    ),
    "missing_eligible_key_count": (
        merge_summary.get("missing_eligible_key_count") == 0
    ),
    "unexpected_key_count": merge_summary.get("unexpected_key_count") == 0,
}
audit_passed = observed == expected and all(summary_checks.values())
payload = {
    "audit_passed": audit_passed,
    "dedup_key": list(DEDUP_KEY),
    "keep_best_by": "reward_total",
    "expected": expected,
    "observed": observed,
    "summary_checks": summary_checks,
    "input_output_key_set_equal": (
        eligible_input_keys == output_key_set
    ),
    "missing_eligible_key_examples": [
        list(key) for key in sorted(missing_eligible_keys)[:10]
    ],
    "unexpected_key_examples": [
        list(key) for key in sorted(unexpected_keys)[:10]
    ],
}
Path(os.environ["SEMANTIC_AUDIT"]).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
if not audit_passed:
    raise AssertionError(
        "Mutagenicity pool merge semantic audit failed: "
        f"expected={expected} observed={observed} "
        f"summary_checks={summary_checks}"
    )
print(
    "[MUTAGENICITY_PPO_POOL_KEY_SET_AUDIT_OK] "
    f"keys={len(output_key_set)} parents={len(unique_parents)}"
)
PY

python scripts/audit_candidate_pool.py \
  --config configs/hpc.yaml \
  --pool_jsonl "$MERGED_POOL" \
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
  --pool-jsonl "$MERGED_POOL" \
  --dataset-path "$DATASET_PATH" \
  --teacher-path "$TEACHER_PATH" \
  --out-dir "$FULL_AUDIT_DIR" \
  --label-col label \
  --smiles-col parent_smiles \
  --target-label 1 \
  --sim-sample-size 10000 \
  --topk-show 30 \
  --coverage-parent-limit 0

for output in \
  "$MERGED_POOL" \
  "$MERGE_SUMMARY" \
  "$SEMANTIC_AUDIT" \
  "$AUDIT_DIR/audit_summary.json" \
  "$AUDIT_DIR/audit_report.txt" \
  "$AUDIT_DIR/diversity_summary.json" \
  "$AUDIT_DIR/parent_coverage_summary.json" \
  "$AUDIT_DIR/fragment_frequency_topk.csv" \
  "$FULL_AUDIT_DIR/audit_summary.json" \
  "$FULL_AUDIT_DIR/audit_report.txt"; do
  [[ -s "$output" ]]
done

echo "[MUTAGENICITY_PPO_POOL_MERGE_V2_OK]"
