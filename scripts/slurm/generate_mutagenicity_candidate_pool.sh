#!/bin/bash
# Generate and audit the Fresh Mutagenicity PPO candidate pool.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=mut_ppo_pool
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

resolve_from_project_root() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v2/mutagenicity_ppo_prompts_train_label1_v2.csv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}"
SFT_LORA_PATH="${SFT_LORA_PATH:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_fresh_strict_v2_best}"
PPO_CHECKPOINT_PATH="${PPO_CHECKPOINT_PATH:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/ppo_fresh_strict_v2_best}"
TEACHER_PATH="${TEACHER_PATH:-$PROJECT_ROOT/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "[ERROR] OUTPUT_DIR must be explicitly set by the caller" >&2
  exit 2
fi

DATASET_PATH="$(resolve_from_project_root "$DATASET_PATH")"
BASE_MODEL_PATH="$(resolve_from_project_root "$BASE_MODEL_PATH")"
SFT_LORA_PATH="$(resolve_from_project_root "$SFT_LORA_PATH")"
PPO_CHECKPOINT_PATH="$(resolve_from_project_root "$PPO_CHECKPOINT_PATH")"
TEACHER_PATH="$(resolve_from_project_root "$TEACHER_PATH")"
OUTPUT_DIR="$(resolve_from_project_root "$OUTPUT_DIR")"

NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.5}"
GEN_TOP_P="${GEN_TOP_P:-0.8}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-13}"
LIMIT="${LIMIT:-0}"
RUN_FULL_AUDIT="${RUN_FULL_AUDIT:-auto}"
SIM_SAMPLE_SIZE="${SIM_SAMPLE_SIZE:-10000}"
TOPK_SHOW="${TOPK_SHOW:-30}"

if [[ ! "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] LIMIT must be a non-negative integer: $LIMIT" >&2
  exit 2
fi
if [[ ! "$NUM_RETURN_SEQUENCES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] NUM_RETURN_SEQUENCES must be a positive integer" >&2
  exit 2
fi

EXPECTED_PARENTS="$LIMIT"
if [[ "$LIMIT" -eq 0 ]]; then
  EXPECTED_PARENTS=1448
fi
EXPECTED_ROWS=$((EXPECTED_PARENTS * NUM_RETURN_SEQUENCES))

RUN_FULL_AUDIT_NORMALIZED="$(printf '%s' "$RUN_FULL_AUDIT" | tr '[:upper:]' '[:lower:]')"
case "$RUN_FULL_AUDIT_NORMALIZED" in
  auto|true|false|1|0|yes|no|on|off) ;;
  *)
    echo "[ERROR] RUN_FULL_AUDIT must be auto or a boolean value" >&2
    exit 2
    ;;
esac

DO_FULL_AUDIT=false
if [[ "$LIMIT" -eq 0 ]]; then
  DO_FULL_AUDIT=true
elif [[ "$RUN_FULL_AUDIT_NORMALIZED" =~ ^(true|1|yes|on)$ ]]; then
  DO_FULL_AUDIT=true
fi

if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
  echo "[ERROR] OUTPUT_DIR exists and is not a directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] OUTPUT_DIR already exists and is non-empty: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/logs"

POOL_JSONL="$OUTPUT_DIR/candidate_pool.jsonl"
GENERATION_SUMMARY="$OUTPUT_DIR/generation_summary.json"
AUDIT_DIR="$OUTPUT_DIR/audit"
FULL_AUDIT_DIR="$OUTPUT_DIR/full_audit"
RUN_MANIFEST="$OUTPUT_DIR/run_manifest.json"
STRUCTURAL_AUDIT="$OUTPUT_DIR/structural_audit.json"
mkdir -p "$AUDIT_DIR"

for path in \
  "$DATASET_PATH" \
  "$BASE_MODEL_PATH" \
  "$SFT_LORA_PATH" \
  "$PPO_CHECKPOINT_PATH" \
  "$TEACHER_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -s "$DATASET_PATH" || ! -s "$TEACHER_PATH" ]]; then
  echo "[ERROR] Dataset and teacher files must be non-empty" >&2
  exit 2
fi

for script in \
  scripts/generate_full_candidate_pool.py \
  scripts/audit_candidate_pool.py \
  scripts/export_candidate_pool_audit_artifacts.py \
  scripts/audit_full_candidate_pool.py; do
  if [[ ! -f "$script" ]]; then
    echo "[ERROR] Required script is missing: $script" >&2
    exit 2
  fi
done

GIT_COMMIT="$(git rev-parse HEAD)"
export PROJECT_ROOT DATASET_PATH BASE_MODEL_PATH SFT_LORA_PATH
export PPO_CHECKPOINT_PATH TEACHER_PATH OUTPUT_DIR
export NUM_RETURN_SEQUENCES GEN_TEMPERATURE GEN_TOP_P GEN_DO_SAMPLE
export MAX_NEW_TOKENS BATCH_SIZE SEED LIMIT EXPECTED_PARENTS EXPECTED_ROWS
export RUN_FULL_AUDIT DO_FULL_AUDIT SIM_SAMPLE_SIZE TOPK_SHOW
export GIT_COMMIT RUN_MANIFEST

echo "===== MUTAGENICITY FRESH PPO CANDIDATE POOL ====="
echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "python=$(which python)"
echo "git_commit=$GIT_COMMIT"
echo "DATASET_PATH=$DATASET_PATH"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "SFT_LORA_PATH=$SFT_LORA_PATH"
echo "PPO_CHECKPOINT_PATH=$PPO_CHECKPOINT_PATH"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_RETURN_SEQUENCES=$NUM_RETURN_SEQUENCES"
echo "GEN_TEMPERATURE=$GEN_TEMPERATURE"
echo "GEN_TOP_P=$GEN_TOP_P"
echo "GEN_DO_SAMPLE=$GEN_DO_SAMPLE"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "SEED=$SEED"
echo "LIMIT=$LIMIT"
echo "EXPECTED_PARENTS=$EXPECTED_PARENTS"
echo "EXPECTED_ROWS=$EXPECTED_ROWS"
echo "RUN_FULL_AUDIT=$RUN_FULL_AUDIT"
echo "DO_FULL_AUDIT=$DO_FULL_AUDIT"
python --version
nvidia-smi || true

python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {
    "status": "initialized",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": os.environ["GIT_COMMIT"],
    "inputs": {
        "dataset_path": str(Path(os.environ["DATASET_PATH"]).resolve()),
        "base_model_path": str(Path(os.environ["BASE_MODEL_PATH"]).resolve()),
        "sft_lora_path": str(Path(os.environ["SFT_LORA_PATH"]).resolve()),
        "ppo_checkpoint_path": str(Path(os.environ["PPO_CHECKPOINT_PATH"]).resolve()),
        "teacher_path": str(Path(os.environ["TEACHER_PATH"]).resolve()),
    },
    "input_file_sha256": {
        "dataset": sha256_file(Path(os.environ["DATASET_PATH"])),
        "teacher": sha256_file(Path(os.environ["TEACHER_PATH"])),
    },
    "sampling": {
        "num_return_sequences": int(os.environ["NUM_RETURN_SEQUENCES"]),
        "generation_temperature": float(os.environ["GEN_TEMPERATURE"]),
        "generation_top_p": float(os.environ["GEN_TOP_P"]),
        "generation_do_sample": os.environ["GEN_DO_SAMPLE"],
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "seed": int(os.environ["SEED"]),
    },
    "limit": int(os.environ["LIMIT"]),
    "expected_parent_count": int(os.environ["EXPECTED_PARENTS"]),
    "expected_row_count": int(os.environ["EXPECTED_ROWS"]),
    "run_full_audit": os.environ["DO_FULL_AUDIT"] == "true",
    "output_dir": str(Path(os.environ["OUTPUT_DIR"]).resolve()),
}
Path(os.environ["RUN_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

python scripts/generate_full_candidate_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-path "$DATASET_PATH" \
  --base-model-path "$BASE_MODEL_PATH" \
  --sft-lora-path "$SFT_LORA_PATH" \
  --ppo-checkpoint-path "$PPO_CHECKPOINT_PATH" \
  --teacher-path "$TEACHER_PATH" \
  --out-jsonl "$POOL_JSONL" \
  --out-summary-json "$GENERATION_SUMMARY" \
  --label-col label \
  --smiles-col parent_smiles \
  --target-label 1 \
  --num-return-sequences "$NUM_RETURN_SEQUENCES" \
  --generation-temperature "$GEN_TEMPERATURE" \
  --generation-top-p "$GEN_TOP_P" \
  --generation-do-sample "$GEN_DO_SAMPLE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --limit "$LIMIT" \
  --enable-parent-projection \
  --enable-projected-cf-reward \
  --enable-substructure-distance-reward \
  --substructure-distance-reward-weight 0.1923076923076923 \
  --projection-penalty 0.25 \
  --enable-minimal-syntax-repair \
  --enable-component-salvage

python scripts/audit_candidate_pool.py \
  --config configs/hpc.yaml \
  --pool_jsonl "$POOL_JSONL" \
  --out_json "$AUDIT_DIR/audit_summary.json" \
  --out_txt "$AUDIT_DIR/audit_report.txt" \
  --group_by_label \
  --sim_sample_size "$SIM_SAMPLE_SIZE" \
  --topk_show "$TOPK_SHOW"

python scripts/export_candidate_pool_audit_artifacts.py \
  --config configs/hpc.yaml \
  --audit-json "$AUDIT_DIR/audit_summary.json" \
  --out-dir "$AUDIT_DIR" \
  --topk "$TOPK_SHOW"

if [[ "$DO_FULL_AUDIT" == "true" ]]; then
  python scripts/audit_full_candidate_pool.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --pool-jsonl "$POOL_JSONL" \
    --dataset-path "$DATASET_PATH" \
    --teacher-path "$TEACHER_PATH" \
    --out-dir "$FULL_AUDIT_DIR" \
    --label-col label \
    --smiles-col parent_smiles \
    --target-label 1 \
    --sim-sample-size "$SIM_SAMPLE_SIZE" \
    --topk-show "$TOPK_SHOW" \
    --coverage-parent-limit 0
fi

export POOL_JSONL GENERATION_SUMMARY AUDIT_DIR FULL_AUDIT_DIR STRUCTURAL_AUDIT
python - <<'PY'
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AssertionError(f"Required output is missing or empty: {path}")


def parent_key(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("molecule_id", "parent_id", "parent_index"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return key, str(value)
    value = row.get("parent_smiles")
    if value is None or not str(value).strip():
        raise AssertionError("Candidate row has no stable parent identifier")
    return "parent_smiles", str(value)


pool_path = Path(os.environ["POOL_JSONL"])
summary_path = Path(os.environ["GENERATION_SUMMARY"])
audit_dir = Path(os.environ["AUDIT_DIR"])
full_audit_dir = Path(os.environ["FULL_AUDIT_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
manifest_path = Path(os.environ["RUN_MANIFEST"])
structural_audit_path = Path(os.environ["STRUCTURAL_AUDIT"])

for required in (
    pool_path,
    summary_path,
    audit_dir / "audit_summary.json",
    audit_dir / "audit_report.txt",
    audit_dir / "diversity_summary.json",
    audit_dir / "parent_coverage_summary.json",
    audit_dir / "fragment_frequency_topk.csv",
    manifest_path,
):
    require_nonempty(required)

if os.environ["DO_FULL_AUDIT"] == "true":
    require_nonempty(full_audit_dir / "audit_summary.json")
    require_nonempty(full_audit_dir / "audit_report.txt")

required_fields = {
    "parent_smiles",
    "final_fragment",
    "cf_flip",
    "parse_ok",
    "final_substructure",
}
rows: list[dict[str, Any]] = []
with pool_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            raise AssertionError(f"Blank JSONL row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Invalid JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AssertionError(f"JSONL row {line_number} is not an object")
        missing = sorted(required_fields - set(row))
        if missing:
            raise AssertionError(
                f"JSONL row {line_number} is missing required fields: {missing}"
            )
        if not str(row.get("parent_smiles") or "").strip():
            raise AssertionError(f"JSONL row {line_number} has empty parent_smiles")
        if "label" not in row or int(row["label"]) != 1:
            raise AssertionError(f"JSONL row {line_number} has non-source label")
        if "source_label" in row and int(row["source_label"]) != 1:
            raise AssertionError(
                f"JSONL row {line_number} has non-source source_label"
            )
        rows.append(row)

parent_identifiers = {parent_key(row) for row in rows}
expected_parents = int(os.environ["EXPECTED_PARENTS"])
expected_rows = int(os.environ["EXPECTED_ROWS"])
if len(parent_identifiers) != expected_parents:
    raise AssertionError(
        f"Unique-parent mismatch: expected {expected_parents}, "
        f"found {len(parent_identifiers)}"
    )
if len(rows) != expected_rows:
    raise AssertionError(
        f"Candidate-row mismatch: expected {expected_rows}, found {len(rows)}"
    )

summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_ppo_path = str(Path(os.environ["PPO_CHECKPOINT_PATH"]).resolve())
summary_ppo_path_raw = (summary.get("model_load") or {}).get(
    "ppo_checkpoint_path"
)
if not summary_ppo_path_raw:
    raise AssertionError(
        "generation_summary.json does not record model_load.ppo_checkpoint_path"
    )
summary_ppo_path = str(Path(summary_ppo_path_raw).resolve())
if summary_ppo_path != expected_ppo_path:
    raise AssertionError(
        "PPO adapter-path mismatch: "
        f"expected {expected_ppo_path}, found {summary_ppo_path}"
    )

dataset_path = str(Path(os.environ["DATASET_PATH"]).resolve())
if re.search(r"(^|[/_.-])(calibration|test)([/_.-]|$)", dataset_path.lower()):
    raise AssertionError(
        f"Dataset path points to a forbidden evaluation split: {dataset_path}"
    )

forbidden_input_patterns = (
    "mutagenicity_ppo_prompts_calibration",
    "mutagenicity_ppo_prompts_test",
    "/calibration.csv",
    "/test.csv",
)
text_suffixes = {".json", ".jsonl", ".txt", ".md", ".csv"}
for path in output_dir.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in text_suffixes:
        continue
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    matches = [token for token in forbidden_input_patterns if token in text]
    if matches:
        raise AssertionError(
            f"Forbidden calibration/test input path in {path}: {matches}"
        )

structural_audit = {
    "audit_passed": True,
    "jsonl_parse_ok": True,
    "num_rows": len(rows),
    "num_unique_parents": len(parent_identifiers),
    "expected_rows": expected_rows,
    "expected_unique_parents": expected_parents,
    "required_fields": sorted(required_fields),
    "source_label": 1,
    "expected_ppo_checkpoint_path": expected_ppo_path,
    "summary_ppo_checkpoint_path": summary_ppo_path,
    "ppo_checkpoint_path_match": True,
    "calibration_test_input_path_found": False,
    "full_audit_executed": os.environ["DO_FULL_AUDIT"] == "true",
}
structural_audit_path.write_text(
    json.dumps(structural_audit, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.update(
    {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "actual_parent_count": len(parent_identifiers),
        "actual_row_count": len(rows),
        "structural_audit": str(structural_audit_path.resolve()),
        "structural_audit_passed": True,
    }
)
temporary_manifest = manifest_path.with_suffix(".json.tmp")
temporary_manifest.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary_manifest, manifest_path)

print(
    "[MUTAGENICITY_CANDIDATE_POOL_STRUCTURE_AUDIT_OK] "
    f"parents={len(parent_identifiers)} rows={len(rows)}"
)
PY

echo "[MUTAGENICITY_CANDIDATE_POOL_GENERATION_OK]"
