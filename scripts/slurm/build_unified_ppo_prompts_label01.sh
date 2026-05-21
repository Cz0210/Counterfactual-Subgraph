#!/bin/bash
#SBATCH -J build_unified_p01
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD

mkdir -p logs

DATASET_DIR=${DATASET_DIR:-${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset}
LABEL1_CSV=${LABEL1_CSV:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label1.csv}
LABEL0_CSV=${LABEL0_CSV:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0.csv}
LABEL1_JSON=${LABEL1_JSON:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label1.summary.json}
LABEL0_JSON=${LABEL0_JSON:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0.summary.json}
UNIFIED_CSV=${UNIFIED_CSV:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_unified_label01.csv}
UNIFIED_JSON=${UNIFIED_JSON:-${UNIFIED_CSV%.csv}.summary.json}
BALANCE_JSON=${BALANCE_JSON:-${UNIFIED_CSV%.csv}.balance.json}

SOURCE_INPUT_CSV=${SOURCE_INPUT_CSV:-}
FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS:-false}
MAX_PER_LABEL=${MAX_PER_LABEL:-0}
LABEL_COL=${LABEL_COL:-label}
SMILES_COL=${SMILES_COL:-smiles}
SEED=${SEED:-13}
LABEL_PROMPT_BUILDER=${LABEL_PROMPT_BUILDER:-scripts/build_label_ppo_prompt_csv.py}

case "${FORCE_REBUILD_PROMPTS}" in
  true|TRUE|1|yes|YES) FORCE_REBUILD_PROMPTS=true ;;
  *) FORCE_REBUILD_PROMPTS=false ;;
esac

discover_source_input_csv() {
  DATASET_DIR="${DATASET_DIR}" LABEL_COL="${LABEL_COL}" SMILES_COL="${SMILES_COL}" python - <<'PY'
import csv
import json
import os
from pathlib import Path

dataset_dir = Path(os.environ["DATASET_DIR"]).expanduser()
label_col = os.environ["LABEL_COL"]
smiles_col = os.environ["SMILES_COL"]
label_fallbacks = ("label", "original_label", "y", "HIV_active", "HIV", "activity", "class")
smiles_fallbacks = ("smiles", "parent_smiles", "SMILES")
derived_terms = (
    "ppo_prompts",
    "unified",
    "summary",
    "balance",
    "selected",
    "audit",
    "candidate",
)
preferred_names = (
    "sft_v3_hiv_train.jsonl",
    "sft_v3_hiv_train.csv",
    "train.jsonl",
    "train.csv",
    "train_split.csv",
    "hiv_train.csv",
)

def is_derived(path: Path) -> bool:
    lowered = path.name.lower()
    return any(term in lowered for term in derived_terms)

def has_column(header, requested):
    requested_lower = str(requested or "").strip().lower()
    available = {str(col).strip().lower() for col in header}
    if requested_lower and requested_lower in available:
        return True
    fallbacks = label_fallbacks if requested_lower in {item.lower() for item in label_fallbacks} else smiles_fallbacks
    return any(item.lower() in available for item in fallbacks)

def inspect(path, priority):
    try:
        if path.suffix.lower() == ".jsonl":
            header = []
            row_count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not header:
                        header = list(payload.keys())
                    row_count += 1
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or [])
                row_count = sum(1 for _ in reader)
    except Exception:
        return None
    return {
        "path": path,
        "priority": priority,
        "row_count": row_count,
        "has_required_cols": has_column(header, smiles_col) and has_column(header, label_col),
    }

if not dataset_dir.exists():
    print("")
    raise SystemExit(0)

ordered_paths = []
seen = set()
for index, name in enumerate(preferred_names):
    path = dataset_dir / name
    if path.is_file() and path not in seen:
        ordered_paths.append(path)
        seen.add(path)
for path in sorted(dataset_dir.glob("*.csv")):
    if path in seen or is_derived(path):
        continue
    ordered_paths.append(path)
    seen.add(path)
for path in sorted(dataset_dir.glob("*.jsonl")):
    if path in seen or is_derived(path):
        continue
    ordered_paths.append(path)
    seen.add(path)

inspected = []
for index, path in enumerate(ordered_paths):
    item = inspect(path, index)
    if item:
        inspected.append(item)
valid = [item for item in inspected if bool(item["has_required_cols"])]
if not valid:
    print("")
    raise SystemExit(0)

best = sorted(valid, key=lambda item: (-int(item["row_count"]), int(item["priority"])))[0]
print(best["path"])
PY
}

print_dataset_csv_diagnostics() {
  DATASET_DIR="${DATASET_DIR}" python - <<'PY'
import csv
import os
from pathlib import Path

dataset_dir = Path(os.environ["DATASET_DIR"]).expanduser()
print("[UNIFIED_PROMPT_SOURCE_DIAGNOSTICS] dataset_dir=", dataset_dir, sep="")
if not dataset_dir.exists():
    print("[UNIFIED_PROMPT_SOURCE_DIAGNOSTICS] dataset_dir does not exist")
    raise SystemExit(0)
data_paths = sorted(dataset_dir.glob("*.csv")) + sorted(dataset_dir.glob("*.jsonl"))
if not data_paths:
    print("[UNIFIED_PROMPT_SOURCE_DIAGNOSTICS] no CSV/JSONL files found")
    raise SystemExit(0)
for path in data_paths:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                first = next((line for line in handle if line.strip()), "")
            import json
            header = list(json.loads(first).keys()) if first else []
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or [])
    except Exception as exc:
        print(f"[UNIFIED_PROMPT_SOURCE_DIAGNOSTICS] {path}: failed_to_read={exc}")
        continue
    preview = ", ".join(header[:30])
    if len(header) > 30:
        preview += ", ..."
    print(f"[UNIFIED_PROMPT_SOURCE_DIAGNOSTICS] {path}: columns=[{preview}]")
PY
}

augment_unified_summary() {
  UNIFIED_CSV="${UNIFIED_CSV}" \
  UNIFIED_JSON="${UNIFIED_JSON}" \
  LABEL0_CSV="${LABEL0_CSV}" \
  LABEL1_CSV="${LABEL1_CSV}" \
  SOURCE_INPUT_CSV="${SOURCE_INPUT_CSV}" \
  SOURCE_MODE="${SOURCE_MODE}" \
  SEED="${SEED}" \
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

unified_csv = Path(os.environ["UNIFIED_CSV"]).expanduser().resolve()
summary_path = Path(os.environ["UNIFIED_JSON"]).expanduser().resolve()
label0_csv = str(Path(os.environ["LABEL0_CSV"]).expanduser().resolve())
label1_csv = str(Path(os.environ["LABEL1_CSV"]).expanduser().resolve())
source_input_csv = os.environ.get("SOURCE_INPUT_CSV", "")
source_mode = os.environ["SOURCE_MODE"]
seed = int(os.environ["SEED"])

with unified_csv.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

def normalize_label(value):
    text = str(value or "").strip().lower()
    if text in {"0", "0.0", "false", "inactive", "negative"}:
        return 0
    if text in {"1", "1.0", "true", "active", "positive"}:
        return 1
    return None

label_counts = {0: 0, 1: 0}
for row in rows:
    label = normalize_label(row.get("label"))
    if label in label_counts:
        label_counts[label] += 1

if summary_path.exists():
    data = json.loads(summary_path.read_text(encoding="utf-8"))
else:
    data = {}

data.update(
    {
        "num_total": len(rows),
        "num_label0": int(label_counts[0]),
        "num_label1": int(label_counts[1]),
        "source_mode": source_mode,
        "source_input_csv": str(Path(source_input_csv).expanduser().resolve()) if source_input_csv else "",
        "label0_csv": label0_csv,
        "label1_csv": label1_csv,
        "unified_csv": str(unified_csv),
        "shuffled": True,
        "seed": seed,
    }
)
summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(
    "[UNIFIED_PROMPT_LABEL_COUNTS] "
    f"num_total={data['num_total']} num_label0={data['num_label0']} "
    f"num_label1={data['num_label1']} source_mode={source_mode}"
)
PY
}

build_minimal_unified_csv() {
  LABEL0_CSV="${LABEL0_CSV}" \
  LABEL1_CSV="${LABEL1_CSV}" \
  UNIFIED_CSV="${UNIFIED_CSV}" \
  UNIFIED_JSON="${UNIFIED_JSON}" \
  SOURCE_INPUT_CSV="${SOURCE_INPUT_CSV}" \
  SOURCE_MODE="${SOURCE_MODE}" \
  SEED="${SEED}" \
  MAX_PER_LABEL="${MAX_PER_LABEL}" \
  python - <<'PY'
import csv
import json
import os
import random
from pathlib import Path

label0_csv = Path(os.environ["LABEL0_CSV"]).expanduser().resolve()
label1_csv = Path(os.environ["LABEL1_CSV"]).expanduser().resolve()
unified_csv = Path(os.environ["UNIFIED_CSV"]).expanduser().resolve()
summary_path = Path(os.environ["UNIFIED_JSON"]).expanduser().resolve()
source_input_csv = os.environ.get("SOURCE_INPUT_CSV", "")
source_mode = os.environ["SOURCE_MODE"]
seed = int(os.environ["SEED"])
max_per_label = int(os.environ.get("MAX_PER_LABEL") or 0)

def normalize_label(value):
    text = str(value or "").strip().lower()
    if text in {"0", "0.0", "false", "inactive", "negative"}:
        return 0
    if text in {"1", "1.0", "true", "active", "positive"}:
        return 1
    return None

def resolve_smiles(row):
    for key in ("smiles", "parent_smiles", "SMILES"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""

def load_minimal(path, expected_label):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    bad_labels = 0
    missing_smiles = 0
    for row in rows:
        smiles = resolve_smiles(row)
        label = normalize_label(row.get("label"))
        if not smiles:
            missing_smiles += 1
            continue
        if label != expected_label:
            bad_labels += 1
            continue
        output.append({"smiles": smiles, "label": expected_label})
    if bad_labels:
        raise ValueError(f"{path} has {bad_labels} rows whose label is not {expected_label}")
    if missing_smiles:
        raise ValueError(f"{path} has {missing_smiles} rows without smiles/parent_smiles")
    return output

rows0 = load_minimal(label0_csv, 0)
rows1 = load_minimal(label1_csv, 1)
rng0 = random.Random(seed)
rng1 = random.Random(seed + 1)
rng0.shuffle(rows0)
rng1.shuffle(rows1)
if max_per_label > 0:
    rows0 = rows0[:max_per_label]
    rows1 = rows1[:max_per_label]
combined = list(rows0) + list(rows1)
random.Random(seed).shuffle(combined)

unified_csv.parent.mkdir(parents=True, exist_ok=True)
summary_path.parent.mkdir(parents=True, exist_ok=True)
with unified_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["smiles", "label"])
    writer.writeheader()
    writer.writerows(combined)

summary = {
    "source_mode": source_mode,
    "source_input_csv": str(Path(source_input_csv).expanduser().resolve()) if source_input_csv else "",
    "label0_csv": str(label0_csv),
    "label1_csv": str(label1_csv),
    "unified_csv": str(unified_csv),
    "num_label0": len(rows0),
    "num_label1": len(rows1),
    "num_total": len(combined),
    "seed": seed,
    "shuffled": True,
    "max_per_label": max_per_label,
    "output_schema": ["smiles", "label"],
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(
    "[UNIFIED_PROMPT_LABEL_COUNTS] "
    f"num_total={summary['num_total']} num_label0={summary['num_label0']} "
    f"num_label1={summary['num_label1']} source_mode={source_mode}"
)
PY
}

SOURCE_MODE=unresolved
if [ -n "${SOURCE_INPUT_CSV}" ]; then
  if [ -f "${SOURCE_INPUT_CSV}" ]; then
    SOURCE_MODE=explicit_source_input_csv
  else
    echo "[ERROR] explicit SOURCE_INPUT_CSV does not exist: ${SOURCE_INPUT_CSV}"
    print_dataset_csv_diagnostics
    echo "Run with:"
    echo "sbatch --export=ALL,SOURCE_INPUT_CSV=/path/to/train.csv,LABEL_COL=HIV_active,SMILES_COL=smiles scripts/slurm/build_unified_ppo_prompts_label01.sh"
    exit 1
  fi
fi

if [ -z "${SOURCE_INPUT_CSV}" ]; then
  if [ "${FORCE_REBUILD_PROMPTS}" = "false" ] && [ -s "${LABEL0_CSV}" ] && [ -s "${LABEL1_CSV}" ]; then
    SOURCE_MODE=existing_label_prompt_csvs
  else
    SOURCE_INPUT_CSV="$(discover_source_input_csv)"
    if [ -n "${SOURCE_INPUT_CSV}" ]; then
      SOURCE_MODE=auto_discovered_source_input_csv
    fi
  fi
fi

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "DATASET_DIR=${DATASET_DIR}"
echo "SOURCE_INPUT_CSV=${SOURCE_INPUT_CSV}"
echo "SOURCE_MODE=${SOURCE_MODE}"
echo "LABEL1_CSV=${LABEL1_CSV}"
echo "LABEL0_CSV=${LABEL0_CSV}"
echo "UNIFIED_CSV=${UNIFIED_CSV}"
echo "MAX_PER_LABEL=${MAX_PER_LABEL}"
echo "FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS}"
echo "LABEL_COL=${LABEL_COL}"
echo "SMILES_COL=${SMILES_COL}"
echo "SEED=${SEED}"
echo "LABEL_PROMPT_BUILDER=${LABEL_PROMPT_BUILDER}"
echo "====================="
echo "[UNIFIED_PROMPT_BUILD_CONFIG] DATASET_DIR=${DATASET_DIR} SOURCE_INPUT_CSV=${SOURCE_INPUT_CSV} LABEL_COL=${LABEL_COL} SMILES_COL=${SMILES_COL} MAX_PER_LABEL=${MAX_PER_LABEL} FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS} SEED=${SEED} label_prompt_builder=${LABEL_PROMPT_BUILDER}"
echo "[UNIFIED_PROMPT_SOURCE_MODE] source_mode=${SOURCE_MODE} source_input_csv=${SOURCE_INPUT_CSV:-None} label0_csv=${LABEL0_CSV} label1_csv=${LABEL1_CSV}"

mkdir -p "${DATASET_DIR}"

if [ "${SOURCE_MODE}" = "unresolved" ]; then
  echo "[ERROR] could not resolve SOURCE_INPUT_CSV and existing label0/label1 prompt CSVs are not both available."
  print_dataset_csv_diagnostics
  echo "Run with:"
  echo "sbatch scripts/slurm/build_sft_v3_hiv_ppo_prompts_label0_same_as_label1.sh"
  echo "or:"
  echo "sbatch --export=ALL,SOURCE_INPUT_CSV=/path/to/train.csv,LABEL_COL=HIV_active,SMILES_COL=smiles scripts/slurm/build_unified_ppo_prompts_label01.sh"
  exit 1
fi

if [ ! -f "${LABEL_PROMPT_BUILDER}" ]; then
  echo "[ERROR] label prompt builder not found: ${LABEL_PROMPT_BUILDER}"
  exit 1
fi

if [ "${SOURCE_MODE}" != "existing_label_prompt_csvs" ] && { [ "${FORCE_REBUILD_PROMPTS}" = "true" ] || [ ! -s "${LABEL0_CSV}" ]; }; then
  python "${LABEL_PROMPT_BUILDER}" \
    --config configs/hpc.yaml \
    --source-path "${SOURCE_INPUT_CSV}" \
    --target-label 0 \
    --out-csv "${LABEL0_CSV}" \
    --out-json "${LABEL0_JSON}" \
    --label-col "${LABEL_COL}" \
    --smiles-col "${SMILES_COL}"
else
  echo "skip label0 build: existing non-empty ${LABEL0_CSV}"
fi

if [ "${SOURCE_MODE}" != "existing_label_prompt_csvs" ] && { [ "${FORCE_REBUILD_PROMPTS}" = "true" ] || [ ! -s "${LABEL1_CSV}" ]; }; then
  python "${LABEL_PROMPT_BUILDER}" \
    --config configs/hpc.yaml \
    --source-path "${SOURCE_INPUT_CSV}" \
    --target-label 1 \
    --out-csv "${LABEL1_CSV}" \
    --out-json "${LABEL1_JSON}" \
    --label-col "${LABEL_COL}" \
    --smiles-col "${SMILES_COL}"
else
  echo "skip label1 build: existing non-empty ${LABEL1_CSV}"
fi

if [ ! -s "${LABEL0_CSV}" ]; then
  echo "[ERROR] label0 prompt CSV is missing or empty: ${LABEL0_CSV}"
  exit 1
fi

if [ ! -s "${LABEL1_CSV}" ]; then
  echo "[ERROR] label1 prompt CSV is missing or empty: ${LABEL1_CSV}"
  exit 1
fi

build_minimal_unified_csv

python scripts/check_unified_prompt_balance.py \
  --config configs/hpc.yaml \
  --dataset-path "${UNIFIED_CSV}" \
  --out-json "${BALANCE_JSON}" \
  --block-size 50

echo "===== OUTPUT CHECK ====="
for output_path in "${LABEL0_CSV}" "${LABEL0_JSON}" "${LABEL1_CSV}" "${LABEL1_JSON}" "${UNIFIED_CSV}" "${UNIFIED_JSON}" "${BALANCE_JSON}"; do
  if [ -e "${output_path}" ]; then
    ls -lh "${output_path}"
  else
    echo "[WARN] optional output missing: ${output_path}"
  fi
done
echo "[UNIFIED_PROMPT_OUTPUTS] label0_csv=${LABEL0_CSV} label1_csv=${LABEL1_CSV} unified_csv=${UNIFIED_CSV} unified_summary=${UNIFIED_JSON} balance_summary=${BALANCE_JSON}"
echo "===== BALANCE SUMMARY ====="
cat "${BALANCE_JSON}"
