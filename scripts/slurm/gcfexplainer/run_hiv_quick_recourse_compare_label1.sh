#!/bin/bash
#SBATCH -J hiv_quick_rec_label1
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

PROJECT_ROOT=/share/home/u20526/czx/counterfactual-subgraph
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD

mkdir -p logs

TARGET_LABEL=${TARGET_LABEL:-1}
TEACHER_PATH=${TEACHER_PATH:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/oracle/aids_rf_model.pkl}
OURS_SELECTED_DIR=${OURS_SELECTED_DIR:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
TOP_K_LIST=${TOP_K_LIST:-"10 20"}
THETA_LIST=${THETA_LIST:-"0.05 0.10 0.15 0.20"}
MAX_GT_CANDIDATES=${MAX_GT_CANDIDATES:-2000}
SEED=${SEED:-13}
JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/comparison/hiv_quick/label1_${JOB_ID}}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
DISABLE_TQDM=${DISABLE_TQDM:-false}
ENABLE_CAMC=${ENABLE_CAMC:-true}
CAMC_DELTA_LIST=${CAMC_DELTA_LIST:-"0.1 0.2 0.3 0.5"}
CAMC_TOP_K_LIST=${CAMC_TOP_K_LIST:-"${TOP_K_LIST}"}
CAMC_MIN_MOTIF_ATOMS=${CAMC_MIN_MOTIF_ATOMS:-2}
CAMC_USE_STRICT_THETA=${CAMC_USE_STRICT_THETA:-false}
CAMC_EXTRACTION_THETA_LIST=${CAMC_EXTRACTION_THETA_LIST:-}
CAMC_EXTRA_FULLGRAPH_SELECTED_CSV=${CAMC_EXTRA_FULLGRAPH_SELECTED_CSV:-}

mkdir -p "${OUT_DIR}" logs
export OUT_DIR
export HIV_COMPARE_EXTERNAL_TEE=1
exec > >(tee -a "${OUT_DIR}/progress.log") 2>&1

if [ -z "${HIV_CSV:-}" ]; then
  HIV_CSV_CANDIDATES=()
  for root in outputs/hpc/sft_v3_hiv_runs data; do
    if [ -d "${root}" ]; then
      while IFS= read -r -d '' csv_path; do
        if [[ "${csv_path}" =~ [Hh][Ii][Vv] ]]; then
          HIV_CSV_CANDIDATES+=("${PROJECT_ROOT}/${csv_path}")
        fi
      done < <(find "${root}" -type f -name "*.csv" -print0 2>/dev/null | sort -z)
    fi
  done
  if [ "${#HIV_CSV_CANDIDATES[@]}" -ne 1 ]; then
    echo "[ERROR] Could not uniquely determine HIV_CSV." >&2
    echo "[ERROR] Set HIV_CSV explicitly, for example:" >&2
    echo "  HIV_CSV=/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv sbatch $0" >&2
    echo "[ERROR] candidates (${#HIV_CSV_CANDIDATES[@]}):" >&2
    printf '  %s\n' "${HIV_CSV_CANDIDATES[@]}" >&2
    exit 1
  fi
  HIV_CSV=${HIV_CSV_CANDIDATES[0]}
fi

if [ ! -f "${HIV_CSV}" ]; then
  echo "[ERROR] HIV_CSV not found: ${HIV_CSV}" >&2
  exit 1
fi
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] TEACHER_PATH not found: ${TEACHER_PATH}" >&2
  exit 1
fi
if [ ! -d "${OURS_SELECTED_DIR}" ]; then
  echo "[ERROR] OURS_SELECTED_DIR not found: ${OURS_SELECTED_DIR}" >&2
  exit 1
fi

echo "===== HIV QUICK RECOURSE COMPARISON ENV CHECK ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "PYTHONPATH=${PYTHONPATH}"
python --version
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
try:
    from rdkit import Chem
    print("rdkit available: true")
    print("rdkit Chem module:", Chem.__name__)
except Exception as exc:
    print("rdkit available: false")
    print("rdkit import error:", repr(exc))
PY
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "HIV_CSV=${HIV_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OURS_SELECTED_DIR=${OURS_SELECTED_DIR}"
echo "TOP_K_LIST=${TOP_K_LIST}"
echo "THETA_LIST=${THETA_LIST}"
echo "MAX_INPUTS=${MAX_INPUTS:-unset}"
echo "MAX_GT_CANDIDATES=${MAX_GT_CANDIDATES}"
echo "SEED=${SEED}"
echo "OUT_DIR=${OUT_DIR}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "DISABLE_TQDM=${DISABLE_TQDM}"
echo "ENABLE_CAMC=${ENABLE_CAMC}"
echo "CAMC_DELTA_LIST=${CAMC_DELTA_LIST}"
echo "CAMC_TOP_K_LIST=${CAMC_TOP_K_LIST}"
echo "CAMC_MIN_MOTIF_ATOMS=${CAMC_MIN_MOTIF_ATOMS}"
echo "CAMC_USE_STRICT_THETA=${CAMC_USE_STRICT_THETA}"
echo "CAMC_EXTRACTION_THETA_LIST=${CAMC_EXTRACTION_THETA_LIST:-unset}"
echo "CAMC_EXTRA_FULLGRAPH_SELECTED_CSV=${CAMC_EXTRA_FULLGRAPH_SELECTED_CSV:-unset}"
echo "==================================================="

read -r -a TOP_K_ARGS <<< "${TOP_K_LIST}"
read -r -a THETA_ARGS <<< "${THETA_LIST}"
read -r -a CAMC_DELTA_ARGS <<< "${CAMC_DELTA_LIST}"
read -r -a CAMC_TOP_K_ARGS <<< "${CAMC_TOP_K_LIST}"

CMD=(
  python scripts/eval/compare_hiv_recourse_baselines.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --hiv-csv "${HIV_CSV}"
  --teacher-path "${TEACHER_PATH}"
  --target-label "${TARGET_LABEL}"
  --ours-selected-dir "${OURS_SELECTED_DIR}"
  --top-k-list "${TOP_K_ARGS[@]}"
  --theta-list "${THETA_ARGS[@]}"
  --max-gt-candidates "${MAX_GT_CANDIDATES}"
  --out-dir "${OUT_DIR}"
  --seed "${SEED}"
  --progress-every "${PROGRESS_EVERY}"
  --camc-delta-list "${CAMC_DELTA_ARGS[@]}"
  --camc-top-k-list "${CAMC_TOP_K_ARGS[@]}"
  --camc-min-motif-atoms "${CAMC_MIN_MOTIF_ATOMS}"
)

if [ "${DISABLE_TQDM}" = "true" ]; then
  CMD+=(--disable-tqdm)
fi
if [ "${ENABLE_CAMC}" = "true" ]; then
  CMD+=(--enable-camc)
else
  CMD+=(--disable-camc)
fi
if [ "${CAMC_USE_STRICT_THETA}" = "true" ]; then
  CMD+=(--camc-use-strict-theta)
fi
if [ -n "${CAMC_EXTRACTION_THETA_LIST}" ]; then
  read -r -a CAMC_EXTRACTION_THETA_ARGS <<< "${CAMC_EXTRACTION_THETA_LIST}"
  CMD+=(--camc-extraction-theta-list "${CAMC_EXTRACTION_THETA_ARGS[@]}")
fi
if [ -n "${CAMC_EXTRA_FULLGRAPH_SELECTED_CSV}" ]; then
  read -r -a CAMC_EXTRA_FULLGRAPH_ARGS <<< "${CAMC_EXTRA_FULLGRAPH_SELECTED_CSV}"
  for extra_fullgraph_spec in "${CAMC_EXTRA_FULLGRAPH_ARGS[@]}"; do
    CMD+=(--extra-fullgraph-selected-csv "${extra_fullgraph_spec}")
  done
fi
if [ -n "${SMILES_COL:-}" ]; then
  CMD+=(--smiles-col "${SMILES_COL}")
fi
if [ -n "${LABEL_COL:-}" ]; then
  CMD+=(--label-col "${LABEL_COL}")
fi
if [ -n "${MAX_INPUTS:-}" ]; then
  CMD+=(--max-inputs "${MAX_INPUTS}")
fi

echo "===== RUNNING HIV QUICK RECOURSE COMPARISON ====="
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "===== comparison_table.csv ====="
cat "${OUT_DIR}/comparison_table.csv"

if [ "${ENABLE_CAMC}" = "true" ]; then
  echo "===== camc_comparison_table.csv ====="
  cat "${OUT_DIR}/camc_comparison_table.csv"
fi

echo "===== diagnostic_counts.json ====="
cat "${OUT_DIR}/diagnostic_counts.json"

echo "===== recourse_monotonicity_warnings ====="
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OUT_DIR"]) / "comparison_summary.json"
payload = json.loads(path.read_text())
print(json.dumps(payload.get("recourse_monotonicity_warnings", []), indent=2, sort_keys=True))
PY

if [ "${ENABLE_CAMC}" = "true" ]; then
  echo "===== camc_monotonicity_warnings ====="
  python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OUT_DIR"]) / "camc_summary.json"
payload = json.loads(path.read_text())
print(json.dumps(payload.get("camc_monotonicity_warnings", []), indent=2, sort_keys=True))
PY
fi

echo "===== MorganGenerator deprecation warning check ====="
MORGAN_WARNING_COUNT=$(grep -c "DEPRECATION WARNING: please use MorganGenerator" "${OUT_DIR}/progress.log" || true)
echo "morgan_generator_deprecation_warning_count=${MORGAN_WARNING_COUNT}"
if [ "${MORGAN_WARNING_COUNT}" -gt 0 ]; then
  echo "[WARNING] MorganGenerator deprecation warnings found in progress.log."
else
  echo "[OK] no MorganGenerator deprecation warnings found."
fi

echo "===== comparison_summary.json ====="
cat "${OUT_DIR}/comparison_summary.json"

echo "===== HIV QUICK RECOURSE COMPARISON DONE ====="
echo "comparison_table=${OUT_DIR}/comparison_table.csv"
echo "comparison_summary=${OUT_DIR}/comparison_summary.json"
if [ "${ENABLE_CAMC}" = "true" ]; then
  echo "camc_comparison_table=${OUT_DIR}/camc_comparison_table.csv"
  echo "camc_summary=${OUT_DIR}/camc_summary.json"
fi
echo "diagnostic_counts=${OUT_DIR}/diagnostic_counts.json"
echo "progress_log=${OUT_DIR}/progress.log"
