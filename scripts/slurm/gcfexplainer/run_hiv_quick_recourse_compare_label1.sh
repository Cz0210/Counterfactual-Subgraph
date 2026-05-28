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

mkdir -p "${OUT_DIR}"

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
echo "==================================================="

read -r -a TOP_K_ARGS <<< "${TOP_K_LIST}"
read -r -a THETA_ARGS <<< "${THETA_LIST}"

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
)

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

echo "===== comparison_summary.json ====="
cat "${OUT_DIR}/comparison_summary.json"

echo "===== HIV QUICK RECOURSE COMPARISON DONE ====="
echo "comparison_table=${OUT_DIR}/comparison_table.csv"
echo "comparison_summary=${OUT_DIR}/comparison_summary.json"
