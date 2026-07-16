#!/bin/bash
# Confirmed resource template: scripts/baselines/clear/slurm_clear.sbatch
# Its A800 partition/GRES combination has successful job records in logs/clear/.
#SBATCH --job-name=fgw_sota_gpu
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
set +u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
CONDA_SH=/share/home/u20526/anaconda3/etc/profile.d/conda.sh

if [ -f "${HOME}/.bashrc" ]; then
  source "${HOME}/.bashrc"
fi
if [ ! -f "${CONDA_SH}" ]; then
  echo "[ERROR] Conda setup script not found: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-smiles_pip118}"

cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
export MPLBACKEND=Agg
mkdir -p logs

FIG3_REPORT_DIR=${FIG3_REPORT_DIR:-outputs/hpc/eval/paper/molclr_node_fgw_q30_main_figure3_table2}
FIG4_CSV=${FIG4_CSV:-outputs/hpc/eval/paper/molclr_node_fgw_dense_threshold_k20/fgw_dense_k20_figure4_fgw_coverage_vs_threshold.csv}
FINAL_OUT=${FINAL_OUT:-outputs/hpc/eval/paper/molclr_node_fgw_sota_figures}
Q20=${Q20:-0.0229636285221722}
Q30=${Q30:-0.0328363645853374}
FIGURE4_DISPLAY_MIN=${FIGURE4_DISPLAY_MIN:-0.015}

echo "===== FGW SOTA GPU PLOT ====="
echo "hostname=$(hostname)"
echo "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "partition=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "python_path=$(command -v python)"
echo "FIG3_REPORT_DIR=${FIG3_REPORT_DIR}"
echo "FIG4_CSV=${FIG4_CSV}"
echo "FINAL_OUT=${FINAL_OUT}"
echo "Q20=${Q20}"
echo "Q30=${Q30}"
echo "FIGURE4_DISPLAY_MIN=${FIGURE4_DISPLAY_MIN}"
echo "MPLBACKEND=${MPLBACKEND}"
echo "============================="

nvidia-smi || true

if [ ! -d "${FIG3_REPORT_DIR}" ]; then
  echo "[ERROR] Figure 3 report directory not found: ${FIG3_REPORT_DIR}" >&2
  exit 2
fi
if [ ! -f "${FIG4_CSV}" ]; then
  echo "[ERROR] Figure 4 CSV not found: ${FIG4_CSV}" >&2
  exit 2
fi

python -m py_compile scripts/plot_fgw_sota_figures.py

# Validate the precise Figure 3 source schema on the allocated compute node.
python - "${FIG3_REPORT_DIR}" "${Q30}" <<'PY'
import csv
import math
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
q30 = float(sys.argv[2])
priority = (
    "fgw_q30_k10_main_figure3_fgw_coverage_cost_vs_k.csv",
    "figure3_fgw_coverage_cost_vs_k.csv",
)
path = None
for name in priority:
    candidate = report_dir / name
    if candidate.is_file():
        path = candidate
        break
if path is None:
    candidates = sorted(report_dir.rglob("*figure3*coverage*cost*.csv"))
    if len(candidates) != 1:
        raise SystemExit(f"[ERROR] Expected one Figure 3 CSV under {report_dir}, found {candidates}")
    path = candidates[0]

with path.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
columns = list(rows[0]) if rows else []
required = {"method", "k", "theta", "coverage"}
cost_candidates = (
    "conditional_median_cost",
    "Conditional median cost",
    "theta_covered_conditional_median_cost",
    "covered_conditional_median_cost",
    "conditional_median_cost_covered",
)
cost_column = next((name for name in cost_candidates if name in columns), None)
missing = required - set(columns)
if missing or cost_column is None:
    raise SystemExit(
        f"[ERROR] Figure 3 schema invalid: missing={sorted(missing)}, "
        f"conditional_cost={cost_column}, columns={columns}"
    )
normalized = {}
for row in rows:
    key = row["method"].strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key.startswith("ours"):
        name = "Ours"
    elif key.startswith("globalgce"):
        name = "GlobalGCE"
    elif key.startswith("clear"):
        name = "CLEAR"
    elif key.startswith("gcf"):
        name = "GCFExplainer"
    else:
        continue
    normalized.setdefault(name, []).append(row)
if set(normalized) != {"Ours", "GlobalGCE", "CLEAR", "GCFExplainer"}:
    raise SystemExit(f"[ERROR] Figure 3 methods are incomplete: {sorted(normalized)}")
print("[FGW_FIGURE3_CSV_AUDIT]")
print("path=", path)
print("shape=", (len(rows), len(columns)))
print("columns=", columns)
print("methods=", sorted(normalized))
all_k = [int(float(row["k"])) for row in rows]
print("k_min=", min(all_k), "k_max=", max(all_k))
for method, method_rows in sorted(normalized.items()):
    k10 = [row for row in method_rows if int(float(row["k"])) == 10]
    if not k10:
        raise SystemExit(f"[ERROR] Missing K=10 for {method}")
    row = min(k10, key=lambda item: abs(float(item["theta"]) - q30))
    print(
        f"K10 {method}: theta={row['theta']} coverage={row['coverage']} "
        f"conditional_median_cost={row[cost_column]}"
    )
PY

python scripts/plot_fgw_sota_figures.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --figure3-report-dir "${FIG3_REPORT_DIR}" \
  --figure4-csv "${FIG4_CSV}" \
  --output-dir "${FINAL_OUT}" \
  --q20 "${Q20}" \
  --q30 "${Q30}" \
  --figure4-display-min "${FIGURE4_DISPLAY_MIN}"

expected_outputs=(
  figure3_main_k1_10_coverage_conditional_cost.png
  figure3_main_k1_10_coverage_conditional_cost.pdf
  figure3_supplement_k1_20_coverage_conditional_cost.png
  figure3_supplement_k1_20_coverage_conditional_cost.pdf
  figure4_main_low_cost_ccrcov_0_q30.png
  figure4_main_low_cost_ccrcov_0_q30.pdf
  figure4_supplement_full_ccrcov_0_010.png
  figure4_supplement_full_ccrcov_0_010.pdf
  table2_main_k10_q30_compact.csv
  table2_main_k10_q30_compact.md
  table2_main_k10_q30_compact.png
  table2_main_k10_q30_compact.pdf
  figure4_low_cost_auc_0_q30.csv
  selected_figure3_prefix_data.csv
  selected_figure4_threshold_data.csv
  sota_presentation_audit.txt
)
for filename in "${expected_outputs[@]}"; do
  if [ ! -s "${FINAL_OUT}/${filename}" ]; then
    echo "[ERROR] Expected non-empty output missing: ${FINAL_OUT}/${filename}" >&2
    exit 3
  fi
done

echo "[FGW_SOTA_GPU_PLOT_SUCCESS]"
