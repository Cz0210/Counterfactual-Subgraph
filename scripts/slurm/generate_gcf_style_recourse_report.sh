#!/bin/bash
#SBATCH --job-name=gcf_style_report
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/eval/reports/gcf_style_molclr_node_fgw_final}
DISTANCE_LABEL=${DISTANCE_LABEL:-MolCLR-Node-FGW}
TABLE_PREFIX=${TABLE_PREFIX:-}
K=${K:-10}
THETA_STAR=${THETA_STAR:-0.0328}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
THRESHOLD_GRID=${THRESHOLD_GRID:-}
INSET_MAX_K=${INSET_MAX_K:-}
FIGURE3_COST_METRIC=${FIGURE3_COST_METRIC:-theta_covered_conditional_median_cost}
FIGURE3_COST_STAT=${FIGURE3_COST_STAT:-}
TABLE_COST_METRIC=${TABLE_COST_METRIC:-theta_covered_conditional_median_cost}
TABLE_COST_STAT=${TABLE_COST_STAT:-}
TABLE_INCLUDE_APPLICABLE_RATE=${TABLE_INCLUDE_APPLICABLE_RATE:-0}
TABLE_INCLUDE_MEDIAN_COST=${TABLE_INCLUDE_MEDIAN_COST:-0}
REFERENCE_PARENT_IDS=${REFERENCE_PARENT_IDS:-}
REFERENCE_PARENT_ID_COL=${REFERENCE_PARENT_ID_COL:-parent_id}
OURS_RUN=${OURS_RUN:-}
GLOBALGCE_RUN=${GLOBALGCE_RUN:-}
CLEAR_RUN=${CLEAR_RUN:-}
GCFEXPLAINER_RUN=${GCFEXPLAINER_RUN:-}

echo "===== GCF-STYLE RECOURSE REPORT ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DISTANCE_LABEL=${DISTANCE_LABEL}"
echo "TABLE_PREFIX=${TABLE_PREFIX}"
echo "K=${K}"
echo "THETA_STAR=${THETA_STAR}"
echo "BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES}"
echo "BOOTSTRAP_SEED=${BOOTSTRAP_SEED}"
echo "THRESHOLD_GRID=${THRESHOLD_GRID:-default_101_point_grid}"
echo "FIGURE3_COST_METRIC=${FIGURE3_COST_METRIC}"
echo "FIGURE3_COST_STAT=${FIGURE3_COST_STAT:-not_set}"
echo "TABLE_COST_METRIC=${TABLE_COST_METRIC}"
echo "TABLE_COST_STAT=${TABLE_COST_STAT:-not_set}"
echo "TABLE_INCLUDE_APPLICABLE_RATE=${TABLE_INCLUDE_APPLICABLE_RATE}"
echo "TABLE_INCLUDE_MEDIAN_COST=${TABLE_INCLUDE_MEDIAN_COST}"
echo "REFERENCE_PARENT_IDS=${REFERENCE_PARENT_IDS:-auto_from_ours_pair_details}"
echo "REFERENCE_PARENT_ID_COL=${REFERENCE_PARENT_ID_COL}"
echo "OURS_RUN=${OURS_RUN:-built_in_default}"
echo "GLOBALGCE_RUN=${GLOBALGCE_RUN:-built_in_default}"
echo "CLEAR_RUN=${CLEAR_RUN:-built_in_default}"
echo "GCFEXPLAINER_RUN=${GCFEXPLAINER_RUN:-built_in_default}"

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --output-dir "${OUTPUT_DIR}"
  --distance-label "${DISTANCE_LABEL}"
  --table-prefix "${TABLE_PREFIX}"
  --k "${K}"
  --theta-star "${THETA_STAR}"
  --figure3-cost-metric "${FIGURE3_COST_METRIC}"
  --table-cost-metric "${TABLE_COST_METRIC}"
  --table-include-applicable-rate "${TABLE_INCLUDE_APPLICABLE_RATE}"
  --table-include-median-cost "${TABLE_INCLUDE_MEDIAN_COST}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --seed "${BOOTSTRAP_SEED}"
)

if [ -n "${THRESHOLD_GRID}" ]; then
  args+=(--threshold-grid "${THRESHOLD_GRID}")
fi
if [ -n "${INSET_MAX_K}" ]; then
  args+=(--inset-max-k "${INSET_MAX_K}")
fi
if [ -n "${FIGURE3_COST_STAT}" ]; then
  args+=(--figure3-cost-stat "${FIGURE3_COST_STAT}")
fi
if [ -n "${TABLE_COST_STAT}" ]; then
  args+=(--table-cost-stat "${TABLE_COST_STAT}")
fi
if [ -n "${REFERENCE_PARENT_IDS}" ]; then
  args+=(--reference-parent-ids "${REFERENCE_PARENT_IDS}" --reference-parent-id-col "${REFERENCE_PARENT_ID_COL}")
fi
if [ -n "${OURS_RUN}" ] || [ -n "${GLOBALGCE_RUN}" ] || [ -n "${CLEAR_RUN}" ] || [ -n "${GCFEXPLAINER_RUN}" ]; then
  if [ -z "${OURS_RUN}" ] || [ -z "${GLOBALGCE_RUN}" ] || [ -z "${CLEAR_RUN}" ] || [ -z "${GCFEXPLAINER_RUN}" ]; then
    echo "[ERROR] Set all four run directories when overriding built-in report runs." >&2
    exit 1
  fi
  args+=(--run "Ours=${OURS_RUN}")
  args+=(--run "GlobalGCE=${GLOBALGCE_RUN}")
  args+=(--run "CLEAR=${CLEAR_RUN}")
  args+=(--run "GCFExplainer=${GCFEXPLAINER_RUN}")
fi

python scripts/generate_gcf_style_recourse_report.py "${args[@]}"
