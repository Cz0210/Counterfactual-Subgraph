#!/bin/bash
#SBATCH --job-name=globalgce_fgw_audit
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

RUN_DIR=${RUN_DIR:-outputs/hpc/eval/ccrcov_molclr_node_fgw_globalgce_frequency_top20_lam05}
SELECTED_TOP20=${SELECTED_TOP20:-outputs/hpc/selectors/globalgce_fullgraph_frequency_top20/selected_top20_for_eval.csv}
REPORT_DIR=${REPORT_DIR:-outputs/hpc/eval/paper/molclr_node_fgw_gcf_style}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/audits/globalgce_frequency_top20_fgw_v2}
CORRECTED_OUTPUT_DIR=${CORRECTED_OUTPUT_DIR:-${RUN_DIR}/corrected_teacher_strict}
REFERENCE_PARENT_IDS=${REFERENCE_PARENT_IDS:-}
REFERENCE_PARENT_ID_COL=${REFERENCE_PARENT_ID_COL:-parent_id}
AUTO_REFERENCE_FROM_OURS=${AUTO_REFERENCE_FROM_OURS:-0}
REFERENCE_OURS_RUN=${REFERENCE_OURS_RUN:-}
EXPECTED_REFERENCE_PARENTS=${EXPECTED_REFERENCE_PARENTS:-1283}
VALID_CANDIDATES=${VALID_CANDIDATES:-}
METHOD_NAME=${METHOD_NAME:-globalgce_frequency_top20}
TARGET_LABEL=${TARGET_LABEL:-1}
THETA=${THETA:-0.0328}
TABLE_K=${TABLE_K:-10}
MAX_K=${MAX_K:-20}
FAIL_ON_CRITICAL=${FAIL_ON_CRITICAL:-0}

echo "===== GLOBALGCE FREQUENCY TOP20 NODE-FGW AUDIT ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_DIR=${RUN_DIR}"
echo "SELECTED_TOP20=${SELECTED_TOP20}"
echo "REPORT_DIR=${REPORT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CORRECTED_OUTPUT_DIR=${CORRECTED_OUTPUT_DIR}"
echo "REFERENCE_PARENT_IDS=${REFERENCE_PARENT_IDS:-none_all_label_parent_diagnostic}"
echo "REFERENCE_PARENT_ID_COL=${REFERENCE_PARENT_ID_COL}"
echo "AUTO_REFERENCE_FROM_OURS=${AUTO_REFERENCE_FROM_OURS}"
echo "REFERENCE_OURS_RUN=${REFERENCE_OURS_RUN:-unset}"
echo "EXPECTED_REFERENCE_PARENTS=${EXPECTED_REFERENCE_PARENTS}"
echo "VALID_CANDIDATES=${VALID_CANDIDATES:-auto}"
echo "METHOD_NAME=${METHOD_NAME}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "THETA=${THETA}"
echo "TABLE_K=${TABLE_K}"
echo "MAX_K=${MAX_K}"
echo "distance_recomputed=false"

if [ ! -f "${RUN_DIR}/details/pair_details.csv" ]; then
  echo "[ERROR] Missing pair details: ${RUN_DIR}/details/pair_details.csv"
  exit 2
fi
if [ ! -f "${SELECTED_TOP20}" ]; then
  echo "[ERROR] Missing selected Top20: ${SELECTED_TOP20}"
  exit 2
fi

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --run-dir "${RUN_DIR}"
  --selected-top20 "${SELECTED_TOP20}"
  --report-dir "${REPORT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --corrected-output-dir "${CORRECTED_OUTPUT_DIR}"
  --reference-parent-id-col "${REFERENCE_PARENT_ID_COL}"
  --expected-reference-parents "${EXPECTED_REFERENCE_PARENTS}"
  --method-name "${METHOD_NAME}"
  --target-label "${TARGET_LABEL}"
  --theta "${THETA}"
  --table-k "${TABLE_K}"
  --max-k "${MAX_K}"
)
if [ -n "${VALID_CANDIDATES}" ]; then
  args+=(--valid-candidates "${VALID_CANDIDATES}")
fi
if [ -n "${REFERENCE_PARENT_IDS}" ]; then
  args+=(--reference-parent-ids "${REFERENCE_PARENT_IDS}")
fi
if [ "${AUTO_REFERENCE_FROM_OURS}" = "1" ]; then
  args+=(--auto-reference-from-ours)
fi
if [ -n "${REFERENCE_OURS_RUN}" ]; then
  args+=(--reference-ours-run "${REFERENCE_OURS_RUN}")
fi
if [ "${FAIL_ON_CRITICAL}" = "1" ]; then
  args+=(--fail-on-critical)
fi

python scripts/audit_globalgce_frequency_top20_fgw.py "${args[@]}"

echo "[GLOBALGCE_FREQUENCY_FGW_AUDIT_OUTPUTS]"
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort
