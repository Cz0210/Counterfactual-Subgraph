#!/bin/bash
#SBATCH --job-name=globalgce_fgw_select
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Some HPC bashrc scripts reference unset variables. Keep nounset disabled
# until shell and conda initialization have completed.
set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

PAIR_DETAILS=${PAIR_DETAILS:-${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_molclr_node_fgw_medium_globalgce_lam05/details/pair_details.csv}
CANDIDATES_CSV=${CANDIDATES_CSV:-}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/hpc/selectors/globalgce_node_fgw_top20}
TOP_K=${TOP_K:-20}
METHOD_NAME=${METHOD_NAME:-globalgce}
CANDIDATE_SMILES_COL=${CANDIDATE_SMILES_COL:-candidate_smiles}
THRESHOLD=${THRESHOLD:-}
THRESHOLD_QUANTILE=${THRESHOLD_QUANTILE:-0.2}

echo "[GLOBALGCE_FGW_SELECTOR_CONFIG]"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PAIR_DETAILS=${PAIR_DETAILS}"
echo "CANDIDATES_CSV=${CANDIDATES_CSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOP_K=${TOP_K}"
echo "METHOD_NAME=${METHOD_NAME}"
echo "THRESHOLD=${THRESHOLD:-auto}"
echo "THRESHOLD_QUANTILE=${THRESHOLD_QUANTILE}"

if [ ! -f "${PAIR_DETAILS}" ]; then
  echo "[ERROR] PAIR_DETAILS does not exist: ${PAIR_DETAILS}"
  exit 2
fi
if [ -z "${CANDIDATES_CSV}" ] || [ ! -f "${CANDIDATES_CSV}" ]; then
  echo "[ERROR] CANDIDATES_CSV is required and must point to the evaluated GlobalGCE top2000 candidate CSV."
  echo "Example: CANDIDATES_CSV=/path/to/globalgce_top2000.csv sbatch $0"
  exit 2
fi

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --pair-details "${PAIR_DETAILS}"
  --candidates-csv "${CANDIDATES_CSV}"
  --out-dir "${OUT_DIR}"
  --top-k "${TOP_K}"
  --method-name "${METHOD_NAME}"
  --candidate-smiles-col "${CANDIDATE_SMILES_COL}"
)
if [ -n "${THRESHOLD}" ]; then
  args+=(--threshold "${THRESHOLD}")
else
  args+=(--threshold-quantile "${THRESHOLD_QUANTILE}")
fi

python scripts/select_fullgraph_candidates_by_fgw_coverage.py "${args[@]}"
