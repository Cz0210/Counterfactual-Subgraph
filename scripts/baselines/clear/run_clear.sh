#!/usr/bin/env bash
# Run official CLEAR stages from baselines/clear_official/src.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/baselines/clear/common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/baselines/clear/run_clear.sh <dataset> <stage>

Datasets:
  community | ogbg_molhiv | imdb_m

Stages:
  pred | train | test | baseline_random | baseline_IST | baseline_RM | all
EOF
}

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 2
fi

DATASET="$1"
STAGE="$2"

ensure_clear_dirs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  JOB_SUFFIX="_job${SLURM_JOB_ID}"
fi
LOG_FILE="${CLEAR_LOG_DIR}/clear_${DATASET}_${STAGE}_${TIMESTAMP}${JOB_SUFFIX}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[CLEAR_RUN] dataset=${DATASET}"
echo "[CLEAR_RUN] stage=${STAGE}"
echo "[CLEAR_RUN] log_file=${LOG_FILE}"
check_clear_dataset "${DATASET}"
print_clear_env

if [ ! -f "${CLEAR_SRC_DIR}/train_pred.py" ] || [ ! -f "${CLEAR_SRC_DIR}/main.py" ]; then
  echo "[CLEAR_ERROR] CLEAR official source is incomplete under: ${CLEAR_SRC_DIR}" >&2
  exit 1
fi

cd "${CLEAR_SRC_DIR}"
echo "[CLEAR_RUN] changed directory to $(pwd)"

run_command() {
  echo "[CLEAR_COMMAND] $*"
  "$@"
}

run_stage() {
  local stage="$1"
  case "${stage}" in
    pred)
      run_command python train_pred.py --dataset "${DATASET}" --epochs 600 --lr 0.001 --batch_size 500
      ;;
    train)
      run_command python main.py --dataset "${DATASET}" --experiment_type train --epochs 1000 --lr 0.001 --batch_size 500
      ;;
    test)
      run_command python main.py --dataset "${DATASET}" --experiment_type test --batch_size 500
      ;;
    baseline_random)
      run_command python main.py --dataset "${DATASET}" --experiment_type baseline --baseline_type random --batch_size 500
      ;;
    baseline_IST)
      run_command python main.py --dataset "${DATASET}" --experiment_type baseline --baseline_type IST --batch_size 500
      ;;
    baseline_RM)
      run_command python main.py --dataset "${DATASET}" --experiment_type baseline --baseline_type RM --batch_size 500
      ;;
    all)
      run_stage pred
      run_stage train
      run_stage test
      ;;
    *)
      echo "[CLEAR_ERROR] Unsupported CLEAR stage: ${stage}" >&2
      usage >&2
      return 2
      ;;
  esac
}

run_stage "${STAGE}"

echo "[CLEAR_DONE] dataset=${DATASET} stage=${STAGE}"
echo "[CLEAR_DONE] log_file=${LOG_FILE}"
