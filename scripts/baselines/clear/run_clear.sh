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
  community | ogbg_molhiv | aids | imdb_m

Stages:
  pred | train | test | export_test | export_test_small | baseline_random | baseline_IST | baseline_RM | all
EOF
}

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 2
fi

DATASET="$1"
STAGE="$2"
CLEAR_BATCH_SIZE="${CLEAR_BATCH_SIZE:-500}"
CLEAR_EXPORT_DIR="${CLEAR_EXPORT_DIR:-../../../outputs/hpc/baselines/clear}"
CLEAR_EXPORT_FULL_TEST="${CLEAR_EXPORT_FULL_TEST:-1}"
CLEAR_EXPORT_MAX_ITEMS="${CLEAR_EXPORT_MAX_ITEMS:-}"

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

if [ -f "${SCRIPT_DIR}/apply_clear_patches.sh" ]; then
  bash "${SCRIPT_DIR}/apply_clear_patches.sh"
fi

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
      run_command python train_pred.py --dataset "${DATASET}" --epochs 600 --lr 0.001 --batch_size "${CLEAR_BATCH_SIZE}"
      ;;
    train)
      run_command python main.py --dataset "${DATASET}" --experiment_type train --epochs 1000 --lr 0.001 --batch_size "${CLEAR_BATCH_SIZE}"
      ensure_clear_cfe_checkpoint_aliases "${DATASET}"
      ;;
    test)
      ensure_clear_cfe_checkpoint_aliases "${DATASET}"
      run_command python main.py --dataset "${DATASET}" --experiment_type test --batch_size "${CLEAR_BATCH_SIZE}"
      ;;
    export_test)
      ensure_clear_cfe_checkpoint_aliases "${DATASET}"
      export_args=(python main.py --dataset "${DATASET}" --experiment_type test --batch_size "${CLEAR_BATCH_SIZE}" --export_counterfactuals --export_dir "${CLEAR_EXPORT_DIR}")
      if [ "${CLEAR_EXPORT_FULL_TEST}" != "0" ]; then
        export_args+=(--export_full_test)
      fi
      if [ -n "${CLEAR_EXPORT_MAX_ITEMS}" ]; then
        export_args+=(--export_max_items "${CLEAR_EXPORT_MAX_ITEMS}")
      fi
      run_command "${export_args[@]}"
      ;;
    export_test_small)
      ensure_clear_cfe_checkpoint_aliases "${DATASET}"
      export_args=(python main.py --dataset "${DATASET}" --experiment_type test --batch_size "${CLEAR_BATCH_SIZE}" --export_counterfactuals --export_dir "${CLEAR_EXPORT_DIR}")
      if [ -n "${CLEAR_EXPORT_MAX_ITEMS}" ]; then
        export_args+=(--export_max_items "${CLEAR_EXPORT_MAX_ITEMS}")
      fi
      run_command "${export_args[@]}"
      ;;
    baseline_random)
      run_command python main.py --dataset "${DATASET}" --experiment_type baseline --baseline_type random --batch_size "${CLEAR_BATCH_SIZE}"
      ;;
    baseline_IST)
      run_command python main.py --dataset "${DATASET}" --experiment_type baseline --baseline_type IST --batch_size "${CLEAR_BATCH_SIZE}"
      ;;
    baseline_RM)
      run_command python main.py --dataset "${DATASET}" --experiment_type baseline --baseline_type RM --batch_size "${CLEAR_BATCH_SIZE}"
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
