#!/bin/bash
# Submit the full HIV-derived SFT v3 pipeline with automatic Slurm dependencies.
#
# Example:
#   RUN_NAME=sft_v3_hiv_20260508_full \
#   ORACLE_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/oracle/aids_rf_model.pkl \
#   USE_ORACLE_RANKING=true \
#   MAX_STEPS=500 \
#   REPORT_TO=none \
#   bash scripts/slurm/submit_sft_v3_hiv_pipeline.sh
#
# Dependency graph:
#   build
#     ├─ audit
#     └─ train
#          └─ eval
#
# Notes:
# - This script is meant to run on the HPC login node, not through sbatch.
# - It preserves your current environment via --export=ALL, so stage-specific
#   variables like ORACLE_PATH, MAX_PARENTS, MAX_STEPS, REPORT_TO, MAX_EXAMPLES,
#   CHECKPOINT_PATH, and PROJECT_DIR can be set before invoking it.

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
RUN_NAME=${RUN_NAME:-sft_v3_hiv_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/${RUN_NAME}}

BUILD_SCRIPT=${BUILD_SCRIPT:-${PROJECT_DIR}/scripts/slurm/build_sft_v3_from_hiv.sh}
AUDIT_SCRIPT=${AUDIT_SCRIPT:-${PROJECT_DIR}/scripts/slurm/audit_sft_v3_dataset.sh}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${PROJECT_DIR}/scripts/slurm/train_sft_v3.sh}
EVAL_SCRIPT=${EVAL_SCRIPT:-${PROJECT_DIR}/scripts/slurm/eval_sft_v3_infer.sh}

SUBMIT_BUILD=${SUBMIT_BUILD:-true}
SUBMIT_AUDIT=${SUBMIT_AUDIT:-true}
SUBMIT_TRAIN=${SUBMIT_TRAIN:-true}
SUBMIT_EVAL=${SUBMIT_EVAL:-true}
PIPELINE_DRY_RUN=${PIPELINE_DRY_RUN:-false}

mkdir -p "${RUN_ROOT}/logs"

for required_script in "${BUILD_SCRIPT}" "${AUDIT_SCRIPT}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}"; do
  if [ ! -f "${required_script}" ]; then
    echo "[ERROR] Required script not found: ${required_script}"
    exit 1
  fi
done

echo "===== SFT V3 HIV PIPELINE ====="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "BUILD_SCRIPT=${BUILD_SCRIPT}"
echo "AUDIT_SCRIPT=${AUDIT_SCRIPT}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "SUBMIT_BUILD=${SUBMIT_BUILD}"
echo "SUBMIT_AUDIT=${SUBMIT_AUDIT}"
echo "SUBMIT_TRAIN=${SUBMIT_TRAIN}"
echo "SUBMIT_EVAL=${SUBMIT_EVAL}"
echo "PIPELINE_DRY_RUN=${PIPELINE_DRY_RUN}"
echo "==============================="

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

submit_job() {
  local dependency="$1"
  local script_path="$2"
  local stage_name="$3"

  local cmd=(sbatch --parsable --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_NAME=${RUN_NAME},RUN_ROOT=${RUN_ROOT}")
  if [ -n "${dependency}" ]; then
    cmd+=(--dependency "${dependency}")
  fi
  cmd+=("${script_path}")

  echo "Submitting ${stage_name}:"
  printf '  %q ' "${cmd[@]}"
  printf '\n'

  if is_truthy "${PIPELINE_DRY_RUN}"; then
    echo "  [DRY RUN] ${stage_name} not submitted."
    printf 'dryrun-%s\n' "${stage_name}"
    return 0
  fi

  "${cmd[@]}"
}

build_job_id=""
audit_job_id=""
train_job_id=""
eval_job_id=""

if is_truthy "${SUBMIT_BUILD}"; then
  build_job_id="$(submit_job "" "${BUILD_SCRIPT}" "build")"
  build_job_id="$(printf '%s\n' "${build_job_id}" | tail -n 1)"
  build_job_id="${build_job_id%%;*}"
  echo "build_job_id=${build_job_id}"
else
  echo "[ERROR] SUBMIT_BUILD=false is not supported for the current pipeline helper."
  echo "        The dataset build stage is the root dependency for later stages."
  exit 1
fi

if is_truthy "${SUBMIT_AUDIT}"; then
  audit_job_id="$(submit_job "afterok:${build_job_id}" "${AUDIT_SCRIPT}" "audit")"
  audit_job_id="$(printf '%s\n' "${audit_job_id}" | tail -n 1)"
  audit_job_id="${audit_job_id%%;*}"
  echo "audit_job_id=${audit_job_id}"
fi

if is_truthy "${SUBMIT_TRAIN}"; then
  train_job_id="$(submit_job "afterok:${build_job_id}" "${TRAIN_SCRIPT}" "train")"
  train_job_id="$(printf '%s\n' "${train_job_id}" | tail -n 1)"
  train_job_id="${train_job_id%%;*}"
  echo "train_job_id=${train_job_id}"
fi

if is_truthy "${SUBMIT_EVAL}"; then
  if [ -z "${train_job_id}" ]; then
    echo "[ERROR] SUBMIT_EVAL=true requires SUBMIT_TRAIN=true in the current helper."
    exit 1
  fi
  eval_job_id="$(submit_job "afterok:${train_job_id}" "${EVAL_SCRIPT}" "eval")"
  eval_job_id="$(printf '%s\n' "${eval_job_id}" | tail -n 1)"
  eval_job_id="${eval_job_id%%;*}"
  echo "eval_job_id=${eval_job_id}"
fi

cat <<EOF
===== SUBMISSION SUMMARY =====
RUN_NAME=${RUN_NAME}
RUN_ROOT=${RUN_ROOT}
build_job_id=${build_job_id}
audit_job_id=${audit_job_id:-<not submitted>}
train_job_id=${train_job_id:-<not submitted>}
eval_job_id=${eval_job_id:-<not submitted>}

Expected artifact tree:
  ${RUN_ROOT}/dataset
  ${RUN_ROOT}/audit
  ${RUN_ROOT}/train
  ${RUN_ROOT}/eval
  ${RUN_ROOT}/logs
==============================
EOF
