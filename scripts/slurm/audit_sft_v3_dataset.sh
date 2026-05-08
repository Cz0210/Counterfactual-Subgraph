#!/bin/bash
# Audit both train/val SFT v3 splits under one RUN_NAME-rooted directory layout.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=audit_sft_v3

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
RUN_NAME=${RUN_NAME:-sft_v3_hiv_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/${RUN_NAME}}
DATASET_DIR=${DATASET_DIR:-${RUN_ROOT}/dataset}
AUDIT_ROOT=${AUDIT_ROOT:-${RUN_ROOT}/audit}
AUDIT_SPLITS=${AUDIT_SPLITS:-train,val}
TRAIN_INPUT=${TRAIN_INPUT:-${DATASET_DIR}/sft_v3_hiv_train.jsonl}
VAL_INPUT=${VAL_INPUT:-${DATASET_DIR}/sft_v3_hiv_val.jsonl}
PROGRESS_EVERY=${PROGRESS_EVERY:-200}
START_INDEX=${START_INDEX:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
NEAR_PARENT_THRESHOLD=${NEAR_PARENT_THRESHOLD:-0.8}
TINY_FRAGMENT_THRESHOLD=${TINY_FRAGMENT_THRESHOLD:-0.08}
MID_SIZE_MIN=${MID_SIZE_MIN:-0.10}
MID_SIZE_MAX=${MID_SIZE_MAX:-0.55}
EXTRA_ARGS=${EXTRA_ARGS:-}
WARN_LOG=${WARN_LOG:-${RUN_ROOT}/logs/audit.warn.log}

cd /share/home/u20526/czx/counterfactual-subgraph
if [ "${PROJECT_DIR}" != "/share/home/u20526/czx/counterfactual-subgraph" ]; then
  cd "${PROJECT_DIR}"
fi
mkdir -p logs
mkdir -p "${AUDIT_ROOT}" "$(dirname "${WARN_LOG}")"

export PYTHONPATH=$PWD

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "python path: $(which python)"
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "AUDIT_ROOT=${AUDIT_ROOT}"
echo "AUDIT_SPLITS=${AUDIT_SPLITS}"
echo "TRAIN_INPUT=${TRAIN_INPUT}"
echo "VAL_INPUT=${VAL_INPUT}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "START_INDEX=${START_INDEX:-<unset>}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<unset>}"
echo "WARN_LOG=${WARN_LOG}"
echo "====================="

IFS=',' read -r -a splits <<< "${AUDIT_SPLITS}"

run_one_split() {
  local split="$1"
  local input_file=""
  local split_dir="${AUDIT_ROOT}/${split}"
  local summary_json="${split_dir}/sft_v3_hiv_${split}.summary.json"
  local details_csv="${split_dir}/sft_v3_hiv_${split}.details.csv"

  case "${split}" in
    train)
      input_file="${TRAIN_INPUT}"
      ;;
    val)
      input_file="${VAL_INPUT}"
      ;;
    *)
      echo "[ERROR] Unsupported split: ${split}. Expected one of: train,val"
      return 1
      ;;
  esac

  if [ ! -f "${input_file}" ]; then
    echo "[ERROR] Input file not found for split=${split}: ${input_file}"
    return 1
  fi

  mkdir -p "${split_dir}"

  cmd=(
    python
    -u
    scripts/analyze_sft_fragment_distribution.py
    --config
    configs/hpc.yaml
    --input
    "${input_file}"
    --summary-json
    "${summary_json}"
    --details-csv
    "${details_csv}"
    --near-parent-threshold
    "${NEAR_PARENT_THRESHOLD}"
    --tiny-fragment-threshold
    "${TINY_FRAGMENT_THRESHOLD}"
    --mid-size-min
    "${MID_SIZE_MIN}"
    --mid-size-max
    "${MID_SIZE_MAX}"
    --progress-every
    "${PROGRESS_EVERY}"
  )

  if [ -n "${START_INDEX}" ]; then
    cmd+=(--start-index "${START_INDEX}")
  fi
  if [ -n "${MAX_SAMPLES}" ]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [ -n "${EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    extra_args=( ${EXTRA_ARGS} )
    cmd+=("${extra_args[@]}")
  fi

  echo "===== RUNNING AUDIT split=${split} ====="
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2> >(tee -a "${WARN_LOG}" >&2)
}

for split in "${splits[@]}"; do
  normalized_split="$(echo "${split}" | xargs)"
  if [ -z "${normalized_split}" ]; then
    continue
  fi
  run_one_split "${normalized_split}"
done

echo "===== DONE ====="
