#!/bin/bash
# HPC wrapper for the SFT fragment-distribution audit.
#
# Example:
#   sbatch --export=ALL,INPUT_FILE=/share/home/u20526/czx/counterfactual-subgraph/data/sft_train.jsonl,START_INDEX=0,MAX_SAMPLES=200,PROGRESS_EVERY=20 scripts/slurm/analyze_sft_fragment_distribution.sh

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=/share/home/u20526/czx/counterfactual-subgraph/logs/%j.out
#SBATCH --error=/share/home/u20526/czx/counterfactual-subgraph/logs/%j.err
#SBATCH --job-name=analyze_sft_fragment_distribution

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
RUN_NAME=${RUN_NAME:-analyze_sft_fragment_distribution_$(date +%Y%m%d_%H%M%S)}
INPUT_FILE=${INPUT_FILE:-${PROJECT_DIR}/data/sft_train.jsonl}
SUMMARY_JSON=${SUMMARY_JSON:-${PROJECT_DIR}/outputs/hpc/logs/${RUN_NAME}.summary.json}
DETAILS_CSV=${DETAILS_CSV:-${PROJECT_DIR}/outputs/hpc/logs/${RUN_NAME}.details.csv}
WARN_LOG=${WARN_LOG:-${PROJECT_DIR}/outputs/hpc/logs/${RUN_NAME}.warn.log}
START_INDEX=${START_INDEX:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
echo "python path: $(which python)"
python -V
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "DETAILS_CSV=${DETAILS_CSV}"
echo "WARN_LOG=${WARN_LOG}"
echo "START_INDEX=${START_INDEX:-<unset>}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<unset>}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<unset>}"
echo "====================="

cd "${PROJECT_DIR}"
mkdir -p logs
mkdir -p "$(dirname "${SUMMARY_JSON}")" "$(dirname "${DETAILS_CSV}")" "$(dirname "${WARN_LOG}")"

export PYTHONPATH=$PWD

echo "===== REPO CHECK ====="
echo "pwd after cd: $(pwd)"
echo "git root:"
git rev-parse --show-toplevel
echo "git commit:"
git rev-parse HEAD
echo "PYTHONPATH=${PYTHONPATH}"
echo "======================"

echo "===== RUNTIME PYTHON CHECK ====="
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
PY
echo "==============================="

cmd=(
  python
  -u
  scripts/analyze_sft_fragment_distribution.py
  --config
  configs/hpc.yaml
  --input
  "${INPUT_FILE}"
  --summary-json
  "${SUMMARY_JSON}"
  --details-csv
  "${DETAILS_CSV}"
)

if [ -n "${START_INDEX}" ]; then
  cmd+=(--start-index "${START_INDEX}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [ -n "${PROGRESS_EVERY}" ]; then
  cmd+=(--progress-every "${PROGRESS_EVERY}")
fi
if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_ARGS} )
  cmd+=("${extra_args[@]}")
fi

echo "===== RUNNING SFT AUDIT ====="
printf '%q ' "${cmd[@]}"
printf '\n'

set +e
"${cmd[@]}" 2>>"${WARN_LOG}"
EXIT_CODE=$?
set -e

echo "===== SFT AUDIT EXIT CODE: ${EXIT_CODE} ====="
echo "WARN_LOG=${WARN_LOG}"

if [ "${EXIT_CODE}" -ne 0 ]; then
  echo "[ERROR] SFT fragment distribution audit failed with exit code ${EXIT_CODE}."
  exit "${EXIT_CODE}"
fi

echo "===== DONE ====="
