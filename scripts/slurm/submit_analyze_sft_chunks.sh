#!/bin/bash
# Submit chunked SFT fragment-distribution audit jobs to Slurm.
#
# Example:
#   INPUT_FILE=/share/home/u20526/czx/counterfactual-subgraph/data/sft_train.jsonl \
#   CHUNK_SIZE=200 \
#   RUN_PREFIX=sft_audit_train \
#   EXTRA_ARGS="--skip-deleteability-check --progress-every 20" \
#   scripts/slurm/submit_analyze_sft_chunks.sh

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
INPUT_FILE=${INPUT_FILE:-${PROJECT_DIR}/data/sft_train.jsonl}
CHUNK_SIZE=${CHUNK_SIZE:-200}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-}
RUN_PREFIX=${RUN_PREFIX:-analyze_sft_chunk}
EXTRA_ARGS=${EXTRA_ARGS:-}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hpc/logs/${RUN_PREFIX}}

if [ ! -f "${INPUT_FILE}" ]; then
  echo "[ERROR] INPUT_FILE not found: ${INPUT_FILE}"
  exit 1
fi

if [ "${CHUNK_SIZE}" -le 0 ]; then
  echo "[ERROR] CHUNK_SIZE must be > 0."
  exit 1
fi

if [ -z "${END_INDEX}" ]; then
  case "${INPUT_FILE}" in
    *.jsonl)
      END_INDEX=$(wc -l < "${INPUT_FILE}")
      ;;
    *.txt)
      END_INDEX=$(grep -c '^\[Sample [0-9]\+\]$' "${INPUT_FILE}")
      ;;
    *)
      echo "[ERROR] Could not infer END_INDEX for ${INPUT_FILE}. Please set END_INDEX explicitly."
      exit 1
      ;;
  esac
fi

mkdir -p "${OUTPUT_DIR}"

echo "===== CHUNK SUBMIT CONFIG ====="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "START_INDEX=${START_INDEX}"
echo "END_INDEX=${END_INDEX}"
echo "RUN_PREFIX=${RUN_PREFIX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<unset>}"
echo "==============================="

chunk_start=${START_INDEX}
while [ "${chunk_start}" -lt "${END_INDEX}" ]; do
  remaining=$((END_INDEX - chunk_start))
  if [ "${remaining}" -lt "${CHUNK_SIZE}" ]; then
    chunk_size=${remaining}
  else
    chunk_size=${CHUNK_SIZE}
  fi

  run_name="${RUN_PREFIX}_start${chunk_start}_count${chunk_size}"
  summary_json="${OUTPUT_DIR}/${run_name}.summary.json"
  details_csv="${OUTPUT_DIR}/${run_name}.details.csv"
  warn_log="${OUTPUT_DIR}/${run_name}.warn.log"

  echo "Submitting chunk: start=${chunk_start} size=${chunk_size} run_name=${run_name}"
  sbatch \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",INPUT_FILE="${INPUT_FILE}",SUMMARY_JSON="${summary_json}",DETAILS_CSV="${details_csv}",WARN_LOG="${warn_log}",RUN_NAME="${run_name}",START_INDEX="${chunk_start}",MAX_SAMPLES="${chunk_size}",PROGRESS_EVERY="${PROGRESS_EVERY}",EXTRA_ARGS="${EXTRA_ARGS}" \
    "${PROJECT_DIR}/scripts/slurm/analyze_sft_fragment_distribution.sh"

  chunk_start=$((chunk_start + chunk_size))
done

echo "All chunk jobs submitted."
