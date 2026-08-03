#!/bin/bash
#SBATCH --job-name=mut_gcf_recover
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$PROJECT_ROOT" ]] || {
  echo "[MUTAGENICITY_GCFEXPLAINER_RECOVERY_CONFIG_ERROR] PROJECT_ROOT is required." >&2
  exit 2
}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

FAILED_RUN_DIR="${FAILED_RUN_DIR:-}"
COUNTERFACTUALS_PATH="${COUNTERFACTUALS_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXPECTED_PROFILE="${EXPECTED_PROFILE:-full}"
EXPECTED_PARENT_LIMIT="${EXPECTED_PARENT_LIMIT:-1448}"
EXPECTED_M="${EXPECTED_M:-50000}"
EXPECTED_ALPHA="${EXPECTED_ALPHA:-1.0}"
EXPECTED_THETA="${EXPECTED_THETA:-0.05}"
EXPECTED_SEED="${EXPECTED_SEED:-13}"
EXPECTED_JOB_ID="${EXPECTED_JOB_ID:-2074516}"
EXPECTED_BYTES="${EXPECTED_BYTES:-210914250}"
EXPECTED_SHA256="${EXPECTED_SHA256:-3b4fc88c1498d830afab03cbbd3e8b0ff9aec45477a7872a17a4531cae43bab2}"

config_error() {
  echo "[MUTAGENICITY_GCFEXPLAINER_RECOVERY_CONFIG_ERROR] $1" >&2
  exit 2
}

[[ -n "$FAILED_RUN_DIR" ]] || config_error "FAILED_RUN_DIR must be provided."
[[ -n "$COUNTERFACTUALS_PATH" ]] || config_error "COUNTERFACTUALS_PATH must be provided."
[[ -n "$OUTPUT_DIR" ]] || config_error "OUTPUT_DIR must be provided."
[[ -d "$FAILED_RUN_DIR" ]] || config_error "FAILED_RUN_DIR does not exist: $FAILED_RUN_DIR"
[[ -f "$COUNTERFACTUALS_PATH" ]] || config_error "COUNTERFACTUALS_PATH does not exist: $COUNTERFACTUALS_PATH"
[[ "$OUTPUT_DIR" != "$FAILED_RUN_DIR" ]] || config_error "OUTPUT_DIR must differ from FAILED_RUN_DIR."
[[ "$EXPECTED_PROFILE" == "full" ]] || config_error "EXPECTED_PROFILE must be full."
[[ "$EXPECTED_PARENT_LIMIT" == "1448" ]] || config_error "EXPECTED_PARENT_LIMIT must be 1448."
[[ "$EXPECTED_M" == "50000" ]] || config_error "EXPECTED_M must be 50000."
[[ "$EXPECTED_ALPHA" == "1.0" ]] || config_error "EXPECTED_ALPHA must be 1.0."
[[ "$EXPECTED_THETA" == "0.05" ]] || config_error "EXPECTED_THETA must be 0.05."
[[ "$EXPECTED_SEED" == "13" ]] || config_error "EXPECTED_SEED must be 13."
[[ "$EXPECTED_JOB_ID" == "2074516" ]] || config_error "EXPECTED_JOB_ID must be 2074516."
[[ "$EXPECTED_BYTES" == "210914250" ]] || config_error "EXPECTED_BYTES must be 210914250."
[[ "$EXPECTED_SHA256" == "3b4fc88c1498d830afab03cbbd3e8b0ff9aec45477a7872a17a4531cae43bab2" ]] || \
  config_error "EXPECTED_SHA256 does not match the audited artifact."
EXPECTED_COUNTERFACTUALS_PATH="$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/gcfexplainer/full_v2/vrrw_m50000_alpha1_v1/counterfactuals.pt"
[[ "$(realpath "$COUNTERFACTUALS_PATH")" == "$EXPECTED_COUNTERFACTUALS_PATH" ]] || \
  config_error "COUNTERFACTUALS_PATH must be the audited job-2074516 artifact."

if [[ -e "$OUTPUT_DIR" ]]; then
  [[ -d "$OUTPUT_DIR" ]] || config_error "OUTPUT_DIR exists and is not a directory: $OUTPUT_DIR"
  if [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    config_error "OUTPUT_DIR must be absent or empty: $OUTPUT_DIR"
  fi
fi
OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
[[ -d "$OUTPUT_PARENT" ]] || config_error "OUTPUT_DIR parent must already exist: $OUTPUT_PARENT"

mkdir -p "$PROJECT_ROOT/logs"
echo "[MUTAGENICITY_GCFEXPLAINER_RECOVERY_STORAGE_PREFLIGHT]"
echo "filesystem_path=$OUTPUT_PARENT"
AVAILABLE_KIB="$(df -Pk "$OUTPUT_PARENT" | awk 'END {print $4}')"
REQUIRED_KIB=$((5 * 1024 * 1024))
echo "available_kib=$AVAILABLE_KIB"
echo "required_kib=$REQUIRED_KIB"
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ && "$AVAILABLE_KIB" -ge "$REQUIRED_KIB" ]] || \
  config_error "At least 5 GiB free space is required."

PROBE_PATH="$OUTPUT_PARENT/.mut_gcf_recovery_probe_${SLURM_JOB_ID:-$$}"
cleanup_probe() {
  rm -f "$PROBE_PATH"
}
trap cleanup_probe EXIT
dd if=/dev/zero of="$PROBE_PATH" bs=1M count=256 conv=fsync status=none
[[ "$(wc -c < "$PROBE_PATH")" -eq 268435456 ]] || config_error "256 MiB storage probe size mismatch."
rm -f "$PROBE_PATH"
trap - EXIT
echo "write_probe_bytes=268435456"
echo "write_probe_fsync=passed"

echo "hostname=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$PWD"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
python --version
echo "failed_run_dir=$FAILED_RUN_DIR"
echo "counterfactuals_path=$COUNTERFACTUALS_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "expected_job_id=$EXPECTED_JOB_ID"
echo "algorithm_rerun=false"
echo "calibration_loaded=false"
echo "test_loaded=false"

python scripts/baselines/gcfexplainer/recover_mutagenicity_vrrw_run.py \
  --failed-run-dir "$FAILED_RUN_DIR" \
  --counterfactuals-path "$COUNTERFACTUALS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --expected-profile "$EXPECTED_PROFILE" \
  --expected-parent-limit "$EXPECTED_PARENT_LIMIT" \
  --expected-m "$EXPECTED_M" \
  --expected-alpha "$EXPECTED_ALPHA" \
  --expected-theta "$EXPECTED_THETA" \
  --expected-seed "$EXPECTED_SEED" \
  --expected-job-id "$EXPECTED_JOB_ID" \
  --expected-bytes "$EXPECTED_BYTES" \
  --expected-sha256 "$EXPECTED_SHA256"

test -s "$OUTPUT_DIR/recovery_manifest.json"
test -s "$OUTPUT_DIR/run_manifest.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_GCFEXPLAINER_VRRW_RECOVERY_WRAPPER_OK]"
