#!/usr/bin/env bash
# CPU-only T8 stress continuation, merge, and packaging wrapper.  The absence
# of GRES is intentional: exact gSpan and archive construction use no GPU.
#SBATCH --job-name=t8-stress-followup
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 64
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for name in T8_CONTROLLER_WORKTREE T8_EXPECTED_CONTROLLER_COMMIT T8_PYTHON; do
  require_env "$name"
done

set +u
source ~/.bashrc
set -u
conda activate "${T8_CONDA_ENV:-smiles_pip118}"
cd "$T8_CONTROLLER_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export T8_CPU_ONLY=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$T8_EXPECTED_CONTROLLER_COMMIT" ]]; then
  echo "controller commit mismatch: expected=$T8_EXPECTED_CONTROLLER_COMMIT actual=$actual_commit" >&2
  exit 65
fi

mode="followup"
if (( $# > 0 )) && [[ "$1" =~ ^(followup|refinement-canary|merge|package)$ ]]; then
  mode="$1"
  shift
fi

echo "python=$T8_PYTHON"
"$T8_PYTHON" --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "controller_commit=$actual_commit"
echo "mode=$mode"

case "$mode" in
  followup)
    exec "$T8_PYTHON" scripts/hpc/t8/run_stress_followup.py "$@"
    ;;
  refinement-canary)
    for name in \
      T8_SCIENCE_WORKTREE T8_EXPECTED_SCIENCE_COMMIT \
      T8_CANARY_ROOT T8_CANARY_TELEMETRY_ROOT; do
      require_env "$name"
    done
    science_commit="$(git -C "$T8_SCIENCE_WORKTREE" rev-parse HEAD)"
    if [[ "$science_commit" != "$T8_EXPECTED_SCIENCE_COMMIT" ]] || \
       [[ "$science_commit" != "481475c31d809577b791f4dd9002f5d2894c65b4" ]]; then
      echo "pinned science worktree mismatch: expected=481475c31d... actual=$science_commit" >&2
      exit 70
    fi
    job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
    science_pid=""
    monitor_pid=""
    terminate_refinement() {
      if [[ -n "$science_pid" ]] && kill -0 "$science_pid" 2>/dev/null; then
        kill -TERM "$science_pid" 2>/dev/null || true
      fi
      if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
      fi
      [[ -z "$science_pid" ]] || wait "$science_pid" 2>/dev/null || true
      [[ -z "$monitor_pid" ]] || wait "$monitor_pid" 2>/dev/null || true
      exit 143
    }
    trap terminate_refinement TERM INT
    bash "$T8_SCIENCE_WORKTREE/scripts/hpc/t8/slurm_canary.sh" &
    science_pid=$!
    "$T8_PYTHON" scripts/hpc/t8/run_stress_followup.py monitor \
      --config configs/hpc.yaml \
      --canary-root "$T8_CANARY_ROOT" \
      --telemetry-root "$T8_CANARY_TELEMETRY_ROOT" \
      --scratch-base "$job_tmp_base" \
      --slurm-job-id "${SLURM_JOB_ID:?SLURM_JOB_ID is required}" \
      --science-pid "$science_pid" &
    monitor_pid=$!
    set +e
    wait "$science_pid"
    science_status=$?
    wait "$monitor_pid"
    telemetry_status=$?
    set -e
    echo "science_status=$science_status telemetry_status=$telemetry_status"
    exit "$science_status"
    ;;
  merge)
    for name in \
      T8_INPUT_MANIFEST T8_EXPECTED_INPUT_MANIFEST_SHA256 \
      T8_EXPECTED_HPC_CONFIG_SHA256 T8_PARTITION_MANIFEST \
      T8_EXPECTED_PARTITION_MANIFEST_SHA256 T8_FULL_SHARDS_ROOT \
      T8_FULL_MERGE_ROOT; do
      require_env "$name"
    done
    [[ "$(sha256_file "$T8_INPUT_MANIFEST")" == "$T8_EXPECTED_INPUT_MANIFEST_SHA256" ]] || {
      echo "input manifest SHA mismatch" >&2
      exit 66
    }
    [[ "$(sha256_file configs/hpc.yaml)" == "$T8_EXPECTED_HPC_CONFIG_SHA256" ]] || {
      echo "HPC config SHA mismatch" >&2
      exit 67
    }
    [[ "$(sha256_file "$T8_PARTITION_MANIFEST")" == "$T8_EXPECTED_PARTITION_MANIFEST_SHA256" ]] || {
      echo "partition manifest SHA mismatch" >&2
      exit 68
    }
    job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
    job_tmp="$(mktemp -d "$job_tmp_base/t8-stress-merge-${SLURM_JOB_ID:-local}.XXXXXX")"
    child_pid=""
    cleanup() { rm -rf "$job_tmp"; }
    terminate_child() {
      if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
      fi
      exit 143
    }
    trap cleanup EXIT
    trap terminate_child TERM INT
    "$T8_PYTHON" scripts/hpc/t8/merge_exact_shards.py \
      --config configs/hpc.yaml \
      --partition-manifest "$T8_PARTITION_MANIFEST" \
      --shards-root "$T8_FULL_SHARDS_ROOT" \
      --output-root "$T8_FULL_MERGE_ROOT" \
      --scratch-root "$job_tmp/merge" &
    child_pid=$!
    wait "$child_pid"
    child_pid=""
    echo "T8_HPC_EXACT_MERGE_PASS root=$T8_FULL_MERGE_ROOT"
    ;;
  package)
    for name in \
      T8_INPUT_MANIFEST T8_EXPECTED_INPUT_MANIFEST_SHA256 \
      T8_EXPECTED_HPC_CONFIG_SHA256 T8_PARTITION_MANIFEST \
      T8_EXPECTED_PARTITION_MANIFEST_SHA256 T8_CANARY_PARITY_RECEIPT \
      T8_EXPECTED_CANARY_PARITY_SHA256 T8_FULL_MERGE_ROOT \
      T8_RESULT_BUNDLE T8_RESULT_MANIFEST T8_ENVIRONMENT_MANIFEST \
      T8_SLURM_INVENTORY T8_RESOURCE_METRICS; do
      require_env "$name"
    done
    [[ "$(sha256_file "$T8_INPUT_MANIFEST")" == "$T8_EXPECTED_INPUT_MANIFEST_SHA256" ]] || {
      echo "input manifest SHA mismatch" >&2
      exit 66
    }
    [[ "$(sha256_file configs/hpc.yaml)" == "$T8_EXPECTED_HPC_CONFIG_SHA256" ]] || {
      echo "HPC config SHA mismatch" >&2
      exit 67
    }
    [[ "$(sha256_file "$T8_PARTITION_MANIFEST")" == "$T8_EXPECTED_PARTITION_MANIFEST_SHA256" ]] || {
      echo "partition manifest SHA mismatch" >&2
      exit 68
    }
    [[ "$(sha256_file "$T8_CANARY_PARITY_RECEIPT")" == "$T8_EXPECTED_CANARY_PARITY_SHA256" ]] || {
      echo "canary parity SHA mismatch" >&2
      exit 69
    }
    "$T8_PYTHON" scripts/hpc/t8/build_result_bundle.py \
      --config configs/hpc.yaml \
      --partition-manifest "$T8_PARTITION_MANIFEST" \
      --merge-root "$T8_FULL_MERGE_ROOT" \
      --parity-receipt "$T8_CANARY_PARITY_RECEIPT" \
      --output-tar "$T8_RESULT_BUNDLE" \
      --output-manifest "$T8_RESULT_MANIFEST" \
      --environment-manifest "$T8_ENVIRONMENT_MANIFEST" \
      --slurm-inventory "$T8_SLURM_INVENTORY" \
      --resource-metrics "$T8_RESOURCE_METRICS"
    echo "T8_HPC_RESULT_BUNDLE_PASS bundle=$T8_RESULT_BUNDLE manifest=$T8_RESULT_MANIFEST"
    ;;
  *)
    echo "unknown mode: $mode (expected followup, merge, or package)" >&2
    exit 64
    ;;
esac
