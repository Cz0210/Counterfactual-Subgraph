#!/usr/bin/env bash
# CPU-only package and streaming verification after exact final merge.
#SBATCH --job-name=t8-hier-package
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
require_env() { [[ -n "${!1:-}" ]] || { echo "missing required environment variable: $1" >&2; exit 64; }; }
for name in T8_EXECUTION_WORKTREE T8_EXPECTED_COMMIT T8_PYTHON T8_PARTITION_MANIFEST \
  T8_FULL_SHARDS_ROOT T8_HIERARCHICAL_MERGE_ROOT T8_CANARY_PARITY_RECEIPT \
  T8_ENVIRONMENT_MANIFEST T8_HIERARCHICAL_SLURM_INVENTORY T8_RESOURCE_METRICS \
  T8_HIERARCHICAL_PACKAGE_ROOT T8_HIERARCHICAL_GROUP_PLAN \
  T8_HIERARCHICAL_GROUPS_ROOT; do require_env "$name"; done

set +u
source ~/.bashrc
set -u
conda activate "${T8_CONDA_ENV:-smiles_pip118}"
cd "$T8_EXECUTION_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export T8_CPU_ONLY=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
[[ "$(git rev-parse HEAD)" == "$T8_EXPECTED_COMMIT" ]] || { echo "execution commit mismatch" >&2; exit 65; }

job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
case "$job_tmp_base" in /share/*|/ssdfs/*) echo "package scratch is not node-local" >&2; exit 67;; esac
job_tmp="$(mktemp -d "$job_tmp_base/t8-hier-package-${SLURM_JOB_ID}.XXXXXX")"
child_pid=""
cleanup() { rm -rf "$job_tmp"; }
terminate_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  exit 143
}
trap cleanup EXIT
trap terminate_child TERM INT
mkdir -p "$(dirname "$T8_HIERARCHICAL_PACKAGE_ROOT")"

echo "python=$T8_PYTHON"
"$T8_PYTHON" --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "execution_commit=$T8_EXPECTED_COMMIT"
"$T8_PYTHON" scripts/hpc/t8/package_hierarchical_merge.py \
  --config configs/hpc.yaml \
  --partition-manifest "$T8_PARTITION_MANIFEST" \
  --shards-root "$T8_FULL_SHARDS_ROOT" \
  --merge-root "$T8_HIERARCHICAL_MERGE_ROOT" \
  --group-plan "$T8_HIERARCHICAL_GROUP_PLAN" \
  --groups-root "$T8_HIERARCHICAL_GROUPS_ROOT" \
  --parity-receipt "$T8_CANARY_PARITY_RECEIPT" \
  --environment-manifest "$T8_ENVIRONMENT_MANIFEST" \
  --slurm-inventory "$T8_HIERARCHICAL_SLURM_INVENTORY" \
  --resource-metrics "$T8_RESOURCE_METRICS" \
  --packaging-commit "$T8_EXPECTED_COMMIT" \
  --scratch-root "$job_tmp" \
  --output-root "$T8_HIERARCHICAL_PACKAGE_ROOT" &
child_pid=$!
wait "$child_pid"
child_pid=""
echo "T8_HPC_HIERARCHICAL_PACKAGE_PASS root=$T8_HIERARCHICAL_PACKAGE_ROOT"
