#!/usr/bin/env bash
# CPU-only exact final merge after all hierarchical groups pass.
#SBATCH --job-name=t8-hier-final
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
require_env() { [[ -n "${!1:-}" ]] || { echo "missing required environment variable: $1" >&2; exit 64; }; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
for name in T8_EXECUTION_WORKTREE T8_EXPECTED_COMMIT T8_PYTHON \
  T8_HIERARCHICAL_GROUP_PLAN T8_EXPECTED_GROUP_PLAN_FILE_SHA256 \
  T8_HIERARCHICAL_GROUPS_ROOT T8_HIERARCHICAL_FINAL_STATE_ROOT \
  T8_HIERARCHICAL_MERGE_ROOT; do require_env "$name"; done

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
[[ "$(sha256_file "$T8_HIERARCHICAL_GROUP_PLAN")" == "$T8_EXPECTED_GROUP_PLAN_FILE_SHA256" ]] || { echo "group plan SHA mismatch" >&2; exit 66; }

job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
case "$job_tmp_base" in /share/*|/ssdfs/*) echo "final scratch is not node-local" >&2; exit 67;; esac
job_tmp="$(mktemp -d "$job_tmp_base/t8-hier-final-${SLURM_JOB_ID}.XXXXXX")"
child_pid=""
cleanup() { rm -rf "$job_tmp"; }
requeue_on_usr1() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
terminate_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  exit 143
}
trap cleanup EXIT
trap requeue_on_usr1 USR1
trap terminate_child TERM INT
mkdir -p "$T8_HIERARCHICAL_FINAL_STATE_ROOT" "$(dirname "$T8_HIERARCHICAL_MERGE_ROOT")"

echo "python=$T8_PYTHON"
"$T8_PYTHON" --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "execution_commit=$T8_EXPECTED_COMMIT"
"$T8_PYTHON" scripts/hpc/t8/run_hierarchical_final_merge.py \
  --config configs/hpc.yaml \
  --group-plan "$T8_HIERARCHICAL_GROUP_PLAN" \
  --groups-root "$T8_HIERARCHICAL_GROUPS_ROOT" \
  --state-root "$T8_HIERARCHICAL_FINAL_STATE_ROOT" \
  --scratch-root "$job_tmp" \
  --output-root "$T8_HIERARCHICAL_MERGE_ROOT" &
child_pid=$!
wait "$child_pid"
child_pid=""
echo "T8_HPC_HIERARCHICAL_FINAL_PASS root=$T8_HIERARCHICAL_MERGE_ROOT"
