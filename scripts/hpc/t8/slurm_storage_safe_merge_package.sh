#!/usr/bin/env bash
# CPU-only, afterok replacement for the held legacy merge/package jobs.  The
# complete exact merge and verification live on node-local scratch; /share
# receives only one deterministic compressed bundle plus its small manifest.
#SBATCH --job-name=t8-storage-safe-merge
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

select_partition() {
  if [[ -n "${HPC_CPU_PARTITION:-}" ]]; then printf '%s\n' "$HPC_CPU_PARTITION"; return; fi
  local preferred
  preferred="$(sinfo -h -o '%P %a' 2>/dev/null | awk '$2 == "up" {gsub(/\*/, "", $1); if ($1 == "intel") {print $1; found=1; exit}; if (!fallback && $1 ~ /^(cpu|compute|normal|batch)$/) fallback=$1} END {if (!found && fallback) print fallback}' || true)"
  printf '%s\n' "${preferred:-${HPC_FALLBACK_PARTITION:-intel}}"
}
require_env() { [[ -n "${!1:-}" ]] || { echo "missing required environment variable: $1" >&2; exit 64; }; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

require_env T8_FULL_ARRAY_JOB_ID
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  [[ "$T8_FULL_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "invalid T8_FULL_ARRAY_JOB_ID" >&2; exit 64; }
  partition="$(select_partition)"
  exec sbatch --parsable --partition="$partition" --dependency="afterok:${T8_FULL_ARRAY_JOB_ID}" --export=ALL "$0" "$@"
fi

for name in T8_EXECUTION_WORKTREE T8_EXPECTED_COMMIT T8_PYTHON T8_INPUT_MANIFEST \
  T8_EXPECTED_INPUT_MANIFEST_SHA256 T8_EXPECTED_CONFIG_SHA256 T8_EXPECTED_HPC_CONFIG_SHA256 \
  T8_PARTITION_MANIFEST T8_EXPECTED_PARTITION_MANIFEST_SHA256 \
  T8_FULL_SHARDS_ROOT T8_CANARY_PARITY_RECEIPT \
  T8_EXPECTED_CANARY_PARITY_SHA256 T8_STORAGE_SAFE_RESULT_ROOT \
  T8_ENVIRONMENT_MANIFEST T8_SLURM_INVENTORY T8_RESOURCE_METRICS; do
  require_env "$name"
done

set +u
source ~/.bashrc
set -u
conda activate "${T8_CONDA_ENV:-smiles_pip118}"
cd "$T8_EXECUTION_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export T8_CPU_ONLY=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

[[ "$(git rev-parse HEAD)" == "$T8_EXPECTED_COMMIT" ]] || { echo "execution commit mismatch" >&2; exit 65; }
[[ "$(sha256_file "$T8_INPUT_MANIFEST")" == "$T8_EXPECTED_INPUT_MANIFEST_SHA256" ]] || { echo "input manifest SHA mismatch" >&2; exit 66; }
[[ "$(sha256_file "$T8_PARTITION_MANIFEST")" == "$T8_EXPECTED_PARTITION_MANIFEST_SHA256" ]] || { echo "partition manifest SHA mismatch" >&2; exit 67; }
[[ "$(sha256_file "$T8_CANARY_PARITY_RECEIPT")" == "$T8_EXPECTED_CANARY_PARITY_SHA256" ]] || { echo "canary parity receipt SHA mismatch" >&2; exit 68; }
[[ "$(sha256_file configs/hpc.yaml)" == "$T8_EXPECTED_HPC_CONFIG_SHA256" ]] || { echo "configs/hpc.yaml SHA mismatch" >&2; exit 69; }
"$T8_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["mining_config_sha256"] == sys.argv[2]; assert p["hpc_runtime_config"]["sha256"] == sys.argv[3]; assert p["split_scope"] == "train_only"; assert p["matrix_publication_allowed_from_hpc"] is False' "$T8_INPUT_MANIFEST" "$T8_EXPECTED_CONFIG_SHA256" "$T8_EXPECTED_HPC_CONFIG_SHA256"

case "$T8_STORAGE_SAFE_RESULT_ROOT" in
  /share/home/u20526/czx/*) ;;
  *) echo "result root must stay below /share/home/u20526/czx" >&2; exit 70 ;;
esac
case "$T8_STORAGE_SAFE_RESULT_ROOT" in
  /ssdfs/*) echo "/ssdfs is forbidden" >&2; exit 70 ;;
esac
[[ ! -e "$T8_STORAGE_SAFE_RESULT_ROOT" ]] || { echo "storage-safe result root must be fresh" >&2; exit 71; }

job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
case "$job_tmp_base" in
  /share/*|/ssdfs/*) echo "merge scratch is not node-local" >&2; exit 72 ;;
esac
job_tmp="$(mktemp -d "$job_tmp_base/t8-storage-safe-${SLURM_JOB_ID}.XXXXXX")"
child_pid=""
cleanup() { rm -rf "$job_tmp"; }
terminate_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  exit 143
}
trap cleanup EXIT
trap terminate_child TERM INT
mkdir -p "$(dirname "$T8_STORAGE_SAFE_RESULT_ROOT")"

echo "python=$T8_PYTHON"
"$T8_PYTHON" --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "execution_commit=$T8_EXPECTED_COMMIT"
echo "array_dependency=$T8_FULL_ARRAY_JOB_ID"
echo "node_local_scratch=$job_tmp"

"$T8_PYTHON" scripts/hpc/t8/run_storage_safe_merge_package.py \
  --config configs/hpc.yaml \
  --partition-manifest "$T8_PARTITION_MANIFEST" \
  --shards-root "$T8_FULL_SHARDS_ROOT" \
  --parity-receipt "$T8_CANARY_PARITY_RECEIPT" \
  --environment-manifest "$T8_ENVIRONMENT_MANIFEST" \
  --slurm-inventory "$T8_SLURM_INVENTORY" \
  --resource-metrics "$T8_RESOURCE_METRICS" \
  --packaging-commit "$T8_EXPECTED_COMMIT" \
  --scratch-root "$job_tmp" \
  --output-root "$T8_STORAGE_SAFE_RESULT_ROOT" \
  --minimum-reserve-bytes "${T8_MIN_PERSISTENT_RESERVE_BYTES:-2147483648}" \
  --reserve-fraction "${T8_PERSISTENT_RESERVE_FRACTION:-0.20}" &
child_pid=$!
wait "$child_pid"
child_pid=""

echo "T8_HPC_STORAGE_SAFE_RESULT_PASS root=$T8_STORAGE_SAFE_RESULT_ROOT"
