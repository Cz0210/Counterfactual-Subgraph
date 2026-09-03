#!/usr/bin/env bash
# CPU-only exhaustive exact mining array; no --gres is intentional.
#SBATCH --job-name=t8-gspan-full
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-0%1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

select_partition() {
  if [[ -n "${HPC_CPU_PARTITION:-}" ]]; then printf '%s\n' "$HPC_CPU_PARTITION"; return; fi
  local preferred
  preferred="$(sinfo -h -o '%P %a' 2>/dev/null | awk '$2 == "up" {gsub(/\*/, "", $1); if ($1 == "intel") {print $1; found=1; exit}; if (!fallback && $1 ~ /^(cpu|compute|normal|batch)$/) fallback=$1} END {if (!found && fallback) print fallback}' || true)"
  printf '%s\n' "${preferred:-${HPC_FALLBACK_PARTITION:-intel}}"
}
require_env() { [[ -n "${!1:-}" ]] || { echo "missing required environment variable: $1" >&2; exit 64; }; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

require_env T8_SHARD_COUNT
require_env T8_ARRAY_CONCURRENCY
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if (( T8_SHARD_COUNT < 1 || T8_ARRAY_CONCURRENCY < 1 )); then echo "invalid shard/concurrency" >&2; exit 64; fi
  last_index=$((T8_SHARD_COUNT - 1))
  concurrency="$T8_ARRAY_CONCURRENCY"
  if (( concurrency > T8_SHARD_COUNT )); then concurrency="$T8_SHARD_COUNT"; fi
  partition="$(select_partition)"
  exec sbatch --parsable --partition="$partition" --array="0-${last_index}%${concurrency}" --export=ALL "$0" "$@"
fi

for name in T8_EXECUTION_WORKTREE T8_EXPECTED_COMMIT T8_PYTHON T8_INPUT_MANIFEST \
  T8_EXPECTED_INPUT_MANIFEST_SHA256 T8_EXPECTED_CONFIG_SHA256 T8_EXPECTED_HPC_CONFIG_SHA256 \
  T8_PARTITION_MANIFEST T8_EXPECTED_PARTITION_MANIFEST_SHA256 \
  T8_CANARY_PARITY_RECEIPT T8_EXPECTED_CANARY_PARITY_SHA256 \
  T8_FULL_SHARDS_ROOT; do
  require_env "$name"
done
require_env SLURM_ARRAY_TASK_ID

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
"$T8_PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); f=json.load(open(sys.argv[2])); required=("patterns_equal","supports_equal","stable_preorder_equal","candidate_inputs_equal","rejection_events_equal","all_events_equal"); assert c["status"]=="PASS" and all(c[k] is True for k in required); assert c["first_event_divergence"] is None and c["first_pattern_divergence"] is None; assert c["search_space_scope"]=="SELECTED_PARTITION_CANARY" and c["scientific_search_pruned"] is False and c["approximation_used"] is False and c["matrix_write_enabled"] is False; assert c["scientific_input_sha256"]==f["scientific_input_sha256"] and c["provenance_sha256"]==f["provenance"]["provenance_sha256"] and c["target_branches"]==f["provenance"]["target_branches"]' "$T8_CANARY_PARITY_RECEIPT" "$T8_PARTITION_MANIFEST"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= T8_SHARD_COUNT )); then
  echo "array index outside configured shard count" >&2
  exit 70
fi

job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
job_tmp="$(mktemp -d "$job_tmp_base/t8-gspan-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")"
child_pid=""
cleanup() { rm -rf "$job_tmp"; }
requeue_on_usr1() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
}
terminate_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  exit 143
}
trap cleanup EXIT
trap requeue_on_usr1 USR1
trap terminate_child TERM INT

printf -v shard_name 'shard-%03d' "$SLURM_ARRAY_TASK_ID"
shard_root="$T8_FULL_SHARDS_ROOT/$shard_name"
mkdir -p "$T8_FULL_SHARDS_ROOT"
echo "python=$T8_PYTHON"
"$T8_PYTHON" --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "execution_commit=$T8_EXPECTED_COMMIT"
echo "partition_manifest_sha256=$T8_EXPECTED_PARTITION_MANIFEST_SHA256"
echo "shard_index=$SLURM_ARRAY_TASK_ID"

"$T8_PYTHON" scripts/hpc/t8/run_exact_mining_shard.py \
  --config configs/hpc.yaml \
  --partition-manifest "$T8_PARTITION_MANIFEST" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$shard_root" \
  --scratch-root "$job_tmp/active" \
  --flush-every "${T8_FLUSH_EVERY:-256}" &
child_pid=$!
wait "$child_pid"
child_pid=""
echo "T8_HPC_EXACT_SHARD_PASS shard=$SLURM_ARRAY_TASK_ID root=$shard_root"
