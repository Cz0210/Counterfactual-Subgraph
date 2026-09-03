#!/usr/bin/env bash
# CPU-only exact gSpan parity canary.  This intentionally overrides the
# repository's GPU Slurm baseline: the offloaded stage has no neural inference.
#SBATCH --job-name=t8-gspan-canary
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

select_partition() {
  if [[ -n "${HPC_CPU_PARTITION:-}" ]]; then
    printf '%s\n' "$HPC_CPU_PARTITION"
    return
  fi
  local preferred
  preferred="$(sinfo -h -o '%P %a' 2>/dev/null | awk '$2 == "up" {gsub(/\*/, "", $1); if ($1 == "intel") {print $1; found=1; exit}; if (!fallback && $1 ~ /^(cpu|compute|normal|batch)$/) fallback=$1} END {if (!found && fallback) print fallback}' || true)"
  if [[ -n "$preferred" ]]; then
    printf '%s\n' "$preferred"
  else
    printf '%s\n' "${HPC_FALLBACK_PARTITION:-intel}"
  fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  partition="$(select_partition)"
  exec sbatch --parsable --partition="$partition" --export=ALL "$0" "$@"
fi

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

for name in \
  T8_EXECUTION_WORKTREE T8_EXPECTED_COMMIT T8_PYTHON \
  T8_GRAPHS_JSONL T8_INPUT_MANIFEST T8_EXPECTED_INPUT_MANIFEST_SHA256 \
  T8_EXPECTED_CONFIG_SHA256 T8_EXPECTED_HPC_CONFIG_SHA256 \
  T8_OFFICIAL_SRC T8_CANARY_ROOT; do
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$T8_EXPECTED_COMMIT" ]]; then
  echo "execution commit mismatch: expected=$T8_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 65
fi
actual_input_sha="$(sha256_file "$T8_INPUT_MANIFEST")"
if [[ "$actual_input_sha" != "$T8_EXPECTED_INPUT_MANIFEST_SHA256" ]]; then
  echo "input manifest SHA mismatch" >&2
  exit 66
fi
actual_hpc_config_sha="$(sha256_file configs/hpc.yaml)"
if [[ "$actual_hpc_config_sha" != "$T8_EXPECTED_HPC_CONFIG_SHA256" ]]; then
  echo "configs/hpc.yaml SHA mismatch" >&2
  exit 67
fi

"$T8_PYTHON" -c 'import hashlib,json,sys; p=json.load(open(sys.argv[1])); graph_sha=hashlib.sha256(open(sys.argv[4],"rb").read()).hexdigest(); assert p["mining_config_sha256"] == sys.argv[2], "mining config SHA mismatch"; assert p["hpc_runtime_config"]["sha256"] == sys.argv[3], "HPC config/input binding mismatch"; assert p["transaction_binding"]["graph_jsonl_sha256"] == graph_sha, "graph/input binding mismatch"; assert p["split_scope"] == "train_only"; assert p["calibration_payload_included"] is False; assert p["test_payload_included"] is False; assert p["matrix_publication_allowed_from_hpc"] is False' "$T8_INPUT_MANIFEST" "$T8_EXPECTED_CONFIG_SHA256" "$T8_EXPECTED_HPC_CONFIG_SHA256" "$T8_GRAPHS_JSONL"

job_tmp_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
job_tmp="$(mktemp -d "$job_tmp_base/t8-gspan-canary-${SLURM_JOB_ID}.XXXXXX")"
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

mkdir -p "$T8_CANARY_ROOT" "$T8_CANARY_ROOT/shards"
partition_manifest="$T8_CANARY_ROOT/partition_manifest.json"
reference_root="$T8_CANARY_ROOT/reference"
merged_root="$T8_CANARY_ROOT/merged"
parity_receipt="$T8_CANARY_ROOT/exact_parity.json"
shard_count="${T8_CANARY_SHARD_COUNT:-2}"
if [[ ! "$shard_count" =~ ^[12]$ ]]; then
  echo "T8_CANARY_SHARD_COUNT must be 1 or 2 for the fixed two-unit canary" >&2
  exit 64
fi

# A caller may pin a previously audited canonical prefix.  Otherwise enumerate
# root 0 deterministically and select the PREFIX_SUBTREE with minimum
# (support_hint, partition_id).  The subsequent bounded manifest validates that
# the supplied/selected ID is a real root-0 prefix, so neither route can
# silently widen the canary.
canary_prefix_unit_id="${T8_CANARY_PREFIX_UNIT_ID:-}"
canary_prefix_selection_mode="EXPLICIT_ENVIRONMENT"
prefix_catalog_sha256=""
if [[ -z "$canary_prefix_unit_id" ]]; then
  canary_prefix_selection_mode="AUTO_MIN_SUPPORT_THEN_PARTITION_ID"
  prefix_catalog="$job_tmp/root0-prefix-catalog.json"
  "$T8_PYTHON" scripts/hpc/t8/build_partition_manifest.py \
    --config configs/hpc.yaml \
    --graphs-jsonl "$T8_GRAPHS_JSONL" \
    --input-manifest "$T8_INPUT_MANIFEST" \
    --expected-commit "$T8_EXPECTED_COMMIT" \
    --official-src "$T8_OFFICIAL_SRC" \
    --output "$prefix_catalog" \
    --shard-count 1 \
    --min-support "${T8_MIN_SUPPORT:-2}" \
    --min-vertices "${T8_MIN_VERTICES:-3}" \
    --max-vertices "${T8_MAX_VERTICES:-20}" \
    --top-k "${T8_TOP_K:-20}" \
    --split-root-indices 0 \
    --split-depth "${T8_SPLIT_DEPTH:-3}" \
    --included-root-indices 0 \
    --canary-root-indices 0
  canary_prefix_unit_id="$("$T8_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); u=sorted((x for x in p["partitions"] if x["root_index"] == 0 and x["partition_type"] == "PREFIX_SUBTREE"), key=lambda x:(int(x["support_hint"]),x["partition_id"])); assert u, "root 0 has no canonical PREFIX_SUBTREE"; print(u[0]["partition_id"])' "$prefix_catalog")"
  prefix_catalog_sha256="$(sha256_file "$prefix_catalog")"
fi
echo "canary_prefix_unit_id=$canary_prefix_unit_id"

echo "python=$T8_PYTHON"
"$T8_PYTHON" --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "execution_commit=$actual_commit"
echo "input_manifest_sha256=$actual_input_sha"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"

"$T8_PYTHON" scripts/hpc/t8/build_partition_manifest.py \
  --config configs/hpc.yaml \
  --graphs-jsonl "$T8_GRAPHS_JSONL" \
  --input-manifest "$T8_INPUT_MANIFEST" \
  --expected-commit "$T8_EXPECTED_COMMIT" \
  --official-src "$T8_OFFICIAL_SRC" \
  --output "$partition_manifest" \
  --shard-count "$shard_count" \
  --min-support "${T8_MIN_SUPPORT:-2}" \
  --min-vertices "${T8_MIN_VERTICES:-3}" \
  --max-vertices "${T8_MAX_VERTICES:-20}" \
  --top-k "${T8_TOP_K:-20}" \
  --split-root-indices 0 \
  --split-depth "${T8_SPLIT_DEPTH:-3}" \
  --included-root-indices "22" \
  --canary-root-indices "0,22" \
  --included-unit-id "$canary_prefix_unit_id"

selection_receipt="$T8_CANARY_ROOT/canary_prefix_selection.json"
"$T8_PYTHON" -c 'import hashlib,json,os,sys,tempfile; from pathlib import Path; manifest_path=Path(sys.argv[1]); output=Path(sys.argv[2]); selected_id=sys.argv[3]; mode=sys.argv[4]; catalog_sha=sys.argv[5] or None; m=json.loads(manifest_path.read_text()); selected=[u for u in m["partitions"] if u["partition_id"]==selected_id]; assert m["scope"]=="SELECTED_PARTITION_CANARY" and m["canary_root_indices"]==[0,22] and m["whole_root_indices"]==[22] and m["selected_partition_ids"]==[selected_id]; assert len(selected)==1 and selected[0]["root_index"]==0 and selected[0]["partition_type"]=="PREFIX_SUBTREE"; p={"schema_version":"t8_hpc_canary_prefix_selection_v1","state":"PASS","selection_mode":mode,"selection_order":"SUPPORT_HINT_ASC_PARTITION_ID_ASC","selected_partition_id":selected_id,"selected_support_hint":int(selected[0]["support_hint"]),"selected_root_index":0,"small_complete_root_index":22,"partition_manifest_sha256":m["manifest_sha256"],"root0_catalog_file_sha256":catalog_sha,"execution_commit":sys.argv[6],"input_manifest_file_sha256":sys.argv[7],"mining_config_sha256":sys.argv[8]}; canonical=lambda value:(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode(); p["selection_receipt_sha256"]=hashlib.sha256(canonical(p)).hexdigest(); data=canonical(p); fd,name=tempfile.mkstemp(prefix="."+output.name+".",dir=output.parent); f=os.fdopen(fd,"wb"); f.write(data); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(name,output); d=os.open(output.parent,os.O_RDONLY); os.fsync(d); os.close(d)' \
  "$partition_manifest" "$selection_receipt" "$canary_prefix_unit_id" \
  "$canary_prefix_selection_mode" "$prefix_catalog_sha256" \
  "$T8_EXPECTED_COMMIT" "$T8_EXPECTED_INPUT_MANIFEST_SHA256" \
  "$T8_EXPECTED_CONFIG_SHA256"

"$T8_PYTHON" scripts/hpc/t8/run_exact_reference_canary.py \
  --config configs/hpc.yaml \
  --partition-manifest "$partition_manifest" \
  --output-root "$reference_root" \
  --root-indices "0,22" \
  --scratch-root "$job_tmp/reference" &
child_pid=$!
wait "$child_pid"
child_pid=""

for ((index = 0; index < shard_count; index++)); do
  printf -v shard_name 'shard-%03d' "$index"
  "$T8_PYTHON" scripts/hpc/t8/run_exact_mining_shard.py \
    --config configs/hpc.yaml \
    --partition-manifest "$partition_manifest" \
    --shard-index "$index" \
    --output-root "$T8_CANARY_ROOT/shards/$shard_name" \
    --scratch-root "$job_tmp/$shard_name" \
    --flush-every "${T8_FLUSH_EVERY:-256}" &
  child_pid=$!
  wait "$child_pid"
  child_pid=""
done

"$T8_PYTHON" scripts/hpc/t8/merge_exact_shards.py \
  --config configs/hpc.yaml \
  --partition-manifest "$partition_manifest" \
  --shards-root "$T8_CANARY_ROOT/shards" \
  --output-root "$merged_root" \
  --scratch-root "$job_tmp/merge"

"$T8_PYTHON" scripts/hpc/t8/verify_exact_parity.py \
  --config configs/hpc.yaml \
  --partition-manifest "$partition_manifest" \
  --reference-root "$reference_root" \
  --merged-root "$merged_root" \
  --output "$parity_receipt"

echo "T8_HPC_EXACT_CANARY_PASS receipt=$parity_receipt selection=$selection_receipt"
