#!/usr/bin/env bash
# Build a fresh plan and submit group -> final -> package dependencies once.
set -euo pipefail
require_env() { [[ -n "${!1:-}" ]] || { echo "missing required environment variable: $1" >&2; exit 64; }; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
for name in T8_EXECUTION_WORKTREE T8_EXPECTED_COMMIT T8_PYTHON T8_PARTITION_MANIFEST \
  T8_FULL_SHARDS_ROOT T8_CANARY_PARITY_RECEIPT T8_ENVIRONMENT_MANIFEST \
  T8_RESOURCE_METRICS T8_HIERARCHICAL_CHAIN_ROOT; do require_env "$name"; done

set +u
source ~/.bashrc
set -u
conda activate "${T8_CONDA_ENV:-smiles_pip118}"
cd "$T8_EXECUTION_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
[[ "$(git rev-parse HEAD)" == "$T8_EXPECTED_COMMIT" ]] || { echo "execution commit mismatch" >&2; exit 65; }

# Historical held jobs are evidence.  Refuse to continue if either ceased to
# be held; this launcher has no mutation path for either historical job.
old_merge_job="${T8_OLD_HELD_MERGE_JOB_ID:-2536786}"
old_package_job="${T8_OLD_HELD_PACKAGE_JOB_ID:-2536787}"
timeout_merge_job="${T8_TIMEOUT_MERGE_JOB_ID:-2538830}"
for job_id in "$old_merge_job" "$old_package_job"; do
  row="$(squeue -h -j "$job_id" -o '%i|%T|%r' 2>/dev/null || true)"
  [[ "$row" == *"JobHeldUser"* ]] || { echo "historical job $job_id is not JobHeldUser: ${row:-MISSING}" >&2; exit 66; }
done
timeout_state="$(sacct -n -X -j "$timeout_merge_job" -o State -P 2>/dev/null | head -n 1 | cut -d'|' -f1 | cut -d'+' -f1 | tr -d ' ')"
[[ "$timeout_state" == "TIMEOUT" ]] || { echo "historical merge $timeout_merge_job is not TIMEOUT: ${timeout_state:-MISSING}" >&2; exit 66; }

[[ ! -e "$T8_HIERARCHICAL_CHAIN_ROOT" ]] || { echo "hierarchical chain root must be fresh" >&2; exit 67; }
mkdir -p "$T8_HIERARCHICAL_CHAIN_ROOT/control" "$T8_HIERARCHICAL_CHAIN_ROOT/artifacts/groups"
export T8_HIERARCHICAL_GROUPS_ROOT="$T8_HIERARCHICAL_CHAIN_ROOT/artifacts/groups"
export T8_HIERARCHICAL_FINAL_STATE_ROOT="$T8_HIERARCHICAL_CHAIN_ROOT/control/final"
export T8_HIERARCHICAL_MERGE_ROOT="$T8_HIERARCHICAL_CHAIN_ROOT/artifacts/merge"
export T8_HIERARCHICAL_PACKAGE_ROOT="$T8_HIERARCHICAL_CHAIN_ROOT/artifacts/package"
export T8_HIERARCHICAL_GROUP_PLAN="$T8_HIERARCHICAL_CHAIN_ROOT/control/group_plan.json"
export T8_HIERARCHICAL_SLURM_INVENTORY="$T8_HIERARCHICAL_CHAIN_ROOT/control/slurm_inventory.json"
array_adoption="$T8_HIERARCHICAL_CHAIN_ROOT/control/array_adoption_manifest.json"
export T8_ARRAY_ADOPTION_MANIFEST="$array_adoption"

"$T8_PYTHON" scripts/hpc/t8/build_hierarchical_merge_plan.py \
  --config configs/hpc.yaml \
  --partition-manifest "$T8_PARTITION_MANIFEST" \
  --shards-root "$T8_FULL_SHARDS_ROOT" \
  --array-adoption "$array_adoption" \
  --group-plan "$T8_HIERARCHICAL_GROUP_PLAN" \
  --group-count "${T8_HIERARCHICAL_GROUP_COUNT:-4}"
export T8_EXPECTED_GROUP_PLAN_FILE_SHA256="$(sha256_file "$T8_HIERARCHICAL_GROUP_PLAN")"
group_count="$($T8_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["group_count"])' "$T8_HIERARCHICAL_GROUP_PLAN")"
last_group=$((group_count - 1))
concurrency="${T8_HIERARCHICAL_GROUP_CONCURRENCY:-4}"
(( concurrency <= group_count )) || concurrency="$group_count"

group_job_id="$(sbatch --parsable --array="0-${last_group}%${concurrency}" --export=ALL scripts/hpc/t8/slurm_hierarchical_group_merge.sh)"
group_job_id="${group_job_id%%;*}"
final_job_id="$(sbatch --parsable --dependency="afterok:${group_job_id}" --export=ALL scripts/hpc/t8/slurm_hierarchical_final_merge.sh)"
final_job_id="${final_job_id%%;*}"
export T8_HIERARCHICAL_GROUP_JOB_ID="$group_job_id"
export T8_HIERARCHICAL_FINAL_JOB_ID="$final_job_id"
export T8_RESOURCE_METRICS="$T8_HIERARCHICAL_CHAIN_ROOT/control/resource_metrics.json"
package_job_id="$(sbatch --parsable --dependency="afterok:${final_job_id}" --export=ALL scripts/hpc/t8/slurm_hierarchical_package.sh)"
package_job_id="${package_job_id%%;*}"

"$T8_PYTHON" - "$T8_HIERARCHICAL_SLURM_INVENTORY" "$group_job_id" "$final_job_id" "$package_job_id" "$group_count" "$array_adoption" "$old_merge_job" "$old_package_job" "$timeout_merge_job" "${T8_TIMEOUT_MERGE_ROOT:-}" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema_version": "t8_hpc_hierarchical_slurm_inventory_v1",
    "status": "PASS",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "group_array_job_id": sys.argv[2],
    "final_merge_job_id": sys.argv[3],
    "package_job_id": sys.argv[4],
    "group_count": int(sys.argv[5]),
    "array_adoption_manifest": sys.argv[6],
    "dependencies": [
        {"job_id": sys.argv[3], "dependency": "afterok:" + sys.argv[2]},
        {"job_id": sys.argv[4], "dependency": "afterok:" + sys.argv[3]},
    ],
    "historical_held_jobs_released": False,
    "historical_held_jobs": [sys.argv[7], sys.argv[8]],
    "superseded_timeout_job_id": sys.argv[9],
    "superseded_timeout_root": sys.argv[10],
    "superseded_timeout_state": "SUPERSEDED_MERGE_TIMEOUT_NO_COMPLETE_RESULT",
    "matrix_write_enabled": False,
}
payload["inventory_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
fd, tmp = tempfile.mkstemp(prefix=".slurm-inventory.", dir=path.parent)
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
finally:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
PY

printf 'array_adoption_manifest=%s\n' "$array_adoption"
printf 'group_plan=%s\n' "$T8_HIERARCHICAL_GROUP_PLAN"
printf 'group_array_job_id=%s\n' "$group_job_id"
printf 'final_merge_job_id=%s\n' "$final_job_id"
printf 'package_job_id=%s\n' "$package_job_id"
printf 'package_root=%s\n' "$T8_HIERARCHICAL_PACKAGE_ROOT"
