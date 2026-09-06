# Same-stage T14 GPU-mask bootstrap correction

The actual 4e524bdc owner325558 failed on2026-09-06T03:00:12Z before creating
its science directory. `gpu_lock.py`'s generic run branch emitted the GPU UUID,
but `require_gpu_runtime` required numeric `CUDA_VISIBLE_DEVICES=2`. The prior
d9297578 generic branch emitted the numeric index. This mismatch was introduced
by41c36f535, independently of the storage repair. It is not a new graph-store
failure, CUDA OOM or science checkpoint failure.

The fix restores the original generic branch only. Dedicated LLM dispatch still
uses its explicit UUID/FD binding. Physical UUID/index identity and the existing
GPU lease are not relaxed. Tiny tests call the real producer and real strict
consumer with only the CUDA availability functions mocked.

## Exactly one same-stage bootstrap correction

The original retry2/stage replacement receipt stays immutable. The already
allocated continuous UUIDf7c88839 and reload UUID22b41c6b remain unchanged,
as do their output roots. Before creating the new binding, both science roots
must be absent, old owner and exact writers absent, and the precise GPU-mask
failure present in the bounded log tail. There is no checkpoint or RNG state
to resume, and no reference500 regeneration.

`--prepare-bootstrap-rebind` creates new spec files with only execution commit,
wrapper/owner paths and their derived hashes changed. The original old specs
remain intact. The loader validates the original replacement against its actual
old clean worktree and the new specs against the new clean driver. It does not
weaken or skip either commit check.

Use a new small user-authorization file:

```json
{
  "schema_version": "t14_same_stage_bootstrap_user_authorization_v1",
  "authorized_by": "user_project_owner",
  "retry_index": 2,
  "stage_replacement_count": 1,
  "bootstrap_correction_index": 1,
  "max_bootstrap_corrections": 1,
  "same_stage_uuid_required": true,
  "require_science_output_absent": true,
  "checkpoint_resume": false,
  "retry3_allowed": false,
  "reference_rerun_allowed": false,
  "formal_execution_rebind_required": true,
  "original_replacement": {"path": "<old-owner>/stage_replacement_retry2/receipt.json", "sha256": "<file SHA>"},
  "failed_terminal": {"path": "<old-owner>/terminal.json", "sha256": "<file SHA>"},
  "driver_commit": "<new final execution commit>"
}
```

Prepare once, through the same owner lock:

```bash
python -I -B <new-tree>/scripts/autodl/run_t14_route_c_owner.py \
 --config <new-tree>/configs/hpc.yaml --task-spec <unchanged-master> \
 --continuation-spec <unchanged-continuation> \
 --failed-stage-replacement <old-owner>/stage_replacement_retry2/receipt.json \
 --prepare-bootstrap-rebind <old-owner>/stage_replacement_retry2/bootstrap_rebind/receipt.json \
 --bootstrap-authorization <new-authorization>
```

Then launch the same retry2 owner with the same two original specs/roots:

```bash
python -I -B <new-tree>/scripts/autodl/run_t14_route_c_owner.py \
 --config <new-tree>/configs/hpc.yaml --task-spec <unchanged-master> \
 --continuation-spec <unchanged-continuation> \
 --failed-stage-replacement <old-owner>/stage_replacement_retry2/receipt.json \
 --bootstrap-rebind <old-owner>/stage_replacement_retry2/bootstrap_rebind/receipt.json
```

A second bootstrap prepare is rejected. A further failure is reported, not
automatically repaired/retried. A created science root cannot be relabeled by
this mechanism. The old master/authorization/cadence three-field formal hold
still applies after actual exact continuous/reload parity; no formal start is
authorized by successful bootstrap alone.
