# T14 retry2: replace one failed empty canary stage

## Evidence and scope

The existing retry2 owner `route-c-59f101cd-f30b-458d-aa8c-2eb93ae82609`
completed its reference500 ledger/checkpoint boundary. Its continuous low-memory
stage exited1 before step1 (`completed_step=0`, empty checkpoint directory).
The old owner resumes a completed510 boundary but cannot resume this empty
failed stage. Re-entering its old plan therefore fails immediately.

The user authorized existing resource/dispatch binding repair and continuation
of existing main successors, not retry3. The narrow new option creates only two
fresh low-memory canary specs inside the same retry2 owner. The reference500
spec, failed root, master, original owner plan, cadence and authorization remain
byte-identical. The existing owner lock, GPU2 wrapper, resource watchdog, exact
parity and independent reload are retained. No large reference/forbidden legacy
payload is deserialized for this binding; reference metadata plus the existing
scientific ledger are read. Metadata validation is not a new payload-reload PASS.

## Preparation

Use the final clean integrated worktree containing both the graph-store repair
and this owner patch. Write one fresh user-authorization JSON with these fields:

```json
{
  "schema_version": "t14_retry2_failed_stage_replacement_authorization_v1",
  "authorized_by": "user_project_owner",
  "same_retry_index": 2,
  "max_stage_replacements": 1,
  "stage": "LOW_MEMORY_CONTINUOUS_510",
  "allow_storage_only_driver_repair": true,
  "reference_rerun_allowed": false,
  "old_failed_root_mutation_allowed": false,
  "retry3_allowed": false,
  "formal_execution_rebind_required": true,
  "master_spec": {"path": "<unchanged absolute master>", "sha256": "<file SHA>"},
  "failed_terminal": {"path": "<old owner>/terminal.json", "sha256": "<file SHA>"},
  "driver_commit": "<final integrated commit>"
}
```

Then run the existing owner in preparation-only mode:

```bash
python -I -B <new-worktree>/scripts/autodl/run_t14_route_c_owner.py \
  --config <new-worktree>/configs/hpc.yaml \
  --task-spec <unchanged-master> --continuation-spec <unchanged-continuation> \
  --prepare-failed-stage-replacement <old-owner>/stage_replacement_retry2/receipt.json \
  --replacement-authorization <new-authorization.json>
```

This acquires the old owner lock, verifies old owner/science absence, step0 and
empty checkpoints, preserves the complete reference ledger, and creates the
one fixed replacement receipt. It does not launch science. A second replacement
preparation is rejected; an interrupted preparation is not silently retried into
another root. No normal owner evidence is overwritten during preparation failure.

## Existing-owner execution

```bash
python -I -B <new-worktree>/scripts/autodl/run_t14_route_c_owner.py \
  --config <new-worktree>/configs/hpc.yaml \
  --task-spec <unchanged-master> --continuation-spec <unchanged-continuation> \
  --failed-stage-replacement <old-owner>/stage_replacement_retry2/receipt.json
```

The original failed terminal is retained in the owner's existing terminal
history. The old plan is read, not rewritten. Exactly these science stages run:

1. Fresh corrected low-memory continuous1–510.
2. Fresh corrected low-memory1–250; independent checkpoint reload/promotion.
3. Resume that same250 checkpoint to510.

The original reference500 is never submitted again. Both new ledger comparisons
retain the original exact gate. Failure stops this owner; there is no automatic
stage repetition or retry3. Successful original boundaries are reused on explicit
same-receipt owner continuation, not generated again.

## Formal execution remains explicitly blocked

The original retry2 master binds science commit `d9297578...`; its signed original
authorization and formal cadence independently bind the same commit. Replacing
only a wrapper would fail the full runner's actual-execution check or incorrectly
label new code as old science. This implementation does neither.

After both new canaries and all three exact ledger comparisons pass, write:

`stage_replacement_retry2/canary_verification.json`

with `CANARY_PARITY_PASS_FORMAL_EXECUTION_REBIND_REQUIRED`, and return75 before
formal generation. The receipt names the original science commit and actual
storage-repair driver, keeps retry_index2, and identifies the three exact fields:

- `master.execution_commit`
- `fresh_retry.authorization_receipt.corrected_execution_commit`
- `fresh_retry.formal_cadence_contract.execution_commit`

A subsequent narrow same-attempt runtime rebind must preserve these original
documents and the full scientific contract. Until that binding is accepted by
the existing strict validator, do not claim unattended full continuation is ready.
No budget, split, GINE, RNG, checkpoint cadence, selector or publisher is changed.

The paired Slurm script accepts `T14_FAILED_STAGE_REPLACEMENT`; this is only CLI
synchronization. Actual production remains on AutoDL's existing GPU2 reservation,
not a newly submitted HPC workload.

## Focused tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/autodl/test_t14_failed_stage_replacement.py \
  tests/autodl/test_t14_route_c_fresh.py
```

36 tests pass locally, including a tiny mocked full owner control-flow test that
proves no reference submission, exactly the three low-memory launch/resume calls,
and no formal run before the typed execution-rebind hold. These are engineering
tests, not real GPU parity or science PASS.
