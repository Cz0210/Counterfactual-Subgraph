# Mut sealed chemistry startup control repair

The first evaluation activation exited before any science child. The common
stage runner read Mut's shared `stage_state.json`, containing successful
chemistry, as if it were a failed/running `unified_eval` checkpoint. Its strict
same-stage check correctly refused that caller bookkeeping mismatch.

`scripts/autodl/resume_mut_chemistry_startup.py` is a narrow control adapter for
this exact failure, not a new scientific runner or partial-evaluation recovery.
It validates the sealed continuation and its unchanged original checkout, then
calls that original runner with unchanged argv, imports, inputs and environment.
The original chemistry contract, invocation lease, startup barrier, fresh
resource admission, RF/WNode evaluation, full gate and freeze remain in force.

Only startup bookkeeping moves to a fresh control root, with independent
`stage_checkpoints/{unified_eval,full_gate,freeze}.json` files. The exact prior
`FAILED.json` diagnostic destination is redirected there on future exceptions.
Original chemistry, its root stage-state, boundary and failure remain intact;
byte-exact copies of the three small historical records are also preserved.
The repair receipt separately binds the actual new control-driver commit and
the original scientific commit. Gate/chemistry commit fields remain the honest
original scientific commit; no Git identity is overridden.

Run with the sealed spec's same Python executable:

```sh
python -I -B /absolute/new-checkout/scripts/autodl/resume_mut_chemistry_startup.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --continuation-spec /absolute/control/evaluation_continuation.json \
  --recovery-root /absolute/control/fresh-startup-repair \
  --expected-driver-commit ACTUAL_NEW_CONTROL_COMMIT
```

This does not dispatch exports/publication, clear reservations, grant a new
science retry, duplicate an existing evaluation directory or claim final PASS.
The matching Slurm script is intentionally a static AutoDL-only refusal.

## Publishing the completed repaired run

The original `FAILED.json` remains an honest historical diagnostic after the
three later stages finish. Default terminal validation continues to reject it.
For this one pinned `94300ea34724088f3336fab95aba5ea5855f58e3` startup adapter,
export and publication accept an explicit `--startup-repair-receipt` pointing
to its existing `runtime/control_adapter_receipt.json`. The flag is propagated
through export reopening, canonical publication and the shared matrix append;
there is no inferred locator, environment override or blanket sentinel bypass.

The read-only `mut_pre_science_startup_failure_supersession_v1` proof requires
byte-identical historical failure, chemistry state and sealed boundary; the
exact historical ValueError; the original adapter code hash; unchanged
`fbefa4caff172453d42afd90f8518bc7e8bddf47` scientific checkout; three ordered,
original-argv-bound PASS checkpoints with their barrier/marker hashes; later
matching final manifests; dead recorded owners/children; and no writable FD
under either the scientific or repair-control root. Additional failures or
missing/mismatched evidence reject publication. Only small metadata/code files
are hashed; science payloads, checkpoints and caches are not opened here.

The ordinary independent scientific terminal validation still runs in full.
Fresh export and matrix-append receipts retain the typed supersession and its
small-file inventory as evidence, without rewriting the scientific root or
historical failure. Execute export/publish using the actual new publication
checkout and bind that new commit in the canonical publisher registry. The
science manifests and full-gate expected commit remain the original `fbefa4ca`;
do not claim the new publication driver produced the scientific results.
