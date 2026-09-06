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
