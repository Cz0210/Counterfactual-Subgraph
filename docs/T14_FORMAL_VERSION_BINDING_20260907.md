# T14 retry2: formal execution identity binding

This is a narrow adapter for the existing Route-C owner and continuation. It
does not start another canary, create retry3, alter the matrix publisher, or
modify the current 39a97e41 execution worktree. Current continuous/reload
canaries retain their own roots and identities; reference500 is never rerun.

## Why the three pins are not mechanically replaced

| Field | Existing role | New binding |
|---|---|---|
| `master.execution_commit` | Actual formal science/checkpoint provenance | Fresh task spec records the deployed driver SHA |
| `fresh_retry.authorization_receipt.corrected_execution_commit` | Retry2 authority, linked to preserved retirement/config evidence | Old receipt remains unchanged; explicit current user authorization binds its source master to the new execution |
| `fresh_retry.formal_cadence_contract.execution_commit` | Source-pinned checkpoint/watchdog/convergence/publisher cadence | Same cadence contract retained, with an explicit new runtime source inventory |

The optional `formal_runtime_binding` is accepted only after reopening the
original unmodified master through its original complete validator, exact
comparison of all non-runtime fields, and exact current user authorization.
It is not a generic PASS exception. Sources used by the current canary are
AST-equal to formal sources; only `validate_spec` is excluded in the Route-C
module because it is the field-validation adapter changed here. Every
ComRecGC module and the full runner/oracle entry are included. Any changed
algorithm AST rejects adoption and requires the affected scientific parity.

## Actual entrypoints

`scripts/autodl/prepare_t14_formal_binding.py --help` documents the required
physical paths. It creates exactly one `formal_execution_binding` under the
existing owner namespace: binding, formal task spec, continuation spec, and
dispatch JSON. It does not acquire the active owner or GPU lock, edit the old
plan, write active roots, signal a process, or submit science.

The sealed dispatch calls the existing owner with `--formal-binding`. Before
writing owner state it requires the complete original three-way parity receipt,
same bootstrap/driver binding, prior owner's natural exit, and no exact
scientific writer. The existing `owner.lock` remains the final exclusive gate.
It then rechecks only the short step ledgers and enters the shared existing
formal implementation. No second waiting owner is started by preparation.

## Diagnostic and production boundary

The continuous510 and reload510 roots are diagnostic evidence, not production
checkpoints. The old master already reserved an as-yet-unstarted
`PROMOTABLE_LOW_MEMORY` branch. This same retry2 UUID/output namespace starts
from zero, seals/reloads50/100/250/500 using the original cadence, compares
against the preserved continuous1–500 ledger, then promotes only its own
legally typed500 checkpoint. The original20k/25k policy, training cohort,
classifier, convergence checks and single publication chain are unchanged.

Until original canary parity and resource admission pass, formal state is
`SEALED_WAITING_EXISTING_CANARY`, not RUNNING or scientific PASS. A current
non-reloadable owner will exit with its typed formal hold; only after natural
exit may the sealed command be started in the same namespace.

## Verification

Focused CPU tests cover source-AST equality, exact spec and authorization
bindings, source receipts unchanged, live owner/orphan rejection, incomplete
parity refusal, diagnostic/non-promotion boundary and shared formal function.
No GPU/model/checkpoint/OT is loaded in these tests. The paired preparation
Slurm script is CPU-only under this task's explicit exception; the existing
owner Slurm script forwards the new binding argument without restarting anyone.
