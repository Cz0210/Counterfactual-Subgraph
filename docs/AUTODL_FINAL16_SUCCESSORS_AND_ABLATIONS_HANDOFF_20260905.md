# Final16 successors and ablation gates — 2026-09-05

## Authority and current matrix

- Matrix authority: `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`
- Last observed count: 12/16.
- Missing cells: Mutagenicity/ComRecGC and TasteMolNet/{GCFExplainer, GlobalGCE, ComRecGC}.
- AutoDL remains the only matrix authority.  HPC has no matrix, GINE,
  calibration, or test permission.

## Live science and successor state

- Mut trace-on: owner `193161`, worker `193180`, continuation `193450`, GPU0.
- T12 reference: owner `162844`, worker `173495`, GPU3; sealed step-250 is the
  diagnostic fork authority.
- T12 accelerated diagnostic: owner `218411`, science `218782`, GPU1, fresh
  output root.  Both T12 arms were still in the 250-to-500 diagnostic phase at
  the latest observation.  Their checkpoints are not production-20k resume
  authorities (`promotion_allowed=false`).
- T14's first authorized fresh Route-C retry is terminal failed.  It reached
  the first step-50 boundary and failed closed because the full-parameter
  cadence drifted.  The failed root is preserved; there is no current T14
  science owner and GPU2 is free.
- Legacy AutoDL T8 remains a fallback and has not been signalled.

All PIDs above are observations, not restart or signalling authority.  Reopen
`/proc/<pid>/stat`, command, root, and heartbeat before any process action.

## HPC T8 chain

- Sixteen source shards from array `2536781` are adopted and must not rerun.
- Old jobs `2536786` and `2536787` remain `JobHeldUser`.
- Timed-out merge `2538830` is preserved as evidence.
- Fresh group array: `2541889`; all four deterministic groups completed with
  exit code `0:0`.
- Fresh final merge: `2541890`, `afterok:2541889`; RUNNING on `cpui034` at the
  latest read-only snapshot (`2026-09-05 03:23:32 CST`).
- Fresh package: `2541891`, PENDING on `afterok:2541890`.
- Root: `/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/continuations/stress-2535373-e8be657a-2223-476b-94a3-fd14997e48ad/hierarchical-08a63955-20260904T181200Z`.
- Path-specific free space was `89,830,723,584` bytes (about `83.66 GiB`) at
  that snapshot.

The Mac relay is intentionally not running.  Desktop auto-review rejected an
unattended background transfer of the completed HPC package to the external
disk and AutoDL.  Do not work around that review.  A fresh, informed user
authorization is required before starting the relay.

## Predeployed continuation code

- Canonical owner registry: `scripts/autodl/reconcile_final16_owner_registry_v1.py`.
- Read-only closeout observer: `scripts/autodl/launch_final16_successors_v1.sh`.
- T8 import/T13 specs: `scripts/autodl/build_t8_hpc_t13_successor_specs_v1.py`.
- T8 import owner: `scripts/autodl/launch_t8_hpc_import_owner_v1.sh`.
- T13 owner: `scripts/autodl/launch_t13_from_hpc_owner_v1.sh`.
- LLM owner-bound snapshot: `scripts/autodl/build_llm_early_launch_snapshot_v1.py`.

The concrete Mut executor is already deployed and waiting, without restarting
the active A/B run:

- Execution commit: `ebada9c0117e57d2babecaebe18f393fd26e76fb`.
- Immutable worktree: `/root/autodl-tmp/worktrees/final16-successors-ebada9c0`.
- Executor PID: `222378` (observed start ticks `28623524`).
- Control root: `/autodl-fs/data/counterfactual-subgraph-runtime/control/mut_next_stage_executor_ebada9c0_20260904T195542Z`.
- State: `WAITING_FOR_NEXT_ACTION`.

The canonical owner registry is:

`/autodl-fs/data/counterfactual-subgraph-runtime/control/final16-owner-registry/current.json`

Its registry id is `final16-canonical-ebada9c0-20260904T200000Z`; registry
self-SHA is `cb479fcd94ec0db7e3a56d7a1ee2b1a55a83a52f262549b5157e9dc824bf8123`.
The read-only closeout observer is PID `222653`, with heartbeat under
`/autodl-fs/data/counterfactual-subgraph-runtime/control/final16-successors-observer-ebada9c0-20260904T200200Z/heartbeat.json`.

The T8 import owner must remain `WAITING_HPC_PACKAGE` until the relay publishes
the exact ready marker.  The T13 owner must not take a GPU before import PASS.

## Ablation gates

- LLM: blocked at 12/16.  It additionally requires registered Mut PASS, healthy
  owners and unique publishers for every incomplete cell, no main GPU waiter,
  and one GPU idle for at least 1200 seconds.  At most one early LLM GPU.
- GNN: blocked until 16/16 plus final Figure 3, Figure 4, Table 2, and a
  hash-closed combined audit.
- No ablation science is running.

## Decisions still requiring explicit authorization

No new algorithm choice is required.  Three operational authorizations remain:

1. T14: allow one engineering-corrected second fresh retry, with
   `ALLOW_T14_ENGINEERING_CORRECTED_SECOND_FRESH_RETRY=1` and
   `T14_ROUTE_C_FRESH_RETRY_MAX_ATTEMPTS=2`.  It must use a fresh UUID/root and
   must not resume the failed attempt.
2. T12: allow a fresh accelerated production run from step zero, with
   `ALLOW_T12_FRESH_ACCELERATED_PRODUCTION_FULL=1`,
   `ALLOW_T12_EXISTING_PUBLISHER_LOCATOR_HANDOFF=1`, and
   `ALLOW_T12_DIAGNOSTIC_CHECKPOINT_PROMOTION=0`.  The existing diagnostic
   checkpoints remain parity evidence only.
3. Mac relay: after the desktop auto-review rejection, give fresh informed
   permission to start the scoped background `caffeinate` relay.  It will only
   copy the completed hash-closed HPC package through the external disk to a
   fresh AutoDL import root; it will not delete files or write the matrix.

## Read-only status commands

```bash
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
ssh tongji-hpc 'squeue -j 2541889,2541890,2541891,2536786,2536787'
```

Do not poll an unchanged hour-scale job repeatedly.  Once all owners and
successors are durable, return and check again after the next natural stage
boundary.
