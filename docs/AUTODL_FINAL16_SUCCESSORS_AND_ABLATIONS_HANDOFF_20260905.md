# Final16 successors and ablation gates — 2026-09-05

## Authority and current matrix

- Matrix authority: `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`
- Last observed count: 12/16.
- Missing cells: Mutagenicity/ComRecGC and TasteMolNet/{GCFExplainer, GlobalGCE, ComRecGC}.
- AutoDL remains the only matrix authority.  HPC has no matrix, GINE,
  calibration, or test permission.

## Live science protected from restart

- Mut trace-on: owner `193161`, worker `193180`, continuation `193450`, GPU0.
- T12 reference: owner `162844`, worker `173495`, GPU3; sealed step-250 is the
  fork authority.
- T12 accelerated: owner `218411`, science `218782`, GPU1, fresh output root.
- T14 fresh Route-C retry: owner `217867`, current science `218069`, GPU2.
- Legacy AutoDL T8 remains a fallback and has not been signalled.

All PIDs above are observations, not restart or signalling authority.  Reopen
`/proc/<pid>/stat`, command, root, and heartbeat before any process action.

## HPC T8 chain

- Sixteen source shards from array `2536781` are adopted and must not rerun.
- Old jobs `2536786` and `2536787` remain `JobHeldUser`.
- Timed-out merge `2538830` is preserved as evidence.
- Fresh group array: `2541889` (four deterministic groups).
- Fresh final merge: `2541890`, `afterok:2541889`.
- Fresh package: `2541891`, `afterok:2541890`.
- Root: `/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/continuations/stress-2535373-e8be657a-2223-476b-94a3-fd14997e48ad/hierarchical-08a63955-20260904T181200Z`.

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

The T8 import owner must remain `WAITING_HPC_PACKAGE` until the relay publishes
the exact ready marker.  The T13 owner must not take a GPU before import PASS.

## Ablation gates

- LLM: blocked at 12/16.  It additionally requires registered Mut PASS, healthy
  owners and unique publishers for every incomplete cell, no main GPU waiter,
  and one GPU idle for at least 1200 seconds.  At most one early LLM GPU.
- GNN: blocked until 16/16 plus final Figure 3, Figure 4, Table 2, and a
  hash-closed combined audit.
- No ablation science is running.

## Read-only status commands

```bash
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
ssh tongji-hpc 'squeue -j 2541889,2541890,2541891,2536786,2536787'
```

Do not poll an unchanged hour-scale job repeatedly.  Once all owners and
successors are durable, return and check again after the next natural stage
boundary.
