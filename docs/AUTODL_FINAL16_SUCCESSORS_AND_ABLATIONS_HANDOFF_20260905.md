# Final16 successors and ablation gates — 2026-09-05

## Live continuation update — 2026-09-05 17:11 CST

This section supersedes the older observations below.  The canonical matrix is
still 12/16: the missing cells are Mutagenicity/ComRecGC and
TasteMolNet/{GCFExplainer, GlobalGCE, ComRecGC}.  No LLM or GNN ablation
science is running.

- T8/T13: scoped relay attempt `f70b55c0-8b4f-4473-b16b-6fb044da766b`
  completed at 16:08 CST.  The immutable HPC package is preserved on the Mac
  external disk and in the exact AutoDL relay root; both copies bind
  `6,103,923,589` bytes and SHA-256
  `06702fdc97ae2bb3661855497a336d19c6ceb33fd53f2304f41471781629346e`.
  The original import owner `219867` was found to have the old in-memory
  v1-only marker verifier.  After exact PID/start-ticks/command/root checks it
  received SIGTERM (never SIGKILL), exited, and was replaced once by reviewed
  v2-aware import owner `272454` from commit
  `d1e7ab9dc2eabe9993f04c82ff7603609f9adef5`.  That owner was still CPU-bound
  in the first deep streaming verification at 17:08 CST; no import PASS or
  release marker existed yet.  T13 owner `219876` remains healthy in
  `WAITING_HPC_IMPORT_PASS`, owns no physical GPU process, and will consume the
  same canonical release path.  Neither HPC nor the relay/import worker may
  write the matrix.  At 17:12 CST the registry was CAS-reconciled under the
  matrix publication lock: the proven-dead v1 owner row is terminal, the live
  heartbeat-backed T13 owner remains canonical, and the v2 one-shot is recorded
  only in the handoff receipt until it emits its own heartbeat.  The resulting
  registry file SHA-256 is
  `17d3c4632e31128a1181b1dd8dcb17f27984c7ef3c18da694b3efc200beee702`
  and its self-hash is
  `ee742c0fba895b893651056bdba1d2c49e54406c29a437550cf461ed9e37ac59`.
- T14: the user-authorized second and final engineering-corrected retry is now
  live as fresh attempt `59f101cd-f30b-458d-aa8c-2eb93ae82609`, using commit
  `d92975786391f9e211950b78456da07a730787f2`.  Owner PID `268102` and science
  PID `268217` (active child `268321`) own GPU2.  The live state digest had
  reached step 93 at 16:55 CST; the sealed step-50 checkpoint is complete and
  proves that retry 2 passed retry 1's exact cadence-drift failure boundary.
  Independent reload remains pending until reference step 500.  The failed attempt and its
  ledger remain preserved; no failed checkpoint or legacy step-161 state was
  reused.  Its owner/lease rows survived the later registry CAS unchanged.
- T12: the accelerated step-510 engineering failure has a diagnostic-only
  reconciliation overlay at
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/repairs/t12-diagnostic-510-reconcile-24203990-5d23-4ce2-a2f4-c998efa3aa00`.
  It is not promotable.  Reference owner `162844` / science PID `173495`
  remains at the sealed step-250 boundary on GPU3 and must not be restarted
  while its I/O/process evidence remains live.  Formal production remains
  blocked on full per-step 251--500 and 501--510 reference/accelerated parity.
  Commit `25a0475e3c3f3526649cc9389c382090fc7ba208` adds a strictly
  non-dispatchable fresh-zero plan that can bind the canonical future locator
  without creating it and validates the real sealed GINE checkpoint directory
  using the existing T3 content-tree contract.  The first build request failed
  before that validation because the CLI rejected the conda `python` symlink;
  it changed no science.  A corrected low-priority builder using the resolved
  `python3.10` executable completed under tag
  `20260905T091100Z-5cefc663-4d2f-4d06-91f7-566b997a73f1`.  The resulting plan
  is `BLOCKED_WAITING_DIAGNOSTIC_PARITY`, contains all 12 formal generation
  checkpoint boundaries plus generation/postprocess/terminal verification,
  and records `science_started=false`, `gpu_lease_acquired=false`, and
  `dispatchable=false` for every stage.  Plan SHA-256 is
  `700e0cf9127b942178465ce51c0b5eb2bdf8898bf83dc50d383350f7582f4222`.
  Its canonical path is
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/t12_fresh_zero_predeployment/20260905T091100Z-5cefc663-4d2f-4d06-91f7-566b997a73f1/fresh_zero_plan.json`.
  It does not acquire a GPU lease and cannot dispatch production.  Because it
  intentionally snapshots the registry that existed at build start, a future
  parity-qualified dispatcher must rebind the then-current registry under the
  existing publisher handoff; this does not weaken or delay today's parity
  gate.
- Mut: owner `193161`, active trace-on child `193180`, post-A/B continuation
  `193450`, and final executor `222378` remain live.  The last sealed progress
  observation was step 225 at 16:49 CST; do not restart or duplicate this run.  The normal
  equivalence-to-adoption-to-publication successor is already deployed.

The current long-running stages require no new scientific authorization.
T14 retry-2, T12 diagnostic repair/fresh-zero planning, and the scoped T8 relay
have already been authorized.  T12 production is an evidence gate, not an
authorization gate.

The previous read-only closeout observer could not parse the newly authorized
terminal-engineering registry state and reported `BLOCKED_EVIDENCE`.  After
exact identity checks it was gracefully replaced (SIGTERM only) by the same
read-only implementation from commit `25a0475e`.  Successor controller
`final16-successors-a9206944-989c-4638-8eea-578113a10c96`, PID `274368`, has a
healthy `RUNNING_LONG_EXPERIMENTS` heartbeat at
`/autodl-fs/data/counterfactual-subgraph-runtime/control/final16-successors-observer-25a0475e-20260905T091800Z-7012ad2f/heartbeat.json`.
Its snapshot reports no stale owners/publishers, starts no science or publisher,
takes no GPU lock, sends no science signal, and writes no matrix data.

Read-only status commands:

```bash
bash scripts/local/status_t8_scoped_relay_v2.sh
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/final16-owner-registry/current.json'
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/t14_route_c/current/owner.pid'
```

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

## Authorization state

No new algorithm or operational authorization is pending.  The T14 second
fresh retry, T12 diagnostic reconciliation/fresh-from-zero production policy,
and scoped Mac relay are all explicitly authorized.  Their remaining gates are
runtime or scientific evidence: T8 must finish transport/import verification;
T12 must produce complete per-step parity; T14 must reach its sealed checkpoints
and pass the existing resource/convergence policy.  None of these gates may be
converted into a PASS by another user decision.

## Read-only status commands

```bash
ssh autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
ssh tongji-hpc 'squeue -j 2541889,2541890,2541891,2536786,2536787'
```

Do not poll an unchanged hour-scale job repeatedly.  Once all owners and
successors are durable, return and check again after the next natural stage
boundary.
