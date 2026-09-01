# AutoDL Mutagenicity ComRecGC fast-accurate successor

This campaign-specific successor replaces the unmeasured 440 GiB launch gate
without changing the frozen Mutagenicity science or creating another matrix
authority.

## Scientific route

The preferred route reopens the completed historical 50,000-step generation
read-only.  That source is a trace-enabled artifact; it must never be described
as a trace-off replay or as a successful full trace-on/off parity experiment.
The current launcher deliberately sets
`allow_trace_on_historical_adoption=false`: the frozen automatic Route-A
contract requires `trace_enabled=false`, so the successor can run the
authorized 500-step diagnostic but cannot standardize or publish this
trace-enabled source without a separate scientific decision.  If that decision
is later granted, adoption must still satisfy all of the following:

1. the pinned 500-step legacy/checkpointed behavioral-equivalence gate passes;
2. checkpoint interruption, reload, action trace, RNG, candidate payload, and
   lineage semantics agree;
3. the generation payload SHA-256 is reopened, the pair-store's real
   strict-flip `candidate_graph_hashes_sha256` is used as the candidate
   universe, and DBSCAN is bound through the exact pair-vector path/SHA;
4. all generation and common-recourse roots have no live writer; and
5. the independent terminal verifier accepts the new, truthful historical
   adoption schema.

The receipt records `historical_source_trace_enabled=true`,
`trace_parity_passed=false`, `traceoff_reference_rerun=false`, and
`full_50k_rerun_performed=false`.  This is intentionally distinct from the
older parity-v2 terminal.

The 14 historical selected-event target-parent metadata mismatches are not
silently ignored.  The audit records those event rows separately from the one
unique cross-parent predecessor convergence; recorded-action replay
mismatches, unresolved conflicts, and selected-predecessor parent mismatches
must all remain zero.

## Candidate-universe binding

The historical DBSCAN manifest predates a field literally named
`candidate_universe_sha`.  The successor therefore records an explicit
transitive binding instead of inventing a native DBSCAN claim:

- exact generation `counterfactuals.pt` SHA-256;
- the pair-store's strict-flip candidate graph-universe SHA-256;
- DBSCAN input-vector path/SHA and pair-store scientific-identity bindings.

Pair-store and DBSCAN are reused only when the complete chain reopens.  They
are never recomputed by this route.  Any mismatch fails closed.

## Train-side convergence fallback

The shared Mut evidence helper freezes the authorized Route-B policy without
reading calibration or test data: begin checking at step 20,000, check only
committed/reloadable checkpoints every 2,500 steps, and stop only after two
consecutive windows meet all six preregistered stability/rule/lineage gates.
The historical-adoption successor does not claim this early stop; its adopted
`M_EFFECTIVE` remains the completed 50,000-step source budget.

## Memory admission

`MUT_REQUIRED_FREE_MEMORY_GIB=440` is retained only as superseded historical
evidence.  Generic evidence helpers understand cgroup v1 and v2, but the
audited AutoDL container exposes a read-only cgroup-v1 namespace and no live
systemd manager.  This deployed successor therefore freezes that exact
no-child limitation and requires at least 96 GiB parent headroom.

The 500-step process is sampled every ten seconds.  The empirical full limit
is:

```text
full_max = clamp(3 * canary_peak + 16 GiB, 48 GiB, 128 GiB)
full_high = floor(0.75 * full_max)
required_parent_headroom = full_max + 16 GiB
```

OOM/event deltas or sustained pressure stop only the freshly isolated Mut
session with SIGTERM; already-committed mirrored checkpoints remain the resume
authority.  Unexpected monitor failures apply the same exact-session cleanup,
so generation grandchildren cannot be left orphaned.  The watchdog never
sends SIGKILL, drops global caches, or signals AIDS, BACE, TasteMolNet, or
publisher processes.  When no protected-task throughput baseline is available,
the memory receipt says so and does not claim that gate was measured.

## Queue and matrix publication

The launcher creates one fresh successor with a durable heartbeat.  It reuses
the shared GPU UUID locks and the existing
`fast16_matrix_authority`; it does not create a second authority.  The old
pure waiter may be terminated by exact PID/start-ticks/command verification
only after the successor heartbeat is visible and no duplicate Mut science
child exists.

`RUN_GNN_ABLATION=0` remains mandatory until the matrix reaches 16/16.
