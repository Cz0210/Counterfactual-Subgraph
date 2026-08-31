# AutoDL BACE ComRecGC 20k/25k resource-cap executor

This route closes only the already-running BACE ComRecGC trajectory under the
authorized deadline policy.  It is not a new scheduler and does not change the
frozen GINE, NeuroSED, dataset, RNG, action trace, calibration split, or held-out
test contract.

## Stop policy

The read-only observer remains the sole cap-decision input:

- before 20,000, only a preregistered convergence PASS may request handover;
- at the first committed checkpoint at or above 20,000, lineage errors must be
  zero and valid unique rules must be at least ten;
- if that gate fails, continue only to the first committed checkpoint at or
  above 25,000;
- if 25,000 still has fewer than ten valid unique rules, precisely stop the
  exact worker and record terminal scientific failure without materialization,
  calibration, test, export, or matrix adoption;
- calibration and test are absent from every stop decision.

A stale `progress.json` is insufficient stall evidence.  Any positive CPU-time,
output-byte, progress-step, checkpoint-step, or checkpoint-write observation is
`RUNNING_SLOW` and receives no signal.  `STALLED` needs at least one full hour
with every science indicator unchanged and no fsync/rename evidence.

## Exact handover

`scripts/autodl/run_bace_comrecgc_resource_cap_executor.py` requires the frozen
PID, Linux start ticks, raw cmdline SHA-256, cwd, output root, controller ID, and
controller receipt identity.  It reopens all of them after writing the
checkpoint-bound cap receipt and immediately before calling `SIGTERM` on the
exact PID.  It contains no SIGKILL, process-group, `pkill`, `killall`, or fuzzy
process match.  A 300-second timeout is terminal `SIGTERM_TIMEOUT`; it does not
escalate.

## Checkpoint finalization

The selected checkpoint must be a fully completed-step v2 checkpoint whose
digest and step equal the observer request.  Its original 50k provenance remains
unchanged.  The finalizer:

1. validates and loads that exact checkpoint directory;
2. copies only hash-bound trace chunks into a fresh trace root;
3. restores the checkpoint trace state and flushes its pending events;
4. rebuilds the complete frozen payload against the checkpoint SQLite snapshot;
5. writes fresh `counterfactuals.pt`, manifest, progress, cap receipt, and
   `_RUN_COMPLETE.json` without running step `M_effective + 1`;
6. emits a postprocess task fragment for common recourse, candidate freeze,
   four calibration shards, calibration-only selection, four held-out test
   shards, final freeze, and BACE cell standardization.

The manifest records both the original configured 50k trajectory and the
authorized main-result fields.  ComRecGC may freeze R=10..20 rules.  Prefixes
through K=20 use the unchanged R-rule result for K>R; no rule is duplicated.

## Launch boundary

Use the paired AutoDL wrapper with
`ALLOW_BACE_COMRECGC_20K_EXECUTOR=1` and `RUN_GNN_ABLATION=0`.  Every path and
process identity is supplied explicitly through environment variables.  Start
the executor before 20k if desired; while no eligible handover request exists it
writes only `WAITING_RESOURCE_CAP_REQUEST` heartbeat/state files.

The emitted postprocess fragment must be appended to the existing minimal
continuation controller.  The executor itself does not allocate a GPU or infer
matrix PASS from its own heartbeat.

## Postprocess queue preparation

`POSTPROCESS_QUEUE_READY` is not itself an executable controller manifest.
The executor deliberately emits the native BACE task schema and omits the
already materialized generation task.  Prepare a fresh generic fragment and
controller manifest with:

```bash
python scripts/autodl/prepare_bace_comrecgc_resource_cap_postprocess.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --source-fragment /absolute/executor/postprocess.tasks.json \
  --generic-fragment-output /absolute/fresh/postprocess.generic.tasks.json \
  --manifest-output /absolute/fresh/postprocess.manifest.json \
  --controller-id bace-comrecgc-cap-postprocess-UUID
```

The preparer reopens the cap receipt and the exact materialized generation
manifest, preserves that generation root as one immutable external input, and
still rewrites every mutable cache and downstream output into controller-owned
attempt roots.  It also includes the final BACE frozen-cell standardization
task.  Only the resulting manifest may be passed to
`scripts/autodl/launch_four_by_four.sh`; the native fragment must never be
launched directly or edited by hand.
