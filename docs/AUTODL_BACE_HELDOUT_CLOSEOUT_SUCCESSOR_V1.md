# AutoDL BACE held-out closeout successor v1

## Purpose

This successor is the narrow recovery route for the September 2026 AutoDL
maintenance restart.  Both BACE `GlobalGCE` and `ComRecGC` already have
complete, calibration-only, top-20 frozen selections.  Their prior held-out
controller was interrupted before it published one valid test shard, so there
is no held-out checkpoint to resume and no reason to repeat generation,
training, calibration, or selection.

The successor therefore runs exactly, in this order for each method:

1. four fixed held-out test shards;
2. held-out merge;
3. final freeze against the existing selection;
4. standardized BACE export;
5. serialized append through the unique fast16 matrix authority.

`GlobalGCE` closes before `ComRecGC`.  All GPU stages run on one explicitly
selected physical GPU under the existing UUID advisory lock.  CPU stages keep
the same controller and output lineage.

## Frozen source gate

The runner requires the physical selection-adoption receipt and its exact
SHA-256.  It reopens every file in both receipt inventories, checks size and
SHA-256, requires no active writer, and verifies:

- `status=FROZEN` and `effective_rule_count=20`;
- the same frozen BACE GINE and MolCLR identities;
- the runtime `model.pt` bytes and held-out test path/SHA against that GINE
  bundle before any shard process is started;
- calibration loaded, held-out test not loaded;
- GNN oracle only, RF absent;
- selection was fitted on calibration and frozen before test.

The gate is repeated before each method and after both cells close.  A source
byte change fails closed.  Legal `GlobalGCE` zero-application batches remain
the explicit expected-empty result implemented by the held-out evaluator;
the successor does not synthesize candidates or nonzero coverage.

## Restart behavior

Every scientific stage writes to an immutable `attempt-N` directory.  On a
controller restart, a complete terminal is reopened and adopted; an incomplete
attempt is preserved and a fresh attempt is chosen.  The singleton owner lock
prevents a second BACE held-out writer.  `SIGTERM` is forwarded only to the
exact current child, never by name or process group.

The launcher performs the CPU-only source preflight first, then starts the
runner with `nohup setsid` and waits for a real heartbeat.  Use:

```bash
ALLOW_BACE_HELDOUT_CLOSEOUT_SUCCESSOR=1 \
RUN_GNN_ABLATION=0 \
BACE_HELDOUT_GPU_INDEX=0 \
scripts/autodl/launch_bace_heldout_closeout_successor_v1.sh
```

The launcher defaults are pinned to the audited September recovery roots.  A
different campaign must pass explicit source, receipt, model, split, matrix,
control, and output paths rather than editing the runner.

## Non-goals

This is not a general controller.  It cannot launch candidate generation,
training, calibration, selection, a GNN backbone ablation, or any non-BACE
task.  It does not reuse the interrupted held-out output, because that root
contains no complete terminal/checkpoint.
