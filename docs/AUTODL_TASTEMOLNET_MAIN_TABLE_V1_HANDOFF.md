# AutoDL TasteMolNet main-table v1 handoff

## Scope and truth boundary

This route supersedes the historical
`TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW` execution decision without
modifying its manifest, state, or gate. Policy v2 permits private research
compute and aggregate paper reporting while keeping dataset redistribution
forbidden. It never emits `LICENSE_PASS`.

The scientific contract is one task-specific three-class GINE with labels
`0=Bitter`, `1=Sweet`, `2=Tasteless`, source label `1`, and strict flip
`pred_before == 1 and pred_after != 1`. Sweet-to-Bitter and
Sweet-to-Tasteless are both valid. RF is forbidden. Training and selection use
train/validation only; calibration/test remain metadata-only until their
explicit later gates.

## Fixed private inputs

- Upstream commit: `16af8ead8a17b6bd3941d9eb5879c5be75c14114`
- Source CSV SHA-256: `b7308b3277fd07ed6af4b861c0d2ce2d843f92cc81a9e5e4efd65cf4040a291b`
- Prepared manifest SHA-256: `36aaf17bf45e0a092a96a0379fab31d9e6bfcd719b87cb4ffa4e57a6642bb645`
- Split manifest SHA-256: `841f3b911e5d353c1e00f010bafcc8a6f7b3433082dba8a8979fab1b558251af`
- Rows: train 9437, validation 1328, calibration 1328, test 1328
- Policy v2 raw SHA-256: `b370ed9655f0a566b3615fc321c547945dd73fcee27d637110b801a766e1ca1b`

No download, preparation, split, or graph-cache rebuild is authorized.

## Controller and scheduling contract

`scripts/autodl/launch_tastemolnet_main_v1.sh` creates a fresh
`control/tastemolnet-main-v1/<controller-id>` root and starts the main
controller with `nohup`. T0 and T1 must PASS before T2 becomes RUNNING. All
T0--T16 stages have separate `manifest.json`, `state.json`, `gate.json`,
`input_hashes.json`, and `output_hashes.json` evidence.

- GPU 0 and GPU 3 are protected BACE lanes and are never signalled.
- GPU 1 is the exclusive formal Taste GINE lane.
- GPU 2 is recorded as `READY_CLASSIFIER_INDEPENDENT_PRECOMPUTE`; the current
  commit does not start GPU-2 science automatically.
- GNN backbone ablation remains disabled and is recommended only after the
  main matrix reaches 16/16.
- The planning reservation is 20 GiB and must leave at least 100 GiB free.
- Taste GINE is not a main-result matrix cell. Only final method PASS may add
  one of the four Taste cells.

The GINE bundle must include `model.pt`, `last.pt`,
`last_checkpoint.json`, `checkpoint_reload.json`, model/config/schema/label and
split provenance, metrics, validation predictions, environment/Git evidence,
and `sha256sums.txt`. It emits a first-batch progress event and verifies a real
checkpoint reload before `[TASTE_GINE_THREE_CLASS_PASS]`.

T3 does not refit that bundle. It adopts the temperature already fitted on T2
validation logits only after recomputing NLL, ECE, Brier score, and argmax
invariance from the hash-closed bundle-internal `validation_predictions.csv`
and proving the complete source inventory unchanged. It opens no external
split payload and runs CPU-only without claiming a GPU. Its fresh evidence
root emits `[TASTE_GINE_CALIBRATION_PASS]` and a controller-facing `gate.json`;
it is named `calibrated-<timestamp>-<pid>`, contains no model copy, and binds
the single `checkpoint_id` that every downstream method must consume.
The root must be the exact direct fresh child of
`$AUTODL_ARTIFACT_ROOT/gnn_oracles/tastemolnet/gine/seed7`; neither a copied
checkpoint subtree nor an arbitrary output location is accepted.

T4 depends on that exact T3 gate, uses the selected `model.pt` once on physical
GPU 1, and opens only the frozen graph-cache manifest plus `calibration.pt`.
The cohort is the first sixteen calibration-order rows that are true Sweet,
predicted Sweet, and have exactly four valid connected one/two-atom deletions.
The smoke requires observed strict flips to both Bitter and Tasteless, then
records aggregate three-class destinations and counterfactual drops only. It
writes no CSV, SMILES, molecule identifiers, residual rows, or per-example
predictions, and must record `test_payload_opened=false` before emitting
`[TASTE_MULTICLASS_ORACLE_PASS]`.

Both stages retain the artifact/output-parent/output descriptors across
creation and preparation, then revalidate all source bytes/stat inventories,
both exact tracked policy authorities, and the prepared output closure while
the marker is absent. The PASS marker is the final commit. The T4 root uses the
same exact artifact formula with a
`t4-oracle-smoke-*` basename. Public held-stage and held-checkpoint APIs expose
the exact T2 path plus checkpoint ID/full inventory/stat inventory/manifest
hashes for T6; an equal-byte copied or symlink-aliased checkpoint is rejected.
The supplemental policy contains a typed train-only/no-RF/no-calibration/no-test
T6 authority whose sole training payload is the frozen prepared train CSV, but
T6 runtime/controller implementation remains a separate reviewed successor.

## Runtime evidence locations

The immutable deployment/launch step copies a runtime-populated version of
this handoff to
`$AUTODL_RUNTIME_ROOT/outputs/autodl/handoffs/`. The runtime copy records the
execution commit, controller PID/heartbeat, GPU UUID/PIDs, fresh output roots,
latest progress/checkpoint, matrix before/after hashes, storage evidence, and
status/restart commands. Values must come from live files/processes; this
checked-in document intentionally does not predeclare them.

## Status and restart

Read status without modifying science:

```bash
/root/miniconda3/envs/smiles_pip118/bin/python \
  scripts/autodl/status_tastemolnet_main_v1.py \
  --config configs/hpc.yaml \
  --controller-root /absolute/controller-root
```

Restart is permitted only with the original immutable execution tree and the
same controller ID, policy receipt, controller/science/state roots, and frozen
environment:

```bash
scripts/autodl/launch_tastemolnet_main_v1.sh restart
```

Do not delete or rewrite failed/partial roots, do not count the historical CPU
smoke as the formal model, and do not start an alternative backbone.
