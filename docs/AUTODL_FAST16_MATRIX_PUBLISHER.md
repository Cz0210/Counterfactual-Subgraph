# AutoDL fast16 matrix publisher

This is a CPU-only publication layer for the final 4x4 paper matrix.  It does
not run or standardize science.  Each invocation reopens one exact paper-cell
terminal, acquires one shared `flock`, reads the latest immutable matrix
authority, appends exactly one previously missing cell, independently reopens
the result, and atomically advances the shared pointer.

## Shared authority

All Taste and non-Taste publishers must use the same two paths:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/publish.lock
/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json
```

The state schema is `fast16_matrix_authority_pointer_v1`.  Its authority root,
count, hashes, and applied-cell list are recomputed from the referenced root on
every reopen.  `--prior-authority-root` is only a one-time seed when the state
file is absent.  Once the state exists, a stale seed cannot branch the chain.
Every output root must be fresh; publication uses a staging directory and a
no-replace rename.

## Accepted terminals

- `AIDS/ComRecGC`: the controller-bound exact recovery **final stage**, not an
  exact-only receipt.  The appender reopens the controller terminal, exact
  stage, final stage, common-recourse closure, standardized freeze, RF inputs,
  source integrity, and writer quiescence.
- `Mutagenicity/ComRecGC`: the full
  `mut_comrecgc_exact_postprocess_v1` terminal.  Its original strict matrix
  append is reopened as terminal evidence only.  The old fork is never made
  the shared authority; the standardized cell is appended again from the
  pointer's current root.
- `BACE/GlobalGCE` and `BACE/ComRecGC`: exact
  `bace_frozen_cell_standardization_v1` terminals with 10--20 real unique
  rules, GINE/RF-false identity, calibration freeze before test, frozen input
  replay, and complete hash closure.
- Taste cells: the method-specific T11--T14 final contracts described in
  `AUTODL_TASTEMOLNET_MATRIX_APPEND.md`.  Smoke/canary/generation-only roots
  are not eligible.  In particular `[TASTE_T12_GCF_GENERATION_PASS]` is not a
  T12 paper-cell terminal.

All routes preserve every non-target matrix row byte-for-byte at the JSON
object level, require the dataset's passing Ours cell as identity reference,
and reject RF/GINE, split, checkpoint, MolCLR, threshold, selection/test, or
freeze drift.

## Direct one-cell publication

Example for BACE GlobalGCE (the shared state is assumed to exist):

```bash
AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
"$AUTODL_PYTHON" scripts/autodl/append_non_taste_matrix_authority.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset BACE \
  --method GlobalGCE \
  --cell-terminal-root /absolute/bace/globalgce/standardized \
  --output-root /absolute/fresh/matrix-authority-next
```

For AIDS also pass the exact recovery controller manifest:

```text
--aids-controller-manifest /absolute/controller.manifest.json
```

For Mutagenicity, `--cell-terminal-root` is the outer full exact-postprocess
root containing `PASS`, `run_manifest.json`, `science_manifest.json`, and the
nested `standardized/` directory.  Submit the paired HPC wrapper with:

```bash
sbatch scripts/slurm/append_non_taste_matrix_authority.sh \
  --dataset Mutagenicity --method ComRecGC \
  --cell-terminal-root /absolute/mut/exact-postprocess-final \
  --output-root /absolute/fresh/matrix-authority-next
```

## Durable sequential queue

The minimal queue waits for a fixed terminal root or a locator.  A locator is
written only by the owning terminal stage and has this exact shape:

```json
{
  "schema_version": "fast16_matrix_cell_root_locator_v1",
  "status": "READY",
  "dataset": "TasteMolNet",
  "method": "GCFExplainer",
  "terminal_root": "/absolute/future/t12-paper-terminal"
}
```

A queue manifest uses schema `fast16_matrix_publisher_queue_v1`.  Each cell has
its own fresh output root.  Known fixed Taste roots may be queued immediately:

```json
{
  "schema_version": "fast16_matrix_publisher_queue_v1",
  "initial_authority_root": "/absolute/current-authority-8",
  "authority_state_path": "/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json",
  "authority_lock_path": "/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/publish.lock",
  "poll_seconds": 60,
  "taste": {
    "t3_root": "/absolute/t3-terminal",
    "policy_path": "/absolute/tastemolnet-data-policy.json",
    "policy_receipt": "/absolute/tastemolnet-policy-receipt.json",
    "prepared_root": "/absolute/prepared-root",
    "graph_cache_root": "/absolute/graph-cache-root"
  },
  "cells": [
    {
      "dataset": "TasteMolNet",
      "method": "Ours",
      "terminal_root": "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/ours/t11-full/attempt-b73f789d-3888-4ec8-8e07-0442e224df29",
      "output_root": "/absolute/fresh/authority-after-t11"
    },
    {
      "dataset": "TasteMolNet",
      "method": "GCFExplainer",
      "terminal_root_locator": "/absolute/control/t12-paper-root-locator.json",
      "output_root": "/absolute/fresh/authority-after-t12"
    },
    {
      "dataset": "TasteMolNet",
      "method": "GlobalGCE",
      "terminal_root": "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t13-full/attempt-b2ea7297-72f0-4dcd-bf94-bd9283686db7",
      "output_root": "/absolute/fresh/authority-after-t13"
    },
    {
      "dataset": "TasteMolNet",
      "method": "ComRecGC",
      "terminal_root": "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/comrecgc/t14-postprocess/final-attempt-69290568-5baa-4560-a88b-0d9a6d4940b9",
      "output_root": "/absolute/fresh/authority-after-t14"
    }
  ]
}
```

Non-Taste cells use the same entries; AIDS additionally carries
`aids_controller_manifest`.  Unknown future roots should use locators rather
than globbing output directories.

Launch and inspect the queue:

```bash
export PROJECT_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/worktrees/fast16_matrix_publisher
export QUEUE_MANIFEST=/absolute/control/fast16-publisher-queue.json
export HEARTBEAT_PATH=/absolute/control/fast16-publisher-heartbeat.json
export LOG_PATH=/absolute/logs/fast16-publisher.log
export PID_PATH=/absolute/control/fast16-publisher.pid
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
bash scripts/autodl/launch_fast16_matrix_publisher_queue.sh

STATE_PATH=/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json \
  bash scripts/autodl/status_fast16_matrix_publisher_queue.sh
```

On restart, cells already present in `state.applied_cells` are reported as
`APPLIED` without touching their old output roots.  An absent locator or PASS
is `WAITING`.  A terminal validation failure is cached only for that terminal
fingerprint and retried after its terminal files change; other cells continue.
The queue exits when all configured cells are applied or after `--once`.
