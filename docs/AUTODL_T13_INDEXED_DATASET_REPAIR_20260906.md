# Taste T13 indexed augmentation — 2026-09-06

This is a storage/materialization repair, not an algorithm or decoder change.
The previous isolated T13 child exited with `-9` after preprocessing all 3,823
graphs and before its first optimizer checkpoint. PID-specific OOM is not proven;
the concrete code peak is the official full augmented mask stack, repeated
feature/adjacency/edge tensors and subsequent whole-universe one-hot allocation.
The failed attempt and its existing imported mining artifacts remain unchanged.

## Exact representation

`src/baselines/t13_indexed_augmentation.py` replaces only one FSG instance's
`expand_data_by_fs` on the explicitly isolated, HPC-mining-adopted Taste route.
The official source files are never modified. Original `get_graph_masks` retains
its graph traversal, mapping order, all masks, random draws and sampling. Each
mask is checked against its original full tensor before being stored as an
ordered Cartesian node axis. Sentinel masks remain exactly `-1`.

Parents retain their original shared tensors. A bounded eight-parent immutable
cache reconstructs the official padded tensors; every returned row owns its
mutable tensors. Augmented parent positions/rule indices/mask axes are compact
int32 arrays. Full augmented dense tensors are never stacked or repeated.
The original stratified sample split (`random_state=33`), multiplicity, sample
order, batch500, no-shuffle and zero-worker contract remain unchanged.

Every production parent with masks gets bounded first/last eager-item checks.
Every mask, not only these boundary samples, is reconstructed and checked. The
full index, masks, parent tensors/rules, split and RNG boundary have digests.
Materialization must not consume any Python/NumPy/Torch RNG. The small test
fixture additionally compares the complete original official eager dataset,
all items, splits and post-construction RNG against the indexed implementation.

## Canary, not automatic full permission

Run from the reviewed immutable AutoDL worktree under the existing GPU1 T13
reservation. Only train CSV is accepted; calibration/test payloads are not read.

```bash
python -I -B scripts/autodl/canary_t13_indexed_dataset.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --output-root <fresh-canary-root> \
  --official-root <unchanged-vendor-globalgce-root> \
  --train-csv <original-train.csv> \
  --gnn-checkpoint <original-calibrated-T3-directory> \
  --gspan-adoption-proof <existing-verified-import-adoption-proof.json> \
  --device cuda:0 --targets 0,2
```

Both target branches use the real model, frozen GINE and existing top20 mining
adoption. Each arm executes two optimizer updates, with the original maximum
five batches per update (including the original sixth fetch before break),
Adam/StepLR and loss policy. A bounded official eager batch and indexed batch
must be tensor-identical. Loss, model, optimizer, scheduler and RNG states must
match exactly. The lazy arm saves/loads after its first update and matches the
uninterrupted second update; a separate isolated CPU process reopens the saved
checkpoint. No full trajectory or final rule quality is claimed.

The final `<root>/canary.json` explicitly reports boolean
`index_contract_pass`, `mask_rng_batch_parity`, `training_step_parity`,
`reload_parity`, target order `[0,2]`, and `test_loaded=false`.
`full_trajectory_parity_claimed=false`. It does **not** declare memory admission.
The owner must apply the separately authorized physical lease, protected-task,
RSS/cgroup/VRAM and storage gates before consuming its one full-start receipt.

Continuous memory samples are in `memory_samples.json`; exact batch, forward,
backward, optimizer, save, load and restore boundaries are in
`target_{0,2}/training_canary/memory_boundaries.json`. Samples include process
VmRSS/VmHWM, cgroup usage/limit/failcnt and CUDA current/reserved/peak bytes.
The paired Slurm entrypoint intentionally refuses execution on HPC: Taste
training/GINE remain AutoDL-only.

## Checkpoints and recovery

Production keeps the original epoch optimizer cadence. Each epoch has only one
optimizer step, followed immediately by its atomic checkpoint, so the first
optimizer update is already the first durable training checkpoint. Its added
fields are the complete indexed dataset identity and explicit sampler cursor
`next_epoch`, `next_batch=0`, batch500/no-shuffle/zero-workers. Resume validates
these before restoring model, Adam, scheduler and all RNG state.

The formal top-level `checkpoint.json` is created at INITIALIZED and alone is
not training-resume evidence. A legal same-attempt resume must also bind
`raw/target_<n>/globalgce_training_checkpoints/training_checkpoint.pt`, or a
completed target branch manifest. No diagnostic canary checkpoint is a formal
full checkpoint. Existing failed roots are never overwritten or promoted.

The pinned upstream loop uses `range(epochs + 1)`; the configured100 behavior
is preserved rather than silently replacing it with a different update budget.

## Focused verification

```bash
GLOBALGCE_OFFICIAL_ROOT=<pinned-local-vendor-root> \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/baselines/globalgce/test_t13_indexed_augmentation.py \
  tests/baselines/globalgce/test_resumable_gspan_chunks.py \
  tests/baselines/test_tastemolnet_globalgce_full.py \
  tests/autodl/test_t8_hpc_t13_successor_v1.py
```

Local focused result: 55 PASS. CLI help, targeted compileall, shell syntax and
diff checks pass. Real production canary/admission and the authorized fresh
full successor are separate runtime actions; no such PASS is inferred here.
