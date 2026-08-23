# AutoDL AIDS ComRecGC exact-shortcut fresh-route runbook

This route finishes AIDS ComRecGC from immutable repair-v4 pair chunks without
ever resuming or writing the old attempt.  It is specialized to the audited
full Cartesian source: 71,642 candidates by 1,283 parents, exactly 91,916,686
rows in candidate-major/parent-minor order.  The reviewed adaptive exact-DBSCAN
core is commit `645c6e51b7abcdc5dd4a9e0a1226d71d020880da`.

## Why chunks are sufficient here

Chunkwise floating-point reductions are not allowed.  Instead, a fresh route
first proves every persistent pair chunk's physical path, SHA-256, schema,
scientific identity, row interval, and exact elementwise formula:

```text
parent_index    = row_index % 1283
candidate_index = row_index // 1283
```

It then reconstructs one contiguous `recourse_vectors.npy` on local XFS by
concatenating raw `.npy` data bytes in chunk order.  The target header and full
file SHA are deterministically derived from the persistent chunks.  Pair
indices remain an implicit read-only Cartesian view; no second 1.47 GB pair
array is created.  Adaptive proof and summary therefore read the same
contiguous vector layout as the legacy implementation.  No chunk-local mean,
distance, medoid, or selector reduction is performed.

The local file is a reconstructible cache, never scientific authority.  Its
persistent manifest binds every source chunk, the copied target hash, the
logical pair-array hash, and the formula.  A missing local cache may be rebuilt
under the same identity; source chunks and their fresh audit remain authority.

## Release gates

Do not stop repair-v4 or launch the fresh controller until all gates pass:

1. execution HEAD has reviewed core `645c6e51...` as a true ancestor plus the
   independently reviewed route commit;
2. a fresh snapshot of the old `phase=chunks` checkpoint is hash identical to
   a stable read and resides outside the old root;
3. all 560 pair/vector chunks rehash correctly, total 91,916,686 rows, have no
   overlap/gap, obey the Cartesian formula elementwise, and expose no writable
   FD or mapping; a live old owner is recorded as diagnostic-only and prevents
   actual adoption;
4. after a parent-authorized graceful stage-boundary stop, the formal adoption
   scan finds no old owner process and revalidates the complete source closure;
5. `/root/autodl-tmp` has at least target-vector-size plus 3 GiB available;
   `posix_fallocate` succeeds and the 3 GiB floor remains after reservation;
6. one route-wide exclusive scratch flock is held from cache construction
   through proof, summary, terminal validation, and standardized PASS;
7. adaptive exact proof, implicit-pair summary parity, interruption/resume,
   mutation/tamper, RSS/storage, controller, compile, shell, and diff tests pass;
8. the immutable execution worktree, fresh output root, fresh controller root,
   spec, and manifest are absent before atomic publication.

At current audited sizes, the vector target is 23,530,671,744 bytes and local
free space was 28,926,038,016 bytes, leaving about 5.02 GiB.  The route must
recompute these values immediately before reservation; historical numbers do
not authorize a launch.

## Scientific command contract

```bash
python scripts/baselines/comrecgc/run_common_recourse.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset aids --mode full --parent-limit 1283 --device cpu \
  --upstream-root "$COMRECGC_ROOT" \
  --dataset-dir "$AIDS_DATASET_DIR" \
  --source-csv "$AIDS_SOURCE_CSV" \
  --generation-dir "$AIDS_GENERATION_ROOT" \
  --distance-checkpoint "$AIDS_DISTANCE_CHECKPOINT" \
  --output-dir "$FRESH_COMMON_RECOURSE_ROOT" \
  --engine external_memory_exact_v1 \
  --external-pair-store-source-checkpoint "$FRESH_CHUNK_CHECKPOINT_SNAPSHOT" \
  --external-pair-store-source-owner-root "$OLD_COMMON_RECOURSE_ROOT" \
  --external-vector-cache-root "$LOCAL_VECTOR_CACHE_ROOT" \
  --external-vector-cache-lock "$LOCAL_VECTOR_CACHE_LOCK" \
  --external-vector-cache-min-free-gb 3 \
  --external-dbscan-shortcut-mode all_core_one_component_adaptive_anchor_v1 \
  --external-shortcut-seed-count 3 \
  --external-shortcut-failure-cap 4096 \
  --external-shortcut-query-block-size 65536 \
  --external-exact-fallback-max-samples 0 \
  --external-summary-block-size 65536 \
  --external-max-rss-gb 96 \
  --expected-sklearn-version 1.7.2
```

The fresh supervisor adds `--resume` only for the same output/cache roots and
the same hash-bound continuation contract.  A route-wide shell flock is held
outside the Python child for its full lifetime.  This is CPU-only and allocates
no GPU slot.

## Fail-closed outcomes

- active owner, writable source reference, checkpoint/chunk/path/stat/hash or
  Cartesian-order drift: no adoption;
- local free space below `target_size + 3 GiB`, failed physical allocation, or
  post-allocation floor below 3 GiB: resource block;
- more than 4096 first-pass failures, disconnected anchors, incomplete or
  unauthenticated progress ledger, lower-bound/core/attachment failure, source
  mutation, or sklearn drift: `EXACT_DBSCAN_COMPLEXITY_BLOCKED`/scientific
  failure; dense quadratic DBSCAN is never entered;
- missing or tampered proof, summary, source-adoption, or terminal closure:
  fail before chemistry, unified evaluation, freeze, or standardized PASS.

The paired Slurm wrapper mirrors the CLI only.  This AutoDL campaign does not
submit the route to HPC.
