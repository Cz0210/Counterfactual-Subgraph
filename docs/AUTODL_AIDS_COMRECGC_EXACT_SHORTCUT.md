# AutoDL AIDS ComRecGC exact-shortcut fresh-route runbook

This route finishes AIDS ComRecGC from an immutable repair-v4 pair source
without ever resuming or writing the old attempt.  A terminally promoted
`pair_store/run_manifest.json` is always preferred and is adopted by physical
read-only reference.  Closed Cartesian chunks are a fallback only when that
terminal manifest does not exist.  The route is specialized to 71,642
candidates by 1,283 parents, exactly 91,916,686 rows in
candidate-major/parent-minor order.  The reviewed adaptive exact-DBSCAN core is
commit `645c6e51b7abcdc5dd4a9e0a1226d71d020880da`.

## Pair-source priority and inactivity gate

`COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT` names the old physical pair-store
directory.  At each fresh/same-root invocation the production wrapper chooses
exactly one source:

1. if `run_manifest.json` exists, it must be a nonempty physical regular file;
   the terminal arrays are used and all chunk/cache CLI arguments are removed;
2. only if the terminal manifest is absent may the wrapper forward the closed
   chunk checkpoint and local-cache contract;
3. an empty, symlinked, malformed, hash-drifting, or scientifically mismatched
   terminal is an error, never permission to fall back to chunks.

Terminal adoption recursively rejects every partial file or symlink below the
pair store, every writable FD/mapping to any sibling inode, and every live
process whose command names the old owner root.  It then hashes the promoted
manifest and both arrays and repeats the writer/stat closure at terminal
validation.  The old controller must therefore be naturally terminal or
parent-authorized and gracefully stopped at a safe stage boundary before the
fresh route can adopt it.

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
logical pair-array hash, and the formula.  Cache allocation is a separate,
authenticated phase: exact size/headroom are checked, `posix_fallocate` and
physical block allocation are verified, the NPY header and allocation contract
are checkpointed, and a crash before that checkpoint safely replays
`posix_fallocate`.  Every copy/resume checkpoint revalidates this evidence.
After the persistent cache manifest is terminal, a missing local file requires
a new fresh cache/adoption root; the old terminal closure is never rewritten.

## Release gates

Do not stop repair-v4 or launch the fresh controller until all gates pass:

1. execution HEAD has reviewed core `645c6e51...` as a true ancestor plus the
   independently reviewed route commit;
2. inspect the automatic pair source.  Prefer and fully revalidate an already
   promoted terminal; snapshot the `phase=chunks` checkpoint only when the
   terminal is truly absent;
3. for a chunk fallback, all 560 pair/vector chunks rehash correctly, total
   91,916,686 rows, have no overlap/gap, obey the Cartesian formula
   elementwise, and expose no writable FD or mapping; a live old owner is
   diagnostic-only and prevents adoption;
4. after a parent-authorized graceful stage-boundary stop, the formal adoption
   scan finds no old owner process and revalidates the complete source closure;
5. only for chunk fallback, `/root/autodl-tmp` has at least
   target-vector-size plus 3 GiB available; authenticated `posix_fallocate`
   succeeds and the 3 GiB floor remains after reservation and on every resume;
6. one route-wide exclusive scratch flock is held by the v5 supervisor from
   source choice through cache/proof/summary/terminal validation and
   standardized PASS; cache allocation also has its own non-conflicting lock;
7. adaptive exact proof, implicit-pair summary parity, interruption/resume,
   mutation/tamper, RSS/storage, controller, compile, shell, and diff tests pass;
8. the immutable execution worktree, fresh output root, fresh controller root,
   spec, and manifest are absent before atomic publication.

At current audited sizes, the vector target is 23,530,671,744 bytes and local
free space was 28,926,038,016 bytes, leaving about 5.02 GiB.  The route must
recompute these values immediately before reservation; historical numbers do
not authorize a launch.

## Fresh AutoDL v5 supervisor contract

The fresh controller task command is exactly:

```json
["bash", "{project_root}/scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh"]
```

In addition to the ordinary frozen AIDS continuation inputs, its immutable
environment must contain:

```bash
DATASET=aids
DEVICE=cpu
GPU_REQUIRED=0
COMMON_RECOURSE_ENGINE=external_memory_exact_v1
COMRECGC_COMMON_RECOURSE_RESUME=1
COMRECGC_EXTERNAL_MAX_RSS_GB=96
COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE=8
COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS=1
COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE=all_core_one_component_adaptive_anchor_v1
COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT=3
COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP=4096
COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE=65536
COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES=0
COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE=65536
COMRECGC_EXPECTED_SKLEARN_VERSION=1.7.2
COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT=<old-attempt>/common_recourse/external_memory/pair_store
COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT=<old-attempt>/common_recourse
COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT=<auto-root>/checkpoint.json
COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT=/root/autodl-tmp/<fresh-v5-id>/vector-cache
COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK=/root/autodl-tmp/<fresh-v5-id>/allocation.lock
COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB=3
COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT=/proc
COMRECGC_EXTERNAL_ROUTE_LOCK=/root/autodl-tmp/locks/aids-comrecgc-exact-v5.lock
AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

`COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST` must be absent: only the wrapper
may select the physical terminal after checking the automatic root.  The task
is CPU-only and owns no GPU UUID/slot.  Build the immutable controller/spec only
after the execution commit, source paths, headroom gate, and independent review
are frozen.  Start/restart it through the ordinary persistent controller using
the same manifest SHA; never construct a second attempt merely to recover a
live same-root supervisor.

## Chunk-fallback scientific CLI

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
the same hash-bound continuation contract.  It allows exactly one
process-loss-only same-root retry; semantic, identity, RSS, storage, sklearn,
or proof failures remain terminal.  This is CPU-only and allocates no GPU slot.

## Fail-closed outcomes

- active owner, any terminal partial/symlink, writable source-tree inode,
  checkpoint/chunk/path/stat/hash or Cartesian-order drift: no adoption;
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
