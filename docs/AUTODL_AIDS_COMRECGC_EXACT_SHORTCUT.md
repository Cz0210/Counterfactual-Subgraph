# AutoDL AIDS ComRecGC exact-shortcut fresh-route runbook

This route finishes AIDS ComRecGC from an immutable repair-v4 pair source
without ever resuming, writing, or signalling the old attempt.  The protected
repair-v4 process still holds its terminal vector inode through an `O_RDWR`
mapping, so production v5 first copies the promoted pair arrays to fresh
physical inodes on persistent AutoFS.  Direct old-inode adoption and chunk
fallback are both forbidden in production.  The route is specialized to 71,642
candidates by 1,283 parents, exactly 91,916,686 rows in
candidate-major/parent-minor order.  The independently reviewed adaptive
exact-DBSCAN source commit is
`645c6e51b7abcdc5dd4a9e0a1226d71d020880da`; its exact two-file blob-equivalent
integration commit is `8c371b1c8ee1d8188555581c4f8e8b6060ae42eb`.

## Production physical-snapshot contract (supersedes chunk fallback below)

The snapshot task binds the terminal manifest SHA-256 and the exact old
PID/start-ticks/raw-command/cwd/output identity.  Only that frozen generation
may retain a writable source reference.  The task performs a full source
manifest/array hash, stat, NPY-schema, and all-row Cartesian-order check;
sequentially copies pair indices and vectors through non-authoritative
partials; synchronizes and atomically promotes each destination; and repeats
the complete source hash/stat closure.  Natural exit of the old process is
allowed, but PID reuse, another common-recourse process, or source drift is
not.  Source and destination device/inode pairs must differ, so source
hardlinks are forbidden.

Same-root recovery restarts only an incomplete partial.  A promoted array is
reused only after full hash, size, schema, and distinct-inode validation.  A
terminal manifest-to-PASS crash is reconciled through the same whole-closure
validator.  The task requires at least 40 GiB to remain after all still-missing
copy bytes.  Destination symlinks, partials, and writable FD/mmap references
fail closed.  Science depends on this task and independently revalidates its
terminal closure before starting.

`dbscan_contract.json` states that all 91,916,686 rows are
candidate-by-parent recourse embedding vectors, not precomputed distance edges
or adjacency.  It freezes Euclidean `eps=0.02`, `min_samples=3`, inclusion of
the sample itself, sklearn 1.7.2 brute behavior, and sklearn border/label
ordering.  The adaptive anchor certificate is exact because it proves a
sufficient neighbor lower bound for every sample and connectedness of the
anchor epsilon graph; it never treats `pair_indices` as adjacency.  Certificate
failure blocks because dense fallback is disabled.

The older chunk/cache discussion later in this document describes a retained
generic development capability only.  It is not reachable from the production
v5 manifest.

## Pair-source and process gate

Production requires the old promoted terminal; absence or corruption blocks.
The snapshot source audit allows the frozen old PID's known `O_RDWR` mapping
only while pre/post content and stat closure remains identical.  The fresh
destination allows no writable reference.  Its PID, Linux start ticks, raw
command-line SHA-256, exact output argument, and execution-worktree cwd are
frozen in the v5 spec.  At build and snapshot start that exact generation must
still be present.  Later validation permits its natural exit, but PID reuse,
identity drift, or any second `run_common_recourse.py` fails closed.

While science is running, the same scan repeats every monitor interval and
permits only the old generation (if still alive) plus at most one exact
`OUTPUT_ROOT/common_recourse` child descended from the PID/start-tick-bound v5
science generation, with the frozen script path and execution cwd.  Any rogue
third process or ancestor reuse terminates only the fresh v5 process group.
The cgroup-v1 `memory.limit_in_bytes - memory.usage_in_bytes` 128 GiB floor is
rechecked on the same loop; host `MemFree` is not used as a substitute.

## Retained non-production chunk capability

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

1. execution HEAD has integrated reviewed core `8c371b1c...` as a true
   ancestor plus the independently reviewed route commit; the builder also
   verifies the frozen Git blob IDs and current SHA-256 of the reviewed
   implementation and focused test against review source `645c6e51...`;
2. require and fully revalidate the promoted terminal; production has no
   terminal-absent or chunk fallback;
3. prove all 91,916,686 Cartesian rows, copy both arrays to persistent fresh
   physical inodes, retain at least 40 GiB after missing copy bytes, and bind
   source/destination hashes plus `dbscan_contract.json`;
4. require the exact spec-bound old process at build/snapshot start; later only
   its natural exit is accepted, and any other common-recourse generation is
   rejected;
5. keep local NVMe out of the production scientific-authority chain;
6. one route-wide exclusive scratch flock is held by the v5 supervisor from
   source choice through cache/proof/summary/terminal validation and
   standardized PASS; cache allocation also has its own non-conflicting lock;
7. adaptive exact proof, implicit-pair summary parity, interruption/resume,
   mutation/tamper, RSS/storage, controller, compile, shell, and diff tests pass;
8. the immutable execution worktree, fresh output root, fresh controller root,
   spec, and manifest are absent before atomic publication.

At the audited source, both arrays total 25,001,338,848 bytes and AutoFS had
75,902,263,296 bytes free, leaving about 50.9 GB after the copy.  The route
recomputes these values under its lock; historical numbers do not authorize a
launch.

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
COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT={dep_aids_comrecgc_pair_store_physical_snapshot_v5_output}/pair_store
COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL=1
COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT={dep_aids_comrecgc_pair_store_physical_snapshot_v5_output}/pair_store
COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB=3
COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT=/proc
COMRECGC_EXTERNAL_ROUTE_LOCK=/root/autodl-tmp/locks/aids-comrecgc-exact-v5.lock
AIDS_COMRECGC_V5_ALLOWED_OLD_PID=<frozen old PID>
AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS=<frozen /proc starttime>
AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256=<frozen raw cmdline SHA-256>
AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT=<old common_recourse output>
AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT=<old immutable worktree>
AIDS_COMRECGC_V5_SNAPSHOT_ROOT={dep_aids_comrecgc_pair_store_physical_snapshot_v5_output}
AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT=<old promoted pair_store>
AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256=<frozen terminal manifest SHA-256>
AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT=/proc
AIDS_COMRECGC_V5_SNAPSHOT_MIN_FREE_AFTER_BYTES=42949672960
AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS=91916686
AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM=64
AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT=1283
AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT=71642
COMRECGC_CGROUP_MEMORY_ROOT=/sys/fs/cgroup/memory
AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES=137438953472
AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

`COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST` must be absent: only the wrapper
may select the physical terminal after checking the automatic root.  The task
also forbids the chunk checkpoint/vector-cache variables.  It is CPU-only and
owns no GPU UUID/slot.  Build the immutable controller/spec only after the
execution commit, source paths, terminal source audit, and independent review
are frozen.  Start/restart it through the ordinary persistent controller using
the same manifest SHA; never construct a second attempt merely to recover a
live same-root supervisor.

The generic chunk-rebuild CLI (not used by production v5) also requires
`--external-vector-cache-route-lock`.  A zero-byte, truncated-header, or
wrong-schema cache partial is recoverable only while the authenticated
checkpoint is still `allocate_cache` and both the allocation lock and this
independent outer route lock are held.  The replacement is written through a
separate `.rebuild.partial.npy`, fsynced, schema-validated, and atomically
promoted.  Once `allocation_complete` has been checkpointed, either malformed
artifact is terminal corruption and is never deleted or reconstructed.

### Persistent v5 publication

The frozen identities are:

```text
controller_id=four_methods_four_datasets_aids_comrecgc_exact_route_v5
selector_task_id=aids_comrecgc_exact_route_v5_selector_freeze
snapshot_task_id=aids_comrecgc_pair_store_physical_snapshot_v5
terminal_task_id=aids_comrecgc_standardized_exact_route_v5
snapshot_output=<fresh_output_root>/source_snapshot/attempt-0
terminal_output=<fresh_output_root>/cells/aids/comrecgc/standardized/attempt-0
```

Create an immutable execution worktree whose HEAD contains snapshot release
`87050d3e02f7e3468227eec44e31e86aad048dad` plus its v5 builder integration,
then fill a fresh copy of
`configs/autodl/aids_comrecgc_exact_route_v5.template.json`, then publish and
launch exactly once:

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
CONTROL="$RUNTIME/control"
CID=four_methods_four_datasets_aids_comrecgc_exact_route_v5
SPEC="$CONTROL/four_methods_four_datasets_continuation/specs/$CID.json"
MANIFEST="$CONTROL/four_methods_four_datasets_continuation/manifests/$CID.json"

"$PY" scripts/autodl/build_aids_comrecgc_exact_route_v5_manifest.py \
  --config configs/hpc.yaml validate --spec "$SPEC"
"$PY" scripts/autodl/build_aids_comrecgc_exact_route_v5_manifest.py \
  --config configs/hpc.yaml build --spec "$SPEC" --output "$MANIFEST"

AUTODL_DATA_ROOT=/autodl-fs/data \
AUTODL_RUNTIME_ROOT="$RUNTIME" \
AUTODL_CONTROL_ROOT="$CONTROL" \
AUTODL_PYTHON="$PY" \
scripts/autodl/launch_four_by_four.sh "$MANIFEST"
```

The builder performs a full physical checksum/schema scan of the promoted pair
arrays, rejects partial/symlink/unexpected-writer evidence, binds and verifies
the sole allowed old source generation during snapshot, binds the exact repair-v4
manifest/scientific inputs, verifies live cgroup headroom, and publishes only
at the controller namespace path.  The supervisor repeats the process-set and
128 GiB cgroup-free gates before every attempt while the v5 child keeps its
independent 96 GiB RSS ceiling.  Before the first science child it also queues
a generation-monitored helper on the exact global high-memory flock.  The old
v4 process retains the lock while alive; on natural exit the helper acquires
and holds it through v5 supervisor exit.  A failed/reused helper generation
terminates only the fresh v5 process group and fails closed.  It records a
`mut_dependency` object in the manifest.  A fresh Mut controller must bind the
new manifest physical SHA256 plus the controller/task/output identities above;
the old Mut wait controller is not edited or repointed.

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

- unbound/changed/second common-recourse process, any terminal
  partial/symlink, writable source-tree inode,
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
