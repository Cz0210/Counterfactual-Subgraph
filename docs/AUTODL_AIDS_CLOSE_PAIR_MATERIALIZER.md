# AIDS COMRECGC theta-close materializer

This CPU-only audit reconstructs the logical DBSCAN input from the frozen AIDS
GREED distance interface. It does not infer closeness from the norm of the
stored recourse vector and does not copy or modify the physical pair store.

The command validates the candidate-major/parent-minor row order and the
`(parent_index, candidate_index)` columns, recomputes

```text
normalized_distance = torch.cdist(candidate, parent) /
  (element_count(candidate) + element_count(parent))
```

and materializes the inclusive `normalized_distance <= 0.1` predicate. The
outputs are a float32 distance memmap and a uint8 close-pair bitmap with one
entry per physical row:

```text
distance_scan/normalized_distances.greed.float32.npy
distance_scan/close_pair_bitmap.greed.uint8.npy
```

Both names identify GREED as the distance authority. They are distinct from
any diagnostic recourse-norm scan. The full run also samples 1,000 random
rows, every physical chunk's first and last row, and the 1,000 rows closest to
theta to verify the stored vector direction and element-count scale.

Physical snapshot wrappers may contain no local chunk list. In that case, the
audit accepts chunk-order metadata only from the wrapper's regular,
non-symlink `source_manifest_path`, after checking the source manifest SHA-256,
the stable 560-chunk-list hash, scientific identity, row count, pair/vector
hashes, and pair order. The source metadata is re-statted and re-hashed after
the scan.

Use `--max-chunks 1` or `--max-chunks 2` for a bounded throughput benchmark.
Such a run writes resumable partial arrays and
`BENCHMARK_COMPLETE_NOT_SCIENTIFIC_PASS`; it never writes `PASS`. Resume the
same immutable execution commit and output directory with `--resume` and no
chunk limit. A scientific completion writes the exact bytes `PASS\n` only
after the full distance scan, formula audit, source closure, bitmap/distance
hashes, and (when applicable) `all_pairs_close_certificate.json` are durable.

The production full scan is launched only through
`run_aids_greed_full_scan_supervisor.py`. The controller's attempt-qualified
output is a small `pair_semantics_supervisor_receipt.json`; the arrays and
checkpoint stay in one campaign-owned `pair_semantics_science` root without an
attempt suffix. Attempt zero is always fresh and omits `--resume`. A child
transient failure or a later controller attempt may add `--resume` only after
the checkpoint identity and committed prefix are rehashed, both prior PID
generations are absent, the fixed-root flock inode is unchanged, no writable
FD/mapping targets the science tree, and scientific `PASS` is absent. Semantic
or provenance failures are terminal.

Completion freezes a supervisor terminal manifest and writes its PASS last.
Each fresh receipt binds that immutable manifest, the fixed science-root inode,
the lock inode, and the terminal science hashes. The theta-close view consumes
the receipt as its `input_manifest`, reopens the full receipt-to-science hash
closure, and only then reads the fixed-root distance array and contract.

The direct entrypoint is:

```bash
python scripts/autodl/run_aids_comrecgc_pair_semantics.py \
  --config configs/hpc.yaml \
  --project-root "$PWD" \
  --upstream-root "$COMRECGC_UPSTREAM_ROOT" \
  --dataset-dir "$COMRECGC_DATASET_DIR" \
  --source-csv "$COMRECGC_SOURCE_CSV" \
  --generation-dir "$COMRECGC_GENERATION_DIR" \
  --distance-checkpoint "$COMRECGC_DISTANCE_CHECKPOINT" \
  --pair-store-manifest "$COMRECGC_PAIR_STORE_MANIFEST" \
  --expected-pair-store-manifest-sha256 \
    "$COMRECGC_PAIR_STORE_MANIFEST_SHA256" \
  --output-dir "$COMRECGC_PAIR_SEMANTICS_OUTPUT" \
  --parent-limit 1283 --theta 0.1 --device cpu
```

The paired Slurm wrapper exists only to keep repository entrypoints in sync;
this AutoDL experiment must not be submitted to HPC.
