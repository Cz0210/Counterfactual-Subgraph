# AutoDL BACE Frozen-GNN B7 Prep and B8--B14

This route is AutoDL-first, foreground-only, RF-free, and controlled by the
persistent four-GPU controller.  The wrapper never detaches, acquires GPU
locks, retries, or writes experiment-registry records itself.

```bash
DRIVER=scripts/autodl/run_bace_frozen_gnn_downstream.sh
```

The machine-readable dependency and output contract is:

```text
configs/autodl/bace_frozen_gnn_downstream_tasks.json
```

## B7-parallel read-only preparation

All prep actions require a passing B6-v2 manifest.  They may read train or
calibration as declared, but never test, a policy checkpoint, or an unfinished
candidate pool.

```bash
$DRIVER prep \
  --prep-action CALIBRATION_GNN_BEFORE_CACHE \
  --b6-output "$B6_ROOT" \
  --calibration-split "$BACE_SPLIT_ROOT/calibration.csv" \
  --gnn-checkpoint "$BACE_GNN_CHECKPOINT" \
  --output-dir "$PREP_ROOT/gnn-before"

$DRIVER prep \
  --prep-action CALIBRATION_MOLCLR_PARENT_CACHE \
  --b6-output "$B6_ROOT" \
  --calibration-split "$BACE_SPLIT_ROOT/calibration.csv" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --node-embedding-cache-dir "$NODE_CACHE" \
  --output-dir "$PREP_ROOT/molclr-parents"

$DRIVER prep \
  --prep-action FIXED_SHARD_MANIFESTS \
  --b6-output "$B6_ROOT" \
  --train-split "$BACE_SPLIT_ROOT/train.csv" \
  --calibration-split "$BACE_SPLIT_ROOT/calibration.csv" \
  --output-dir "$PREP_ROOT/shards"

$DRIVER prep \
  --prep-action OUTPUT_PREFLIGHT \
  --b6-output "$B6_ROOT" \
  --planned-output-root "$B8_ROOT" \
  --planned-output-root "$B9_ROOT" \
  --planned-output-root "$B11_ROOT" \
  --output-dir "$PREP_ROOT/output-preflight"
```

## B8 and B9 fixed train-parent shards

Run shard indices 0, 1, 2, and 3 for each stage.  The code verifies the exact
B7 final adapter config/weights identities and recomputes the frozen on-disk
policy checkpoint hash before loading the adapter.

After an interrupted attempt, rerun the same immutable shard command and root
with `--resume`.  The resume fingerprint binds the train split, parent set,
policy bytes, GNN, generation config, stage, and shard.  If the atomic candidate
pool already landed, model generation is not repeated; otherwise the same
frozen shard is regenerated in place and the earlier failure evidence remains.

```bash
$DRIVER pool-shard \
  --stage B8_POOL_BASE \
  --shard-index 0 \
  --train-split "$BACE_SPLIT_ROOT/train.csv" \
  --b7-output "$B7_ROOT" \
  --policy-checkpoint "$B7_ROOT" \
  --base-model-path "$CHEMLLM_MODEL_PATH" \
  --gnn-checkpoint "$BACE_GNN_CHECKPOINT" \
  --output-dir "$B8_ROOT/shard-0"
```

Use `--stage B9_POOL_HIGHTEMP` and its own fresh shard root for B9.  B10 takes
exactly eight `--shard-dir` arguments: four B8 followed by four B9 in any input
order.  The merge itself sorts by stage and shard identity.

```bash
$DRIVER merge-pools \
  --shard-dir "$B8_ROOT/shard-0" \
  --shard-dir "$B8_ROOT/shard-1" \
  --shard-dir "$B8_ROOT/shard-2" \
  --shard-dir "$B8_ROOT/shard-3" \
  --shard-dir "$B9_ROOT/shard-0" \
  --shard-dir "$B9_ROOT/shard-1" \
  --shard-dir "$B9_ROOT/shard-2" \
  --shard-dir "$B9_ROOT/shard-3" \
  --output-dir "$B10_ROOT"
```

## B11 calibration verification and B12 freeze

Run four B11 shards, then merge them.  Each shard enumerates every exact
hard-deletion match, batches GINE inference, computes MolCLR WNode only for
strict flips, and retains the minimum finite WNode per parent-rule pair.

```bash
$DRIVER verify-shard \
  --stage B11_CROSS_PARENT_VERIFIED \
  --shard-index 0 \
  --split-path "$BACE_SPLIT_ROOT/calibration.csv" \
  --predecessor-output "$B10_ROOT" \
  --gnn-checkpoint "$BACE_GNN_CHECKPOINT" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_CACHE" \
  --parent-before-cache "$PREP_ROOT/gnn-before/calibration_parent_gnn_before.jsonl" \
  --output-dir "$B11_SHARDS/shard-0"

$DRIVER merge-verification \
  --stage B11_CROSS_PARENT_VERIFIED \
  --shard-dir "$B11_SHARDS/shard-0" \
  --shard-dir "$B11_SHARDS/shard-1" \
  --shard-dir "$B11_SHARDS/shard-2" \
  --shard-dir "$B11_SHARDS/shard-3" \
  --predecessor-output "$B10_ROOT" \
  --output-dir "$B11_ROOT"

$DRIVER select \
  --matrix-output "$B11_ROOT" \
  --output-dir "$B12_ROOT"
```

## B13 held-out test and B14 manifest-only freeze

B13 rejects raw test access unless both B12 arguments point to the same valid
frozen root.  First create the controller-compatible test parent manifest. This
is the first action allowed to resolve/open `test.csv`; it is intentionally not
part of B7 prep.

```bash
$DRIVER prepare-test-shards \
  --b12-output "$B12_ROOT" \
  --test-split "$BACE_SPLIT_ROOT/test.csv" \
  --output-dir "$B13_PARENT_MANIFEST_ROOT"
```

The output `test_parent_ids.frozen.json` is then materialized by the controller
into four immutable `shard-000` through `shard-003` documents. Run four fixed
test-parent verification shards and merge them as for B11.

```bash
$DRIVER verify-shard \
  --stage B13_FINAL_EVAL \
  --shard-index shard-000 \
  --parent-shard-manifest "$CONTROLLER_SHARDS/shard-000.json" \
  --split-path "$BACE_SPLIT_ROOT/test.csv" \
  --predecessor-output "$B12_ROOT" \
  --frozen-selection-manifest "$B12_ROOT/frozen_selection_manifest.json" \
  --gnn-checkpoint "$BACE_GNN_CHECKPOINT" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_CACHE" \
  --output-dir "$B13_SHARDS/shard-0"

$DRIVER merge-verification \
  --stage B13_FINAL_EVAL \
  --shard-dir "$B13_SHARDS/shard-0" \
  --shard-dir "$B13_SHARDS/shard-1" \
  --shard-dir "$B13_SHARDS/shard-2" \
  --shard-dir "$B13_SHARDS/shard-3" \
  --predecessor-output "$B12_ROOT" \
  --output-dir "$B13_ROOT"

$DRIVER freeze \
  --b12-output "$B12_ROOT" \
  --b13-output "$B13_ROOT" \
  --output-dir "$B14_ROOT"
```

B14 has no split arguments.  It validates B12/B13 manifests and their declared
artifact path, size, and SHA256 identities only.  A successful stage publishes
its atomic `PASS` marker last.  A failed fresh invocation preserves
`FAIL.json` and an atomic `FAILED` marker.

## Four-GPU controller injection contract

Use `configs/autodl/bace_frozen_gnn_downstream_tasks.json` as the source of
command arrays and task order. Replace the controller template's placeholder
B8--B14 tasks with the following graph (the B11/B13 merge tasks, not their
shard launchers, own the official scientific stage):

```text
B8_POOL_BASE[4] ----\
                     -> B10_POOL_MERGED
B9_POOL_HIGHTEMP[4]-/        |
                              v
B11_VERIFICATION_SHARDS[4] -> B11_CROSS_PARENT_VERIFIED (merge)
                              -> B12_SELECTOR
                              -> B13_TEST_PARENT_MANIFEST
                              -> B13_VERIFICATION_SHARDS[4]
                              -> B13_FINAL_EVAL (merge)
                              -> B14_FROZEN
```

The controller must expose successful instance outputs under these exact
tokens, always from the actual passing retry attempt:

```text
{dep_bace_b8_pool_base_shard_000_output} ... shard_003
{dep_bace_b9_pool_hightemp_shard_000_output} ... shard_003
{dep_bace_b11_verification_shards_shard_000_output} ... shard_003
{dep_bace_b13_verification_shards_shard_000_output} ... shard_003
```

Sanitize both task and instance IDs by replacing non-alphanumeric characters
with `_`. Dependency context must also be available while expanding
`shards.parent_manifest`; B8/B9 use the passing B7-prep train manifest, B11
uses its calibration manifest, and B13 uses the post-B12 test manifest. Never
substitute `attempt-0` literals. The controller keeps `{shard_id}` values such
as `shard-000` for stable instance/output naming and passes the separate
numeric `{shard_index}` value (0 through 3) to the foreground driver.  The
driver validates `{shard_manifest}` against its independent
sorted-position-modulo-four assignment.

B14's injected command has exactly `--b12-output`, `--b13-output`, and
`--output-dir`; it must not receive a split path, data root, parent manifest,
or candidate matrix argument.
