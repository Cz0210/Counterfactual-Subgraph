# AutoDL BACE baseline Frozen-GINE routes

This runbook is for the BACE cells `GCFExplainer`, `GlobalGCE`, and
`ComRecGC`. All output roots below must be fresh absolute paths under the
persistent runtime. The only allowed classifier is:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689
```

Every successful stage writes its manifest first and publishes `PASS` last.
Failures publish `BLOCKED` or `BLOCKED_CODE`; they never publish `PASS`.

## Controller contract

| Method | Native action | Generation | Verify | Select | Current route |
| --- | --- | --- | --- | --- | --- |
| GCFExplainer | full counterfactual graph | GPU | GPU (4 deterministic shards) | CPU | READY |
| ComRecGC | lineage-validated common-recourse graph medoid | GPU | GPU (4 deterministic shards) | CPU | READY |
| GlobalGCE | attachment-aware LHS→RHS transformation | CPU | CPU | CPU | `BLOCKED_CODE` |

GlobalGCE is blocked by
`BLOCKED_GLOBALGCE_LHS_RHS_ATTACHMENT_MAPPING_UNAVAILABLE`. The existing
adapter records LHS/RHS identities but does not implement an audited atom and
bond attachment mapping. A full-graph or deletion replacement would change the
method and is therefore forbidden.

The machine-readable summary is available without starting science work:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py describe --method GCFExplainer
$PY scripts/autodl/run_bace_baseline_gnn_route.py describe --method ComRecGC
$PY scripts/autodl/run_bace_baseline_gnn_route.py describe --method GlobalGCE
```

Generate a dependency-complete JSON fragment for the generic four-GPU
controller with `task-fragment`. It emits exact argv/env/input/output/marker
contracts. The controller injects only `CUDA_VISIBLE_DEVICES` after acquiring a
GPU UUID lock. For GlobalGCE, the fragment has `tasks=[]` and one static
`BLOCKED_CODE` terminal, so it cannot consume a scheduler slot:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py task-fragment \
  --method "$METHOD" --python "$PY" --project-root "$PWD" \
  --output-dir "$OUTPUT_ROOT" --gnn-checkpoint "$BACE_GINE" \
  --dataset-dir "$DATASET_DIR" --calibration-split "$BACE_CALIBRATION" \
  --test-split "$BACE_TEST" --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CKPT" \
  --neurosed-checkpoint "$NEUROSED" --official-root "$OFFICIAL_ROOT" \
  --neurosed-manifest "$NEUROSED_MANIFEST"
```

Preflight (CPU, fresh output):

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py preflight \
  --method "$METHOD" \
  --gnn-checkpoint "$BACE_GINE" \
  --output-dir "$OUTPUT_ROOT/preflight"
```

Required preflight artifacts are `route_contract.json`,
`oracle_provenance.json`, `state.json`, and terminal `READY` or
`BLOCKED_CODE`.

## GCFExplainer foreground route

`$DATASET_DIR` is the existing train-only prepared BACE official graph bundle;
`$GCF_OFFICIAL` and the NeuroSED inputs are shared read-only dependencies.

```bash
$PY scripts/baselines/gcfexplainer/run_bace_vrrw.py \
  --dataset-dir "$DATASET_DIR" \
  --official-root "$GCF_OFFICIAL" \
  --gnn-checkpoint "$BACE_GINE" \
  --neurosed-checkpoint "$NEUROSED" \
  --neurosed-manifest "$NEUROSED_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/train_vrrw" \
  --profile full --parent-limit 360 --m 50000 --device1 cuda:0 --device2 cuda:0

$PY scripts/baselines/gcfexplainer/run_bace_summary.py \
  --dataset-dir "$DATASET_DIR" \
  --official-root "$GCF_OFFICIAL" \
  --vrrw-dir "$OUTPUT_ROOT/train_vrrw" \
  --gnn-checkpoint "$BACE_GINE" \
  --neurosed-checkpoint "$NEUROSED" \
  --output-dir "$OUTPUT_ROOT/train_summary" \
  --profile full --native-candidate-limit 0 --device cuda:0

$PY scripts/autodl/run_bace_baseline_gnn_route.py gcf-export \
  --method GCFExplainer \
  --dataset-dir "$DATASET_DIR" \
  --summary-dir "$OUTPUT_ROOT/train_summary" \
  --gnn-checkpoint "$BACE_GINE" \
  --output-dir "$OUTPUT_ROOT/train_candidates" \
  --profile full --parent-limit 360 --scan-limit 0 --device cuda:0
```

Required train artifacts: `candidate_universe.jsonl`,
`candidate_filter_audit.jsonl`, `oracle_provenance.json`, `run_manifest.json`,
and terminal `PASS`. Historical RF candidate/order artifacts are rejected.

## ComRecGC foreground route

The execution worktree and all upstream inputs must remain immutable once the
50,000-step generation begins. Checkpoint and trace roots must be persistent.

```bash
$PY scripts/baselines/comrecgc/run_generation.py \
  --route project --dataset bace --mode full \
  --project-root "$PWD" --upstream-root "$COMRECGC_OFFICIAL" \
  --dataset-dir "$DATASET_DIR" \
  --gnn-checkpoint "$BACE_GINE" \
  --distance-checkpoint "$NEUROSED" \
  --parent-limit 360 --device cuda:0 \
  --output-dir "$OUTPUT_ROOT/train_generation" \
  --checkpoint-root "$OUTPUT_ROOT/checkpoints" \
  --checkpoint-mirror-root "$OUTPUT_ROOT/checkpoint_mirror" \
  --trace-output-dir "$OUTPUT_ROOT/trace" \
  --graph-state-dir "$OUTPUT_ROOT/graph_state" \
  --storage-guard-root "$RUNTIME"

$PY scripts/baselines/comrecgc/run_common_recourse.py \
  --dataset bace --mode full --upstream-root "$COMRECGC_OFFICIAL" \
  --dataset-dir "$DATASET_DIR" \
  --generation-dir "$OUTPUT_ROOT/train_generation" \
  --distance-checkpoint "$NEUROSED" \
  --output-dir "$OUTPUT_ROOT/train_common_recourse" \
  --parent-limit 360 --device cuda:0

$PY scripts/autodl/run_bace_baseline_gnn_route.py comrecgc-export \
  --method ComRecGC \
  --common-recourse-dir "$OUTPUT_ROOT/train_common_recourse" \
  --dataset-summary-json "$DATASET_DIR/dataset_summary.json" \
  --gnn-checkpoint "$BACE_GINE" \
  --output-dir "$OUTPUT_ROOT/train_candidates" --device cuda:0
```

The exporter keeps the official common-recourse order and global graph
identity. Parent metadata remains provenance only; transition uniqueness and
lineage validation are not disabled.

## Shared calibration, selection, test, and freeze

For each READY method, launch shard indices `0..3`. Parent assignment is
`sorted(parent_id) position % 4` and does not depend on available GPU count.

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py verify-shard \
  --method "$METHOD" --verification-stage BASELINE_CALIBRATION_VERIFY \
  --split-path "$BACE_CALIBRATION" \
  --predecessor-output "$OUTPUT_ROOT/train_candidates" \
  --gnn-checkpoint "$BACE_GINE" \
  --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CKPT" \
  --shard-index "$SHARD" --wnode-cache-db "$WNODE_CACHE" \
  --node-embedding-cache-dir "$NODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/calibration/shard-$SHARD" --device cuda:0

$PY scripts/autodl/run_bace_baseline_gnn_route.py merge \
  --method "$METHOD" --verification-stage BASELINE_CALIBRATION_VERIFY \
  --predecessor-output "$OUTPUT_ROOT/train_candidates" \
  --shard-dir "$OUTPUT_ROOT/calibration/shard-0" \
  --shard-dir "$OUTPUT_ROOT/calibration/shard-1" \
  --shard-dir "$OUTPUT_ROOT/calibration/shard-2" \
  --shard-dir "$OUTPUT_ROOT/calibration/shard-3" \
  --output-dir "$OUTPUT_ROOT/calibration/merged"

$PY scripts/autodl/run_bace_baseline_gnn_route.py select \
  --method "$METHOD" --matrix-output "$OUTPUT_ROOT/calibration/merged" \
  --output-dir "$OUTPUT_ROOT/selection"
```

Only after `selection/frozen_selection_manifest.json` and `selection/PASS`
exist may the controller schedule test shards:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py verify-shard \
  --method "$METHOD" --verification-stage BASELINE_TEST_EVAL \
  --split-path "$BACE_TEST" --predecessor-output "$OUTPUT_ROOT/selection" \
  --gnn-checkpoint "$BACE_GINE" \
  --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CKPT" \
  --shard-index "$SHARD" --wnode-cache-db "$WNODE_CACHE" \
  --node-embedding-cache-dir "$NODE_CACHE" \
  --output-dir "$OUTPUT_ROOT/test/shard-$SHARD" --device cuda:0
```

Merge test shards exactly as calibration, with
`--verification-stage BASELINE_TEST_EVAL` and predecessor
`$OUTPUT_ROOT/selection`, then freeze:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py freeze \
  --method "$METHOD" --selection-output "$OUTPUT_ROOT/selection" \
  --test-output "$OUTPUT_ROOT/test/merged" \
  --output-dir "$OUTPUT_ROOT/final"
```

The final required files are `final_metrics.json`, `prefix_metrics.csv`,
`FINAL_PASS.json`, `run_manifest.json`, and terminal `PASS`.
