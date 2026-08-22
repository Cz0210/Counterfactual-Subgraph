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
| GlobalGCE | attachment-aware LHS→RHS transformation | exclusive GPU after bridge smoke | GPU (4 deterministic shards) | CPU | READY behind bridge gate |

The user-approved frozen-GINE bridge releases GlobalGCE only after two
independent gates. The attachment-aware action is checked against pinned
upstream commit `157e65c2850bc787f229a1ee8c60564906b933f2`: exact labelled LHS
subgraph matches determine the official mask order, the RHS overwrites only
that mask square, new RHS nodes are appended, and boundary attachments to
existing parent nodes remain unchanged. Every match is retained and molecule
shape, atom/bond vocabulary, connectivity, sanitization, and match identity
fail closed. The differentiable bridge then evaluates the exact frozen
`MolecularGNN` weights with a straight-through expected-embedding relaxation.
Classifier parameters stay `requires_grad=false`, are excluded from the
optimizer, remain in evaluation mode, and retain their physical checkpoint
hash. Gradients may reach only the official GlobalGCE soft node/adjacency/bond
and decoder parameters. A hard one-hot graph must numerically match the normal
calibrated GINE forward, while every final hard rule product is sanitized and
re-scored by that same ordinary oracle. Official GTGNN, RF, a trainable
classifier, surrogate GNN, full-graph substitution, and deletion substitution
remain forbidden.

The machine-readable summary is available without starting science work:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py describe --method GCFExplainer
$PY scripts/autodl/run_bace_baseline_gnn_route.py describe --method ComRecGC
$PY scripts/autodl/run_bace_baseline_gnn_route.py describe --method GlobalGCE
```

`task-fragment` emits the native method-facing schema
`bace_baseline_controller_fragment_v1`. That schema intentionally retains
`task_id`, `argv`, resource objects, and native output roots; it is not accepted
directly by `build_four_by_four_manifest.py`.

Use `generic-task-fragment` to write a fresh composer input in
`bace_baseline_generic_controller_fragment_v1` schema. The adapter preserves
the native fragment, maps every dependency to the controller's
`{dep_<task_id>_output}` token, replaces task-owned output arguments with
`{task_output}`, and gives every task an immutable `attempt-{attempt}` output.
It also records exact required files, a log marker, split access, and a
non-primary runner dataset. The controller injects `CUDA_VISIBLE_DEVICES` only
after acquiring a GPU UUID lock.

GCFExplainer, ComRecGC, and GlobalGCE train-route GPU tasks have priority below the B11
shard priority (90). Their two READY roots therefore claim two lanes first,
while B11 remains free to use the other lanes. Their later four-way
verification tasks sort after B11, preventing one baseline from taking all
four cards. GlobalGCE contributes a bounded CPU parity preflight, an exclusive
GPU bridge smoke, native rule training on the full frozen train cohort, then
the same calibration-freeze/test-after-freeze chain. It does not enter full
training unless the bridge smoke publishes `BRIDGE_PASS`.

The primary route passes preregistered train-only `min_freq=7`
(`round(0.02 * 360)`) and rejects any native BACE train CSV other than the
frozen 869-row vocabulary. Neither identity is selected from test data.

Example:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py generic-task-fragment \
  --method "$METHOD" --python "$PY" --project-root "$PWD" \
  --output-dir "$OUTPUT_ROOT" --gnn-checkpoint "$BACE_GINE" \
  --dataset-dir "$DATASET_DIR" --calibration-split "$BACE_CALIBRATION" \
  --test-split "$BACE_TEST" --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CKPT" \
  --neurosed-checkpoint "$NEUROSED" --official-root "$OFFICIAL_ROOT" \
  --neurosed-manifest "$NEUROSED_MANIFEST" \
  --globalgce-source-manifest "$BACE_SOURCE_MANIFEST" \
  --globalgce-native-train-csv "$BACE_TRAIN_CSV" \
  --fragment-output "$CONTROL/fragments/bace-${METHOD}.generic.json"

$PY scripts/autodl/build_four_by_four_manifest.py \
  --controller-id "$CONTROLLER_ID" \
  --task-fragment "$CONTROL/fragments/bace-${METHOD}.generic.json" \
  --output "$CONTROL/manifests/${CONTROLLER_ID}.json"
```

The production manifest loader recognizes the baseline selector as a genuine
calibration freeze and permits only the three explicit baseline held-out
stages after that selector. Each held-out task must still declare both
`selector_parameters_frozen=true` and `read_only_test=true`; path-based test
leak detection remains active.

Preflight (CPU, fresh output):

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py preflight \
  --method GlobalGCE \
  --gnn-checkpoint "$BACE_GINE" \
  --official-root "$GLOBALGCE_OFFICIAL" \
  --output-dir "$OUTPUT_ROOT/preflight"
```

`$GLOBALGCE_OFFICIAL` is an explicit read-only input, not an assumed populated
submodule. It must resolve to exact commit
`157e65c2850bc787f229a1ee8c60564906b933f2`; the preflight also verifies the
audited SHA-256 values of the native model, frequent-subgraph, model utility,
data, and utility sources. The final git bundle does not include this checkout,
so deployment must transfer or initialize it separately and pass the exact
path.

GlobalGCE required preflight artifacts are `route_contract.json`,
`oracle_provenance.json`, `official_source_audit.json`,
`official_tensor_parity.json`, `state.json`, `NATIVE_ACTION_READY`,
and terminal `READY`. The following bridge smoke publishes
`bridge_gradient_audit.json`, `BRIDGE_PASS`, and `PASS` only when hard-forward
parity, a nonzero transformation gradient, zero classifier gradients, an
unchanged checkpoint hash, and finite outputs all hold.

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py globalgce-bridge-smoke \
  --method GlobalGCE --gnn-checkpoint "$BACE_GINE" \
  --parent-smiles CCO --atom-symbol C --atom-symbol O \
  --atom-symbol Cl --atom-symbol H --atom-symbol N --atom-symbol F \
  --atom-symbol Br --atom-symbol S --atom-symbol I \
  --output-dir "$OUTPUT_ROOT/bridge-smoke" --device cuda:0

$PY scripts/autodl/run_bace_baseline_gnn_route.py globalgce-train-rules \
  --method GlobalGCE --gnn-checkpoint "$BACE_GINE" \
  --source-manifest "$BACE_SOURCE_MANIFEST" \
  --native-train-csv "$BACE_TRAIN_CSV" \
  --official-root "$GLOBALGCE_OFFICIAL" \
  --output-dir "$OUTPUT_ROOT/train-candidates" \
  --expected-parent-count 360 --epochs 100 --top-k-native 20 \
  --device cuda:0 --resume
```

The native CSV is the complete 959-row processed train split.  Before creating
the training root, the adapter verifies the prepared graph bundle and selects
its exact 869 frozen train IDs (360 project-label-1/GINE-label-0 sources plus
509 targets).  Its 162 validation rows are hash-audited but are not passed to
training; calibration and test rows are forbidden.  This is an ID-bound view,
not positional truncation.

If an audited native rule JSON already exists, the bounded forward canary
applies every exact LHS match and scores all valid products in one loaded-once
calibrated-GINE batch. It uses neither calibration/test rows nor an RF:

```bash
$PY scripts/autodl/run_bace_baseline_gnn_route.py globalgce-forward-canary \
  --method GlobalGCE --gnn-checkpoint "$BACE_GINE" \
  --rule-json "$RULE_JSON" --parent-id "$PARENT_ID" \
  --parent-smiles "$PARENT_SMILES" \
  --output-dir "$FRESH_CANARY_ROOT" --device cuda:0
```

The canary writes `native_gine_forward.json`, `run_manifest.json`,
`state.json`, and publishes `FORWARD_EVAL_PASS` last. It is diagnostic evidence;
only the bridge smoke plus full native rule route can release the paper cell.

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
