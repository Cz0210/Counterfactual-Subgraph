# BACE and TasteMolNet frozen-GNN route on AutoDL

Date: 2026-08-22

## Audit conclusions

- The frozen BACE split is reusable: 1,513 unique valid molecules with
  scaffold-disjoint train/validation/calibration/test partitions of
  959/187/129/238. Validation and test remain unchanged.
- No existing BACE checkpoint qualifies as a task-specific frozen GNN.
- The historical BACE Morgan-RF teacher and every artifact selected or verified
  with it are `RF_CONTAMINATED`; they remain available for diagnosis but cannot
  enter this route.
- ChemLLM weights are proposal assets and MolCLR is a distance encoder. Both are
  independent of the new classifier checkpoint.
- AutoDL has a compatible Python 3.10 / Torch 2.7.1+cu118 / PyG 2.8 / RDKit
  environment. The implementation detects this environment and does not
  reinstall or upgrade it blindly.

## Classifier route

The default classifier is a five-layer GINE with shared atom/bond schemas,
mean pooling, residual connections, batch normalization, a two-layer readout,
class-weighted cross entropy, AdamW, and seed 7. The same registry exposes GIN,
GCN, and GATv2 for later sensitivity studies without duplicating data or
training pipelines.

BACE has two output logits; TasteMolNet has three. Checkpoint selection and
temperature fitting use validation only. Calibration data is reserved for
thresholds, selector fitting, and ordered-rule freezing. Held-out test data is
evaluated only after those choices are frozen. B2/B3 training never parses or
featurizes the test CSV: the immutable checkpoint bundle records only its path
and streaming SHA-256 in `test_evaluation_status.json`, with
`status=NOT_EVALUATED` and `test_loaded=false`. `evaluate_molecular_gnn.py` is
the sole entrypoint that loads the test split after the frozen choices exist.

## Counterfactual semantics

For source class `y`:

```text
strict_flip = pred_before == y and pred_after != y
cf_drop = p_before[y] - p_after[y]
margin_drop = margin_before - margin_after
```

Every record stores full probability vectors, source and destination labels,
the classifier checkpoint identity, and a separate MolCLR checkpoint identity.

## BACE state machine

The AutoDL route uses the following gated stages:

```text
B0_AUDIT -> B1_DATA_READY -> B2_GNN_SMOKE -> B3_GNN_FULL
-> B4_GNN_CALIBRATED -> B5_ORACLE_SMOKE -> B6_PPO_SMOKE
-> B7_PPO_FULL -> B8_POOL_BASE -> B9_POOL_HIGHTEMP
-> B10_POOL_MERGED -> B11_CROSS_PARENT_VERIFIED -> B12_SELECTOR
-> B13_FINAL_EVAL -> B14_FROZEN
```

Each stage writes state, gate, and manifest evidence. RF provenance or a failed
upstream gate prevents downstream launch. The new runner is independent of the
MUT/AIDS recovery controller and never invokes Slurm on AutoDL.

The control plane is always persistent. `AUTODL_CONTROL_ROOT` defaults to
`$AUTODL_DATA_ROOT/counterfactual-subgraph-runtime/control`, must be absolute,
must remain below the selected persistent data root, and must not be inside the
code worktree. Its exact resolved value and the exact Python executable are
frozen into every detached launch spec, so a fast NVMe code clone, tmux server,
and later SSH session all observe the same stage tree. AutoDL shell entrypoints
default `AUTODL_PYTHON` to
`/root/miniconda3/envs/smiles_pip118/bin/python` and fail closed if it is not an
absolute executable path.

TasteMolNet shell defaults are likewise persistent and commit-bound. The
upstream commit is fixed at
`16af8ead8a17b6bd3941d9eb5879c5be75c14114`; split and molecular-graph cache
roots default below the versioned prepared-data/runtime cache directories, not
below the fast code clone. These paths are exported but are not required by
non-Taste commands.

### B4 and B5 launch contract

`run_bace_gnn_calibration.sh` resolves the unique PASS/FROZEN B3 output from
its stage manifest. It atomically copies that uncalibrated bundle to a fresh
persistent B4 directory, fits one temperature from `val.csv` only, verifies
argmax invariance and finite before/after NLL, ECE, and Brier values, and leaves
the B3 bundle unchanged. The supplied validation file must match both the
absolute path and SHA-256 recorded under `files.validation` in B3's frozen
`split_manifest.json`; matching IDs or labels alone is insufficient.

`run_bace_gnn_oracle_smoke.sh` resolves the PASS/FROZEN B4 bundle, evaluates
`calibration.csv` without loading test, and freezes exactly 16 rows satisfying
`label == source_label == pred_before`. One loaded calibrated GNN checks
batch/single equivalence for the cohort and batches real parent/connected
residual pairs from bounded one- and two-atom deletions. It records every
`pred_before`, `pred_after`, `cf_drop`, and strict `cf_flip`, plus empty/invalid
deletion and RF guards. The checkpoint must declare exactly
`dataset=bace`, `num_classes=2`, and `source_label=1`. B5 fails closed unless
all 16 selected parents each yield at least one real connected, sanitized
deletion residual.

```bash
export AUTODL_DATA_ROOT=/autodl-fs/data
export AUTODL_CONTROL_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python

bash scripts/autodl/run_bace_gnn_calibration.sh
# After B4 state=PASS and gate=PASS:
bash scripts/autodl/run_bace_gnn_oracle_smoke.sh
```

## TasteMolNet execution boundary

`RUN_TASTEMOLNET=0` is the default. This task may prepare provenance, clean
data, scaffold splits, graph caches, configs, CPU tests, and a tiny forward
smoke. It must not launch full GNN training, PPO, candidate generation,
verification, selector fitting, or baselines until explicitly enabled.

Graph caches are built by the shared offline CLI. Each of the four split files
is featurized into a plain-tensor payload, reloaded with
`torch.load(..., weights_only=True)`, and bound to one cache manifest. The
destination must be absent, so a cache is never silently overwritten.

```bash
"$AUTODL_PYTHON" scripts/build_molecular_graph_cache.py \
  --config configs/hpc.yaml \
  --config configs/datasets/tastemolnet.yaml \
  --dataset tastemolnet \
  --data-dir "$TASTEMOLNET_SPLIT_ROOT" \
  --output-dir "$TASTEMOLNET_GRAPH_CACHE_ROOT"
```

The fixed upstream currently has no explicit repository/data license. The
prepared data and cache therefore remain private AutoDL foundation artifacts;
`LICENSE_REVIEW_REQUIRED` blocks heavy TasteMolNet training and public
redistribution even after the cache and bounded CPU smoke pass.
When that marker exists, the full launcher exits nonzero without launching and
prints only `[TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW]`; it never emits a
READY marker for a license-blocked foundation.
