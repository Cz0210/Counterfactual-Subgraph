# TasteMolNet multiclass baseline adapter foundation

## Current execution boundary

TasteMolNet remains `BLOCKED_LICENSE_REVIEW`. The foundation in this document
does not train a classifier, run a baseline, score a molecule, read held-out
test data, or allocate an AutoDL GPU.

The current controller fragment contains three terminal tasks:

```text
tastemolnet_gcfexplainer
tastemolnet_globalgce
tastemolnet_comrecgc
```

Each task is `command=null`, `resource=cpu`, `manifest_only=true`, declares no
data split, and carries `blocked_reason=BLOCKED_LICENSE_REVIEW`. The tasks are
evidence of a gate, not placeholder work.

Build the immutable fragment with:

```bash
PYTHONPATH=$PWD python scripts/autodl/build_tastemolnet_multiclass_baseline_tasks.py \
  --config configs/hpc.yaml \
  --license-gate /absolute/path/to/taste_license_gate.json \
  --output /absolute/fresh/path/tastemolnet-baselines-blocked.json
```

The paired Slurm file exists only for repository CLI parity. This campaign is
AutoDL-only and must not submit it.

## Shared classifier and counterfactual contract

All methods must use exactly one task-specific frozen classifier:

```text
dataset=tastemolnet
oracle_backend=gnn
classifier_family=gine
num_classes=3
source_label=1
source_label_name=Sweet
rf_oracle_used=false
CF_MODE=untargeted_strict_flip
```

The strict flip is:

```python
pred_before == 1 and pred_after != 1
```

Both `Sweet -> Bitter` and `Sweet -> Tasteless` are valid. A separate binary
explainee, `1-label`, and `pred_after == 1-label` are forbidden. Every scored
record retains all three before/after probabilities and the destination label.

## Native method adapters

### GCFExplainer

The native action remains one complete counterfactual graph. A generated graph
is a counterfactual candidate when `pred(candidate) != source_label`; it is not
projected into a deletion fragment.

### GlobalGCE

If the native implementation requires a target class, it runs target branches
0 and 2 against the same three-class GINE. Rules are deduplicated by frozen
native `rule_hash` and identical LHS, RHS, and attachment-map identities before
calibration. A same-hash action mismatch fails with
`GLOBALGCE_RULE_HASH_COLLISION_OR_CORRUPTION`. Calibration sees the merged pool
once and freezes one common selector.

### ComRecGC

The destination condition is `pred_after != source_label`, so class 2 is not
lost through a binary target hard-code. A candidate remains admissible only
when it has one pinned-upstream single-edit transition, exact downstream hash,
unique true transition, no graph-hash collision, and
`graph_content_identity=canonical_global_graph_hash`. Parent metadata remains
provenance only.

## Fresh release contract

The blocked fragment is immutable. The blocked-fragment builder intentionally
rejects a license gate with `status=PASS`; it cannot relabel the old tasks.

A future runnable fragment must be created at a fresh path and satisfy all of:

1. an exact-data `tastemolnet_license_audit_v1` gate with `status=PASS`,
   `heavy_route_authorized=true`, `run_tastemolnet=true`, and an explicit reuse
   basis;
2. a frozen three-class GINE manifest with checkpoint, temperature, and feature
   schema hashes, no RF provenance, and no test use for selection;
3. one shared scaffold split and MolCLR checkpoint across the four methods;
4. train-only candidate generation, calibration-only pool merge/selection, and
   held-out test access only after an immutable selector freeze;
5. method-native action manifests and destination-distribution export.

`src/baselines/tastemolnet_multiclass_tasks.py` serializes this exact release
contract into every blocked fragment. Its hash makes later contract drift
visible.
