# TasteMolNet multiclass baseline adapter foundation

## Historical baseline-fragment boundary

The original baseline fragment remains immutable and
`BLOCKED_LICENSE_REVIEW`. It does not train a classifier, run a baseline,
score a molecule, read held-out test data, or allocate an AutoDL GPU. Its old
binary licence gate is now historical evidence only and can no longer emit a
PASS or authorize work.

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

## Scoped research/reporting foundation

The blocked fragment is immutable. The blocked-fragment builder cannot relabel
its tasks or consume a scoped-policy receipt.

A separate foundation now lives at:

```text
configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml
configs/autodl/tastemolnet_gine_research_v1.yaml
src/baselines/tastemolnet_gine_research_tasks.py
scripts/autodl/build_tastemolnet_gine_research_tasks.py
```

The checked-in scoped policy is active for private computation and aggregate
reporting, so its GINE task is runnable only when it consumes the typed policy
receipt and checksum-closed existing prepared/cache roots. It is GPU-2
exclusive, fresh-root only, three-class, and still forbids data redistribution
and every upstream licence-PASS claim.  Its long-running trainer uses a
separate inode-bound epoch-checkpoint root and exact same-contract resume; the
final oracle directory remains absent until a staged bundle has passed the
full frozen-oracle audit.

A runnable fragment must be created at a fresh path and satisfy all of:

1. an independently reviewed active scoped policy and typed read-only
   exact-data receipt, while upstream terms remain `NOT_EXPLICITLY_STATED` and
   no `LICENSE_PASS` is produced;
2. reuse of the checksum-bound existing prepared split and graph cache, with no
   download, data preparation, cache rebuild, or source copy;
3. a frozen three-class GINE manifest with checkpoint, temperature, and feature
   schema hashes, no RF provenance, and no held-out-test use for selection;
4. one shared scaffold split and MolCLR checkpoint across the four methods;
5. train-only candidate generation, calibration-only pool merge/selection, and
   held-out test access only after an immutable selector freeze;
6. method-native action manifests and destination-distribution export; and
7. a sanitized, manifest-closed public artifact that passes
   `scripts/audit_public_artifact_no_dataset_redistribution.py` before any
   aggregate result is released.

`src/baselines/tastemolnet_multiclass_tasks.py` continues to serialize the
historical blocked contract. The new typed GINE fragment is intentionally
separate and requires a dedicated controller; it is ineligible for the generic
four-GPU controller. Its command is
`scripts/autodl/run_tastemolnet_gine_controller.sh`, which owns a fresh
CID/controller root and runs the reviewed GINE worker only through a durable
exec-startup barrier. Controller loss adopts the exact live PID generation;
scientific process loss gets one same-state-root retry; transient GPU-2 waiting
does not consume that retry. A controller `PASS` is valid only after typed
reopen of the full oracle bundle, policy receipt, deterministic finalization
claim/inventory, output-parent authority, training-state root/sentinel/lock,
and file SHA/stat inventories. The status CLI performs that same typed reopen
for terminal roots. The paired Slurm wrapper is a static AutoDL-only refusal.
See `docs/TASTEMOLNET_DATA_USAGE_POLICY.md` for the full
private-use/publication boundary.
