# TasteMolNet GCF NeuroSED protocol

This document freezes the interface and data boundary for the
TasteMolNet-specific GCFExplainer auxiliary distance model. It does not report
a trained model or a scientific result. Training remains blocked until the
reviewed pair builder, labels, loss, validation selector, and checkpoint writer
from the approved NeuroSED/GREED route are present as pinned project code.

## Role and existing implementation

NeuroSED is an auxiliary graph-distance/projection model. It is not the Taste
classifier, oracle, teacher, or a source of class labels. Every T7 prediction
continues to come from the same frozen calibrated three-class GINE used by T3.

The checked-in official GCF route fixes the inference architecture and use:

- `baselines/gcfexplainer_official/neurosed/models.py` defines
  `NormGEDModel`, whose graph encoder is `EmbedModel` and whose predicted
  distance is the L2 norm between the two graph embeddings;
- `baselines/gcfexplainer_official/distance.py::load_neurosed` constructs
  `NormGEDModel(8, input_dim, 64, 64)`, reloads its state dict, switches to
  evaluation mode, and embeds the original target graphs;
- `baselines/gcfexplainer_official/importance.py` uses
  `predict_outer_with_queries`, divides distance by the sum of query and target
  graph element counts, and defines coverage by `normalized_distance <= theta`;
- the T7 adapter calls that exact threshold-coverage function. It does not
  substitute neutral coverage, NetworkX GED, a BACE model, or deletion-only
  semantics.

The vendored GCF tree does not contain the approved NeuroSED training pair
builder, pair-label pipeline, optimizer/seed configuration, early-stopping
selector, or checkpoint bundle writer. Those details must therefore be
adopted from the pinned approved GREED/NeuroSED training implementation before
training; they must not be invented in T7.

## Split boundary

Only Taste `train` graphs may generate training pairs or update parameters.
Only Taste `validation` graphs may be used for early stopping, model selection,
and distance calibration. The `calibration` and `test` splits may not be
loaded, hashed into a pair, embedded, scored, or used to select a checkpoint.
NeuroSED training may be label-agnostic, but any graph-edit targets must come
from the approved existing pair-label pipeline.

The independent health gate must emit and compare:

```text
neurosed_train_graph_ids_hash
neurosed_validation_graph_ids_hash
calibration_loaded=false
test_loaded=false
```

It must prove that NeuroSED train IDs are a subset of the official Taste train
IDs, validation IDs are a subset of official Taste validation IDs, and both
sets are disjoint from calibration and test. The split manifest itself is a
held predecessor and its SHA-256 is part of the PASS closure.

## Required fresh bundle

One successful seed-7 run uses a unique fresh root under:

```text
$RUNTIME/outputs/autodl/tastemolnet/gcfexplainer/neurosed/seed7/<timestamp>/
```

The exact inventory is:

```text
model.pt
best.pt
config.yaml
model_card.json
pair_manifest.json
split_manifest.json
training_metrics.json
validation_metrics.json
feature_schema.json
environment.json
git_state.json
sha256sums.txt
```

The model card must state:

```text
dataset=tastemolnet
role=GCF_AUXILIARY_DISTANCE_MODEL
classifier=false
source_label_independent=true
train_only_fit=true
validation_only_selection=true
calibration_loaded=false
test_loaded=false
```

## Health gate and T7 handoff

The independent NeuroSED verifier must check finite loss and distances,
checkpoint reload, batch/single agreement, CPU/GPU numeric tolerance, a finite
validation rank/error metric, split non-leakage, feature-schema compatibility,
and successful loading by the official GCF runner. Test performance must not
select the checkpoint.

Only the independently verified `[TASTE_GCF_NEUROSED_PASS]` evidence and its
canonical SHA-256 may enter T7. T7 holds both that evidence and `best.pt`, loads
the checkpoint through `/proc/self/fd/<held-fd>`, revalidates it before and
after official loading and after the VRRW continuation, and cross-binds its
path/SHA in the managed-v2 attempt and raw evidence. A missing or drifting
predecessor blocks before science; the T7 worker cannot issue its own
NeuroSED, gate, adoption, or terminal PASS.
