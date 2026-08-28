# TasteMolNet GCF NeuroSED pre-implementation boundary (historical)

This section records the interface and data boundary that preceded the
implementation below. It does not report a trained model or a scientific
result; the current implementation contract begins in the next top-level
section and still requires a real independently verified AutoDL run.

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

# TasteMolNet GCFExplainer NeuroSED Protocol

## 1. Scope and role

TasteMolNet uses a newly trained, dataset-specific NeuroSED model only as
GCFExplainer's auxiliary graph-distance/projection model. It is not a
classifier, oracle, teacher-label model, or replacement for the frozen
three-class calibrated GINE.

The classifier contract remains:

- labels `0=Bitter`, `1=Sweet`, `2=Tasteless`;
- source class `1=Sweet`;
- strict counterfactual flip `pred_before == 1 and pred_after != 1`;
- no RF oracle.

NeuroSED is classifier- and source-label-independent. It is not a main-method
matrix cell.

## 2. Pinned official semantics

The implementation was audited against these immutable sources:

| Authority | Commit / SHA-256 |
| --- | --- |
| `idea-iitd/greed` | `1c756f49625abb62c9f6de5b0059876a4c7499c1` |
| `idea-iitd/greed-expts` | `e85423dc943fda1979811e7449846efffec2a1e1` |
| GREED `neuro/models.py` | `c5653dd9eeec1add8d6ae6253c30908df5ab8962ea0d9f9a6f25d32c393e0e70` |
| GREED `neuro/train.py` | `8e4d425d9d63e0aa56d5a1e6e25738f511ca7b52b08ac297fcf2c1678bdf9e28` |
| bundled GCF `neurosed/models.py` | `8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60` |
| bundled GCF `distance.py` | `d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3` |

The retained method semantics are:

- eight GIN layers;
- 64-dimensional hidden and output representations;
- per-layer concatenation and global additive pooling;
- the official `NormSEDModel` training forward
  `||relu(z_query-z_parent)||_2`;
- interval loss
  `mean(relu(lb-pred)^2 + relu(pred-ub)^2)`;
- AdamW, CyclicLR, and gradient clipping at exactly `0.1`;
- validation-only checkpoint selection;
- a plain PyTorch `state_dict` checkpoint.

The GCFExplainer fork deliberately loads that isomorphic state dictionary into
its `NormGEDModel`, whose runtime forward is `||z_query-z_parent||_2`. The
official coverage path then divides the raw output by the sum of the two graph
element counts, where one graph has
`num_nodes + num_directed_edges / 2` elements. The training and runner forwards
must not be silently made identical: the difference is part of the pinned
pipeline actually used by this project.

## 3. Feature and pair construction

Taste molecules are parsed with RDKit, sanitized, expanded with explicit
hydrogens, and converted to undirected PyG graphs represented by two directed
edge entries. Node features are one-hot atomic-number channels derived only
from train. Validation must contain no train-unseen atomic number; otherwise
the run stops before training.

Pairs are built independently within each admitted split:

1. choose one parent from that split;
2. sample a proper connected induced subgraph by deterministic seeded BFS;
3. retain the original node features and all induced edges;
4. pair the connected subgraph with its own parent;
5. set `lb == ub` to the known construction edit count:
   omitted nodes plus omitted undirected edges.

There are no cross-parent, cross-split, opposite-label, classifier-ranked, or
test-derived pairs. Labels may be present in the admitted split CSV schema but
are never consumed by the pair builder or model.

## 4. Split boundary and leakage proof

The worker may open only:

- `train.csv` for pair construction and fitting;
- `validation.csv` for pair construction, early stopping, and model selection;
- the non-payload preparation `split_manifest.json` for partition authority.

The code rejects `expected_split=calibration` and `expected_split=test`, rejects
renamed split payloads, and requires every opened row to declare the expected
split. It never opens calibration/test SMILES, graph tensors, graph hashes,
pairs, labels, or embeddings.

The published `split_manifest.json` contains only counts and aggregate hashes:

- `neurosed_train_graph_ids_hash`;
- `neurosed_validation_graph_ids_hash`;
- train and validation source-file SHA-256 values;
- `calibration_loaded=false`;
- `test_loaded=false`;
- train/validation membership and disjointness proof;
- empty calibration/test intersection assertions derived from the authenticated
  split partition without opening those payloads.

Raw IDs, SMILES, pair rows, graph caches, and reconstructable split payloads
are not published.

## 5. Checkpoints and training selection

Every validation improvement creates a new UUIDv4 directory:

```text
checkpoints/<checkpoint_uuid>/
  model.pt
  checkpoint.json
```

Checkpoint paths are never deleted, overwritten, or reused. Selection uses
validation interval loss with validation MAE as the tie break. Test metrics do
not exist. After training, the selected checkpoint bytes are copied once to
`best.pt` for the GCF runner; `model.pt` records the final training state.

The finite harness adds a maximum epoch count and epoch-level patience so a
formal AutoDL attempt cannot run forever. This does not change the official
model, forward, loss, optimizer, scheduler, clipping, or split used for model
selection. The formal loader uses `num_workers=0`, so the managed scientific
worker does not create unregistered persistent/subprocess loader generations.

## 6. Managed execution v2 and publication

The AutoDL launcher runs exclusively on physical GPU 1 under the shared
UUID-scoped GPU lock. It uses managed execution v2 with one UUIDv4 attempt.
The scientific child writes only bundle files below the managed artifact root.
The managed worker itself writes only:

- `raw_evidence.json`;
- `worker_exit.json`;
- `SEALED.json`.

The worker cannot create `PASS`, `gate.json`, `verification.json`, an adoption
receipt, or a release marker. A separate verifier process holds and revalidates
the SEALED inventory, rechecks the scientific bundle, and is the only process
that calls the atomic no-replace terminal publisher. The final layout is:

```text
$RUNTIME/outputs/autodl/tastemolnet/gcfexplainer/neurosed/seed7/<timestamp>/
  artifacts/
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
    checkpoint_manifest.json
    health_gate.json
    sha256sums.txt
    checkpoints/<uuid>/...
  raw_evidence.json
  worker_exit.json
  SEALED.json
  verification.json
  gate.json
  PASS
```

No mutable hardlink is used. Identity drift or verification failure produces
`QUARANTINED`, never releases dependencies, and sends no SIGTERM/SIGKILL.
`AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0` is mandatory.

## 7. Independent health gate

Before terminal publication, the independent verifier checks:

- exact file inventory and SHA-256 closure;
- finite train loss, validation interval error, MAE, RMSE, Spearman rank, and
  predicted distances;
- strict reload into the official `NormSEDModel` training schema;
- strict reload into the GCF fork's `NormGEDModel` runner schema;
- batch/single agreement on non-dataset synthetic graph probes;
- CPU/GPU numeric tolerance on AutoDL;
- finite official graph-element-normalized distances;
- feature input compatibility;
- UUID-selected checkpoint binding and byte identity of `best.pt`;
- train/validation-only access and absence of calibration/test evidence;
- model-card role and `classifier=false` contract.

The worker's `health_gate.json` is raw evidence named
`READY_FOR_INDEPENDENT_VERIFICATION`; it is not a PASS. Only successful
independent publication may emit `[TASTE_GCF_NEUROSED_PASS]`.

## 8. AutoDL operation

This route is AutoDL-only. The paired Slurm scripts are static refusals.

Required environment includes the ordinary AutoDL roots plus:

```bash
RUN_TASTEMOLNET=1
TASTE_RESEARCH_COMPUTE_ALLOWED=1
TASTE_DATA_REDISTRIBUTION_ALLOWED=0
RUN_GNN_ABLATION=0
AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
TASTEMOLNET_MAIN_V2_CONTROLLER_ID=<controller-id>
```

The controller invokes:

```bash
scripts/autodl/launch_tastemolnet_neurosed.sh
```

The launcher checks physical GPU 1, acquires the shared GPU UUID lock, binds
the train/validation/preparation-manifest hashes into the managed attempt, and
publishes only to a fresh timestamp root. It neither inspects nor signals the
protected GPU 0/GPU 3 BACE tasks.

## 9. Release status

This repository change implements code, tests, documentation, and a runnable
managed route. It does not itself start training, claim a scientific PASS,
update the matrix, or release T7/T12. A real SEALED artifact and independent
verifier PASS are still required before GCFExplainer may consume `best.pt`.
