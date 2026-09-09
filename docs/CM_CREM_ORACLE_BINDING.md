# CM-CReM original-GINE and WNode binding (2026-09-10)

This is a new, user-authorized **full-graph prototype baseline**, not a change
to the project's deletion-based Ours method. No GINE/GIN/temperature is trained,
and no main-matrix writer is imported. BACE is the only implemented classifier
scope; conditional Taste requires its own future resolved authorization/binding.

## Actual audited upstream direction and node hook

Counterfactual Masking is pinned to
`b5816b502cde00ee24c652a02cbc54664583f773`.

- `source/explainability.py::GradCAM.node_importances` takes the gradient of
  `model.predict` and uses `gradient.mean(dim=1, keepdim=True)`, i.e. a separate
  mean **over channels for each node**. It multiplies the captured activation,
  sums channels, subtracts the minimum and divides by maximum plus `1e-9`.
  There is no ReLU. The adapter deliberately does not replace this with the
  conventional node-mean/channel-weight Grad-CAM formula.
- `source/models.py::GraphIsomorphismNetwork_classification` returns a sigmoid
  class probability. Its `c3` output precedes BatchNorm/ReLU/mean pooling. The
  original frozen project GINE analogue is its last `MolecularMessageLayer`,
  `layers.{num_layers-1}`, before residual addition, BN/ReLU and pooling.
  This cross-architecture correspondence is explicit, not an assertion that
  GINE and the author's GIN are the same model.
- The score is the original GINE's calibrated source-class probability,
  `softmax(raw_logits / original_T)[1]`, not an independent new classifier and
  not a derivative through the oracle's `no_grad` prediction method.
- Author CF generation uses `max(1, n // 5)` initial atoms, descending score with
  stable original-index ties. Its ring function adds rings intersecting the
  **initial** selection once (not a transitive ring closure). A full effective
  mask is `NO_REPLACEABLE_CONTEXT`, as specified in the new task contract.

All parameters remain `requires_grad=False`; the hook introduces a floating
activation leaf when needed. The actual prediction is unchanged. There is no
optimizer and no parameter gradient accumulation. Actual model/buffer tensors,
module modes, grad flags/grad buffers, Torch CPU/CUDA RNG and forward argmax
are checked around attribution. Raw logits must equal the original oracle
path (default exact; any nonzero original tolerance needs its pinned contract).
The hook is removed even on failure. A constant/zero real saliency is retained,
not replaced with random/Ours masks.

## Atom order and full-graph identities

`MolecularGraphFeaturizer.featurize(input_smiles)` preserves the RDKit parse's
atom order. Its `canonical_smiles` is metadata: reparsing that metadata and
retaining old indices is forbidden. Attribution passes the original Mol to
`cm_crem_generation.make_parent_request`, which supplies MolBlock, complete
ordered atom/bond digest, chirality-related transport flags and explicit mask.
The generation environment must reconstruct and verify that request. Explicit
hydrogen-node source inputs fail closed until a reviewed mapping exists.

Generated prototypes are independently sanitized, require one nonempty component,
no dummy/unsupported atoms or bonds, and are re-featurized in canonical isomeric
order. Their identity binds canonical SMILES, complete graph feature digest and
schema. Unchanged parents and non-destination predictions are scientific rejects;
EIO or an incomplete generation stage is not a scientific zero. Every retained
raw output's origin survives within the deduplicated target graph.
Original base/train parents do not inherit that generated-prototype connectivity
filter, but still must satisfy the original frozen featurizer's own input gate.
The adopted BACE featurizer requires one molecular component; this adapter does
not turn that off. A disconnected original input is an explicit source-contract
issue, never silently removed from the cohort. Multiple CM mask/replaceable
components within a connected molecule are a separate supported concept.

## Python interfaces used by the stage driver

```python
oracle = FrozenCMOracle.from_resolved(resolved_spec)
attr = oracle.attribute_train_parent({"parent_id": pid, "smiles": smi, "split": "train"})
request = attr.get("generation_request")  # absent for BEFORE_NOT_SOURCE
filtered = oracle.filter_generated(parent, completed_generation_record)
predictions = oracle.predict_rows(plain_rows, split="calibration")
wnode = FrozenCMWNode.from_resolved(resolved_spec)
encodings = wnode.encode_rows(plain_rows, featurizer=oracle.featurizer)
raw = raw_distance_record(left_encoding, right_encoding, numerical_contract=resolved_spec["resolved_wnode"])
pair = full_graph_pair(parent_prediction, prototype_prediction, raw_distance=raw)
```

The stage driver owns immutable pool selection, split authorization, compact
shards, missing-unit reuse and the global selector/test freeze boundary. This
module never opens split files or chooses a candidate pool. `filter_generated`
requires an explicit train parent and one of the generator's genuine scientific
terminals. A budgeted timeout cannot carry partial outputs.

`resolved_oracle` requires absolute `checkpoint_dir`, `dataset=bace`,
`backbone=gine`, `source_label=1`, `allowed_destinations=[0]`, `model_sha256`,
`temperature_sha256`, `feature_schema_sha256` (file hash), `label_map_sha256`.
`device`, `batch_size`, `node_layer` and existing forward numerical tolerances
are explicit dispatch fields. The original temperature file's actual fields
are `status=fit`, `selection_split=validation`, `num_examples=187`,
`test_used_for_fit=false`; there is no fabricated `fit_split` field.

HPC source located read-only on 2026-09-10:

```text
/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/inputs/bace-seed7-532e8373/reference/gine
model.pt: 4edd23cd206393dc2c424e4bfa1be46adf14a00ec36fa21d5ee3eaa5f591bc47
temperature_scaling.json: a3e48f0c31014c714a10292af804e28919867d3c50c51e83e4c2ed81c801dbef
T: 1.5447202081060156
feature_schema.json: 9a99e877b22cdcfccd67458bb233d3b3825f1dadf3db0cb337af8621d2dd6064
label_map.json: 80a44576b526fd76ce682181ca66b080441cdfce54c1ad47cb0bb9b23e0a7d7d
```

These are source-inventory locators, not a claim that this new experiment's
actual frozen-model Grad-CAM pilot has already passed.

## Original exact WNode, separately bound from oracle eligibility

`resolved_wnode` requires `molclr_root`, `molclr_ckpt`,
`molclr_checkpoint_sha256`, a task-private `node_emb_cache_dir`, fixed `device`,
`encoder_type=gin`, `feature_cost=cosine`, `node_mass=uniform`,
`size_penalty_beta=0.0`, and `numerical_contract_sha256`.

The adapter delegates node extraction to the existing
`MolCLRNodeEmbedder._compute_node_embeddings` and numerical OT to the existing
`compute_node_wasserstein_distance` (uniform float64 cosine costs, exact
`ot.emd2`). It adds no solver, approximate pruning, new SQLite writer or GINE
flip-derived distance cache. Unique encodings are returned as JSON-compatible
records for compact caller-owned shards. A stage can adopt successful sealed
encodings; unchanged successful graph units need not be re-encoded.

Each encoding binds complete graph identity, schema, ordered atom numbers,
float32 node features, extraction version, MolCLR checkpoint, numeric contract
and actual producer device/Torch/NumPy/RDKit/architecture. Raw symmetric pair
keys contain the **two complete encoding identities** and original numerical
contract. Producer conflicts fail; no smaller-value selection or averaging.
Actual raw distances are never capped. Strict-flip masking is a separate step
that requires the same GINE/temperature and verifies both raw graph identities.
Non-source base parents retain their denominator; missing computation raises
instead of becoming infinity. `None` is used only for explicit semantic failure
in JSON, to be converted to infinity by the metric reducer.

## Test and deployment boundary

`tests/test_cm_crem_oracle.py` includes real tiny GINE CPU forward/autograd,
parameter/BN/RNG immutability and atom-order fixtures. Numerical solver-injection
tests validate delegation and cache conflict behavior; they are not a production
POT/MolCLR or original BACE frozen-model execution receipt. The HPC pilot must
execute the actual bound original model and at least 64 non-self graph pairs.
No inference or OT is run on the login node. No checkpoint or source result is
modified by these adapters.

## Resolved HPC cohort inventory (not label-based reconstruction)

The local run artifact `/private/tmp/cm-crem-hpc-bindings-20260910.json` contains
the actual HPC oracle, MolCLR, threshold and split paths plus complete ordered
calibration/test IDs. It is an experiment artifact, not repository source.

- Full training CSV: 959 rows; original frozen proposal train manifest: 386 IDs.
  Actual source prediction is checked before attribution. The 386 count is not
  a claim that every one is predicted source by the current loaded GINE.
- Original calibration main cohort: 66 IDs, digest
  `2380d53fa985b8dd2f967cb622b8c2287c4c57c332e199088f9f49bb87162585`.
- Original test main cohort: 141 IDs, digest
  `6d588d9f48969ff24aad3488321c94d0d1801e221318c0cd34c0e9911eb4de80`.
- These IDs come from the eight original B11/B13 sealed verification manifests,
  not from filtering calibration/test labels to reach desired counts. Every
  shard uses `sorted(parent_id)_position_mod_4`; merging and re-sharding matched
  the source exactly. The eight small file hashes matched existing HPC
  selector/main evaluation manifest pins. No test CSV or pair results were
  opened for this resolution.
- Full calibration/test CSVs have 129/238 rows; the driver selects by the
  adopted ordered IDs only, without changing denominator based on a new
  prediction. The primary source/destination remains 1 -> 0.

The following read-only reproduction snippet uses the already resolved local
binding artifact as its pin inventory, reads only the eight small original
manifests on AutoDL, and emits a regenerated binding JSON to standard output.
It does not read test molecules, run an oracle, write remote files, or hash large
packages. Archive its output through the stage driver's normal immutable output
mechanism, rather than silently overwriting a sealed binding.

```bash
/Users/cz0210/miniconda3/envs/smiles_local/bin/python - <<'PY'
import hashlib, json, shlex, subprocess
from pathlib import Path

binding = json.loads(Path('/private/tmp/cm-crem-hpc-bindings-20260910.json').read_text())
requests = [dict(item, cohort=split)
            for split in ('calibration', 'test')
            for item in binding['data'][split]['source_parent_manifests']]
remote = '''import hashlib,json,pathlib,sys
out=[]
for item in json.load(sys.stdin):
    p=pathlib.Path(item['path']); raw=p.read_bytes(); obj=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==item['sha256'],str(p)
    assert obj['status']=='PASS' and obj['cohort']==item['cohort'],str(p)
    out.append({'path':str(p),'data':obj})
print(json.dumps(out))
'''
command = '/root/miniconda3/envs/smiles_pip118/bin/python -I -B -c ' + shlex.quote(remote)
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                         'autodl-a800', command], input=json.dumps(requests),
                        text=True, capture_output=True, check=True)
records = {r['path']: r['data'] for r in json.loads(result.stdout)}
digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()
for split in ('calibration', 'test'):
    target = binding['data'][split]
    manifests = [records[x['path']] for x in target['source_parent_manifests']]
    ids = sorted(i for item in manifests for i in item['parent_ids'])
    assert len(ids) == len(set(ids))
    assert digest(ids) == target['parent_ids_sha256']
    for shard, item in enumerate(manifests):
        assert item['oracle_checkpoint_hash'] == binding['resolved_oracle']['model_sha256']
        assert item['source_label'] == 1 and item['dataset'] == 'bace'
        assert item['split_identity']['sha256'] == target['sha256']
        assert item['shard_rule'] == 'sorted(parent_id)_position_mod_4'
        assert item['shard_index'] == shard and ids[shard::4] == item['parent_ids']
        assert item['all_parent_ids_sha256'] == digest(ids)
    target['parent_ids'] = ids
print(json.dumps(binding, sort_keys=True, indent=2))
PY
```

Generation runs in its own author-pinned CReM/RDKit environment and validates
its actual `PYTHONHASHSEED=0` state (`-s -B`, not `-I`, which ignores that env
setting). That change does not apply to this original GINE/WNode adapter's
isolated source-bootstrap execution; the scientific environments stay separate.
