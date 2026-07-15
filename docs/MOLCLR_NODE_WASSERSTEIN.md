# MolCLR-Node-Wasserstein CCRCov

## Role

MolCLR-Node-Wasserstein (`node_wasserstein`) is the primary learned distance
line for unified CCRCov. MolCLR-Node-FGW with `lambda=0.5` remains the explicit
structure-aware ablation. Neither distance changes training, PPO, selectors,
candidate generation, or external Top20 order.

MolCLR node states combine atom features with learned context inside the GIN
encoder's finite receptive field. They therefore contain learned local topology,
but they are not an explicit representation of global shortest-path structure.
The extractor builds PyG nodes by iterating the same RDKit atom order and rejects
any embedding/atom-count mismatch before caching.

## Definition

For MolCLR node embedding matrices `H_A` and `H_B`, each row is L2-normalized
and the transport cost is

```text
M_ij = 1 - cosine(H_A[i], H_B[j]).
```

Both molecules use uniform node mass. The implementation computes exact optimal
transport with POT:

```python
a = np.full(n_a, 1.0 / n_a, dtype=np.float64)
b = np.full(n_b, 1.0 / n_b, dtype=np.float64)
distance = ot.emd2(a, b, M.astype(np.float64))
```

No shortest-path matrix, Gromov-Wasserstein term, Sinkhorn approximation, or
networkx GED is used. An optional size penalty is

```text
beta * abs(n_a - n_b) / max(n_a, n_b),
```

with `beta=0.0` by default.

## Caches

Node embeddings use the shared v2 cache:

```text
outputs/hpc/cache/molclr_node_embeddings/
```

Its key contains canonical SMILES, checkpoint identity, encoder type, extraction
version, architecture identity, and feature schema. It deliberately excludes
distance type, `structure_mode`, FGW lambda, feature cost, and size penalty.
On a v2 miss, the loader may read and atomically migrate a valid historical
Node-FGW `shortest_path_unweighted` NPZ without deleting the old file.

WNode pair distances use a separate symmetric namespace and SQLite file:

```text
outputs/hpc/cache/distance_cache/molclr_node_wasserstein_v1.sqlite
```

The pair key includes the canonical pair, WNode namespace, checkpoint identity,
node cache schema, feature cost, node mass, beta, and `exact_emd2` solver.

## Threshold Protocol

Calibration is Ours-only:

```bash
RUN_OURS=1 RUN_FULLGRAPH=0 MAX_PARENTS=0 \
WNODE_THRESHOLDS=auto_quantile \
OUTPUT_DIR=outputs/hpc/eval/wnode_calibration_ours \
sbatch scripts/slurm/molclr_node_wasserstein_eval_ccrcov.sh
```

Use the Ours thresholds at reference quantiles
`0.05,0.10,0.20,0.30,0.50,0.70,0.90` as one explicit absolute grid for every
final method. A baseline-only run must therefore set `WNODE_THRESHOLDS` to that
comma-separated grid; it cannot derive method-specific auto quantiles.

Tiny and smoke runs adjust `MAX_PARENTS`. Final runs use `MAX_PARENTS=0`, one
preselected Top20 input, a method-specific pair-cache DB, and:

```text
CF_MODE=strict_flip
PRESELECTED_TOPK=20
REQUIRE_PRESELECTED_TOPK=1
SKIP_REDUNDANCY=1
RESUME=1
```

## Resume

The evaluator atomically writes a method partial detail CSV and resume JSON.
The JSON stores the config fingerprint and completed `(parent_id,candidate_id)`
pairs. Resume refuses changed datasets, candidates, checkpoint, teacher, feature
cost, beta, or other fingerprinted inputs. Completed pairs are skipped and are
not duplicated. `_RUN_COMPLETE.json` marks a successful finalization.

## Reports

`generate_gcf_style_recourse_report.py` reads existing pair details only. It
collapses multiple match instances by `(parent_id,candidate_id)`, keeps the
minimum finite strict-flip distance and its CFDrop, then applies the external
candidate prefix. It never ranks candidates by WNode.

For the requested GCF-style conditional-cost view, pass:

```bash
--distance-label MolCLR-Node-Wasserstein \
--figure3-cost-stat conditional_median \
--table-cost-stat conditional_median
```

Figure 4 bootstraps parents on one shared absolute threshold grid.

## WNode vs FGW Ablation

After all four methods exist for both lines, run:

```bash
python scripts/compare_node_distance_ablation.py \
  --wnode-run Ours=/path/to/wnode/ours \
  --wnode-run GlobalGCE=/path/to/wnode/globalgce \
  --wnode-run CLEAR=/path/to/wnode/clear \
  --wnode-run GCFExplainer=/path/to/wnode/gcfexplainer \
  --fgw-run Ours=/path/to/fgw/ours \
  --fgw-run GlobalGCE=/path/to/fgw/globalgce \
  --fgw-run CLEAR=/path/to/fgw/clear \
  --fgw-run GCFExplainer=/path/to/fgw/gcfexplainer \
  --output-dir outputs/hpc/eval/ablations/wnode_vs_fgw
```

Cross-distance comparison is aligned by Ours-reference quantile. Absolute
WNode and FGW distance values are not directly comparable.
