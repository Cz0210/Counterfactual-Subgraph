# MolCLR Node-FGW CCRCOV Evaluation

## Motivation

The existing MolCLR CCRCOV line uses graph-level pooled embeddings and
`1 - cosine` distance. On some AIDS/HIV sweeps this graph-level distance can
become too coarse or saturated, making threshold interpretation difficult.

`molclr_node_fgw` is an auxiliary distance line for CCRCOV evaluation only. It
uses the MolCLR pretrained GIN encoder to extract a node-level embedding matrix
for each molecule, then compares two molecules with Fused
Gromov-Wasserstein distance. Node-level embeddings preserve local atom context,
while FGW compares both node features and graph structure.

This line does not modify:

- training loss;
- PPO;
- selector logic;
- candidate generation;
- GREED-GED evaluation;
- graph-level MolCLR evaluation.

## Distance

For molecules \(G\) and \(H\), let:

- \(X_G, X_H\) be MolCLR node embedding matrices;
- \(D_G, D_H\) be normalized shortest-path structure distance matrices;
- \(C_{ij} = 1 - \cos(X_{G,i}, X_{H,j})\) be node feature cost;
- \(T\) be a transport plan between atoms.

The FGW objective is:

\[
\operatorname{FGW}_{\lambda}(G,H)
= \min_T (1-\lambda)\langle T, C\rangle
+ \lambda \sum_{i,j,k,l} L(D_G(i,k), D_H(j,l))T_{ij}T_{kl}
\]

where \(L\) is square loss.

Interpretation:

- `lambda=0`: node-level Wasserstein, only node feature cost;
- `lambda=1`: pure Gromov-Wasserstein, only structure cost;
- `lambda=0.5`: balanced feature + structure comparison, default.

## Implementation

Main files:

```text
src/eval/node_fgw_distance.py
src/eval/distance_cache.py
scripts/evaluate_ccrcov_with_molclr_node_fgw.py
scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
```

The evaluator caches:

- SMILES -> MolCLR node embedding matrix `H` in the shared, structure-independent
  v2 node cache;
- normalized structure distance matrix `D` in process memory for the selected
  `structure_mode`;
- pairwise MolCLR Node-FGW distances.

Default cache paths:

```text
outputs/hpc/cache/molclr_node_embeddings/
outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite
```

The shared node embedding key does not contain `structure_mode` or FGW lambda.
The SQLite pairwise cache uses symmetric sha256 keys, so `(A,B)` and `(B,A)`
share one cached distance.

## Thresholds

Do not reuse graph-level MolCLR thresholds such as `0.02,0.05,...,0.30` by
default. FGW is a different distance scale.

The smoke evaluator defaults to:

```text
FGW_THRESHOLDS=auto_quantile
FGW_QUANTILES=0.05,0.10,0.20,0.30,0.50,0.70,0.90
```

The evaluator first computes finite pair distances, then writes quantile-based
thresholds to:

```text
outputs/hpc/eval/ccrcov_molclr_node_fgw_smoke/distance_quantiles.csv
```

Explicit thresholds are also supported:

```bash
FGW_THRESHOLDS=0.01,0.02,0.05 sbatch scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
```

## GlobalGCE Fullgraph Selection

When a Node-FGW evaluation contains both ours and a GlobalGCE fullgraph pool,
select GlobalGCE candidates only from rows with `method=globalgce`:

```bash
python3 scripts/select_fullgraph_candidates_by_fgw_coverage.py \
  --pair-details outputs/hpc/eval/ccrcov_molclr_node_fgw_medium_globalgce_lam05/details/pair_details.csv \
  --candidates-csv /path/to/globalgce_top2000_candidates.csv \
  --out-dir outputs/hpc/selectors/globalgce_node_fgw_top20 \
  --top-k 20 \
  --threshold-quantile 0.2 \
  --method-name globalgce
```

The selector uses strict-flip close pairs only (`cf_flip=true` and
`distance <= threshold`) and greedily maximizes marginal parent coverage. It
prefers the evaluator's saved `distance_quantiles.csv` for a requested
quantile, then computes the quantile from valid GlobalGCE distances if needed.
`selected_top20_for_eval.csv` is directly usable as a fullgraph-candidate CSV
for the Node-FGW evaluator; it labels the method as `GlobalGCE` and
`fullgraph_method=globalgce_selected20`.

## Preselected Ours Subgraphs

`REQUIRE_PRESELECTED_TOPK=1` supports both fullgraph CSV inputs and an Ours
selector directory containing `selected_subgraphs.csv` or
`selected_subgraphs.json`. For Ours, the evaluator validates the exact unique
fragment count and rank order, reads `selector_summary.json` when available,
and preserves selector output order.

Candidate selection and subgraph matching are separate stages:

```text
external selector -> fixed Top20 fragments
Node-FGW evaluation -> parent matching -> residual construction -> match-instance distances
```

Multiple valid `match_index` rows for one parent-fragment pair do not mean that
the evaluator selected candidates. Ours summaries therefore record:

```text
candidate_set_preselected=true
selection_performed_in_eval=false
evaluation_row_unit=match_instance
```

They also distinguish `num_unique_parent_candidate_pairs`, `num_detail_rows`,
and `num_valid_match_instances`, while retaining `total_pairs` for backwards
compatibility.

For an Ours-only full calibration, set:

```bash
RUN_OURS=1 RUN_FULLGRAPH=0 \
OURS_SELECTED_PATH=outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20 \
PRESELECTED_TOPK=20 REQUIRE_PRESELECTED_TOPK=1 \
MAX_PARENTS=0 MAX_CANDIDATES=20 \
sbatch scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
```

## Outputs

Default smoke output root:

```text
outputs/hpc/eval/ccrcov_molclr_node_fgw_smoke
```

Important files:

```text
details/pair_details.csv
combined/combined_threshold_summary.csv
distance_quantiles.csv
run_config.json
cache_stats.json
```

`combined_threshold_summary.csv` records:

- `distance_type=node_fgw`;
- `distance_line=MolCLR-Node-FGW`;
- `fgw_lambda`;
- `structure_mode`;
- `feature_cost`;
- `atom_penalty`;
- close-only coverage;
- close-CF coverage;
- best-distance and CFDrop summaries;
- pair distance cache hit rate;
- node embedding cache hit rate;
- `skip_redundancy=true`.

## Scope

`molclr_node_fgw` is only an embedding-matrix-based auxiliary CCRCOV distance
line. It does not compute `StructRed`, `CovRed`, or pairwise candidate
redundancy. It should not replace GREED-GED as the main GED-style line.

## Smoke Run

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
```

With experiment tracking:

```bash
scripts/exp_sbatch.sh \
  --name "label1 MolCLR node FGW smoke CCRCOV" \
  --tags "label1,ccrcov,node-fgw,molclr,smoke" \
  --notes "FGW lambda=0.5 with MolCLR node embeddings; CCRCOV only; redundancy skipped" \
  --expected-output-root "outputs/hpc/eval/ccrcov_molclr_node_fgw_smoke" \
  -- scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
```

If POT is missing:

```bash
pip install POT
# or
conda install -c conda-forge pot
```

## Local Checks

```bash
python3 -m py_compile src/eval/node_fgw_distance.py
python3 -m py_compile src/eval/distance_cache.py
python3 -m py_compile scripts/evaluate_ccrcov_with_molclr_node_fgw.py
bash -n scripts/slurm/molclr_node_fgw_eval_ccrcov_smoke.sh
python3 scripts/evaluate_ccrcov_with_molclr_node_fgw.py --help
```
