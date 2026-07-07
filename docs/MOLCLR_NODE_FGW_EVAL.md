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

- SMILES -> MolCLR node embedding matrix `H`;
- SMILES -> normalized structure distance matrix `D`;
- pairwise MolCLR Node-FGW distances.

Default cache paths:

```text
outputs/hpc/cache/molclr_node_embeddings/
outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite
```

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
