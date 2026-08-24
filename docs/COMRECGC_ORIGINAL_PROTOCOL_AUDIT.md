# COMRECGC original-protocol audit

## Scope and verdict

This is a source-only audit of the ICML 2025/PMLR paper and the authors'
official implementation. It freezes the protocol that the AIDS exact route
must reproduce without changing an already completed generation.

The decisive finding is:

> The official implementation computes the full parent/candidate distance and
> embedding-difference grids, but passes **only rows whose normalized distance
> is `<= theta`** to DBSCAN. A physically complete Cartesian pair store is not
> the logical DBSCAN input.

Consequently, an exact engine run over all physical Cartesian rows is not an
exact reproduction of the official protocol unless every row is independently
proved theta-close. The production contract is
`dbscan_input = theta_close_recourse_vectors_only`.

The audit also found three material paper/code differences: the paper uses
`tau=0.05` while the current CLI/run script uses `0.1`; the paper pseudocode
uses a strict classifier boundary (`> 0.5`) while code uses `>= 0.5`; and the
official `common_recourse.py --cf_size` option is declared but never applied.

The machine-readable companion is
[`tests/fixtures/comrecgc_original_protocol_v1.json`](../tests/fixtures/comrecgc_original_protocol_v1.json).

## Frozen primary sources

| Source | Frozen identity |
|---|---|
| Paper | Fournier & Medya, *COMRECGC: Global Graph Counterfactual Explainer through Common Recourse*, ICML 2025, PMLR 267:17522--17541 ([PMLR record](https://proceedings.mlr.press/v267/fournier25a.html), [official PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/fournier25a/fournier25a.pdf)); PDF SHA256 `839ee4acd7f63334698d0a51a1ad2a9bc8368bd29f3aa58a2a684dd0f410ccc4` |
| Official code | [`ssggreg/COMRECGC`](https://github.com/ssggreg/COMRECGC), commit [`122f9341a360e9f06bb58a2f5823bb596021f6bf`](https://github.com/ssggreg/COMRECGC/tree/122f9341a360e9f06bb58a2f5823bb596021f6bf), committed 2025-05-23 |
| Current official HEAD | `git ls-remote ... HEAD` returned `122f9341a360e9f06bb58a2f5823bb596021f6bf` on 2026-08-24 UTC; it equals the project pin |
| Project integration base audited | `25a6a227327306e874eb89b35b9b106f25751740` |

The project does not vendor the upstream source. It pins the commit in
`src/baselines/comrecgc/contracts.py`, fetches it into ignored
`external/COMRECGC`, and permits an optional read-only `vendor_manifest.json`
whose commit and key-file SHA256 values must match. No upstream file was
modified during this audit.

Key upstream SHA256 values are:

| File | SHA256 |
|---|---|
| `comrecgc.py` | `921b9bfc1cc0e3efff90bf24bf9c7b754ea99563a62bba6d7197ede37785f90d` |
| `common_recourse.py` | `c5009ef5d73059dbea2d77e983a36a8140f1c2cca3b89664fec08f1ad7b4d6c5` |
| `util.py` | `6489a02e7a0d6498a5f9e7b1a9a4ebc137e3d26541bd2a605bff9f54b1cf74ce` |
| `run_experiments.sh` | `3378ea8ccc494047fc2ee4bcc5f54e61ada27748b5a37838326a9e0a125771c0` |

All audited upstream file hashes are recorded in the machine-readable fixture.

## Paper protocol

The paper's experimental setup (PDF page 6) freezes the following values for
AIDS:

| Field | Paper value | Evidence |
|---|---:|---|
| counterfactual threshold `theta` | `0.1` | Section 4.2 |
| common-recourse threshold `Delta` | `0.02` | Section 4.2 |
| heads `k` | `5` | Section 4.2 |
| walk steps `M` | `50,000` | Section 4.2 |
| teleport probability `tau` | `0.05` | Section 4.2 |
| summary budget `R` | `100` | Section 4.2 |
| candidate limit entering clustering | `100,000` | Section 4.3 |

Algorithm 2 says to take the top `n` frequently visited counterfactuals before
clustering. Algorithm 5 (PDF page 16) adds a visited state to the candidate set
when `phi(v_i) > 0.5`. The paper specifies DBSCAN and radius `Delta`, but does
not state `min_samples`, a distance metric, or an implementation-level
self-neighbour rule; those fields come from the official code and sklearn
semantics, not from the prose paper.

## Official implementation protocol

### Candidate construction and cap

The official generation CLI defaults are `theta=0.1`, `teleport=0.1`,
`steps=50000`, `heads=5`, `k=100000`, and `sample_size=10000`
([`comrecgc.py` lines 43--53](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/comrecgc.py#L43-L53)).
`--k` is assigned to `MAX_COUNTERFACTUAL_SIZE`
([lines 561--576](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/comrecgc.py#L561-L576)),
and the candidate registry is maintained using visit frequency
([lines 192--261](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/comrecgc.py#L192-L261)).

The common-recourse CLI separately declares `cf_size=100000`
([`common_recourse.py` lines 29--37](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/common_recourse.py#L29-L37)),
but `args.cf_size` has zero references. Candidate loading iterates the complete
saved registry and keeps every row satisfying `importance_parts[0] >= 0.5`;
there is no slice
([lines 199--208](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/common_recourse.py#L199-L208)).

Therefore the required cap audit fields are:

```text
candidate_cap_source=comrecgc.py --k -> MAX_COUNTERFACTUAL_SIZE
candidate_cap_applied=generation-time registry policy; not common_recourse.py --cf_size
candidate_count_before=<must be read from the actual saved generation payload>
candidate_count_after=<must be counted after >=0.5 predicate in frozen order>
```

`cf_size` must never be cited as evidence that the official downstream code
actually sliced the candidate pool. The generation code also contains
`bypass_size=True` call paths, so the strict payload bound and both actual
counts must be checked from the saved payload rather than inferred from the
default alone. A project-side post-predicate `cf_size` slice is a documented
`PROJECT_EXTENSION`, albeit one that enforces the paper's nominal limit.

### Theta-close dataflow and pair axes

The official dataflow is unambiguous
([`common_recourse.py` lines 210--249](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/common_recourse.py#L210-L249)):

1. Form candidate graphs after the `>= 0.5` classifier predicate.
2. Compute the candidate-by-parent NeuroSED distance grid.
3. Divide each distance by
   `element_count(candidate) + element_count(parent)`, where upstream
   `graph_element_counts` is `num_nodes + num_edges / 2`.
4. Transpose to parent-by-candidate orientation and select
   `idxs = where(normalized_distance <= theta)`.
5. Compute the full candidate-major embedding-difference grid, but index it by
   those theta-close pairs before fitting DBSCAN.

For every selected pair:

```text
idxs row = (parent_index, candidate_index)
recourse_vector =
    (embedding(candidate) - embedding(parent))
    / (element_count(candidate) + element_count(parent))
filter_operator = <=
```

The flattening expression
`parent_index + candidate_index * num_parents` confirms candidate-major,
parent-minor physical order while preserving `(parent, candidate)` columns.
It also proves why a full Cartesian physical store cannot be sent directly to
DBSCAN: `rec = flat_diffs[linear_idx]` occurs before `db.fit(rec)`.

### DBSCAN

The official call is:

```python
DBSCAN(eps=args.delta, min_samples=args.cluster_size).fit(rec)
```

Thus the pinned-code defaults are Euclidean metric, `eps=0.02`,
`min_samples=3`, sklearn's inclusive epsilon comparison, and sklearn's
self-neighbour counting. `algorithm` is not specified, so sklearn's `auto`
backend selection applies. HDBSCAN is imported but never used in this path.

An exact replacement may change the execution strategy, but it must preserve
the sklearn partition on the theta-close rows. Approximate neighbours,
HDBSCAN, or clustering the unfiltered Cartesian grid are not equivalent.

### Centroid, radius, coverage, and greedy semantics

Official `coverage_summary`
([lines 69--145](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/common_recourse.py#L69-L145))
does all of the following after DBSCAN:

- computes the arithmetic mean of every non-noise cluster;
- keeps a member for coverage only when its distance to that centroid is
  strictly `< delta`;
- admits a cluster to greedy selection only when its centroid norm is
  strictly `< theta`;
- greedily chooses at most `R=100` clusters by maximum marginal parent
  coverage.

Python's insertion-ordered dictionaries and `max` make the first encountered
cluster label the effective tie-break. A deterministic replacement should make
that cluster-ID tie-break explicit.

There is an upstream output-naming trap. The greedy function stores
`selected[rank] = (cluster_label, cumulative_covered_parent_count)`. Later,
`coverage_summary` calls `selected[rank][1]` a “counterfactual ID” and appends
it to `covering`; it is not a counterfactual ID. Project output must use
separate fields for cluster ID, rank, cumulative covered count, representative
candidate IDs, and centroid norm.

## Paper versus current official code

| Field | Paper | Official current code / `run_experiments.sh` | Consequence |
|---|---:|---:|---|
| `tau` | `0.05` | CLI default `0.1`; run script does not override | Current official command runs at `0.1`; completed project generation must not be silently relabelled as paper-`tau` or regenerated merely to change it. |
| candidate probability boundary | `> 0.5` in Algorithm 5 | `>= 0.5` | Freeze the code boundary for implementation reproduction and report the paper difference. |
| candidate cap | top 100,000 counterfactuals entering clustering | `--k` controls the generation registry; downstream `--cf_size` is unused | Audit actual before/after counts. Do not infer cap application from CLI presence. |
| generation `theta` | defines counterfactual closeness; Appendix D.2 also describes a `3 theta / 2` walk constraint | `comrecgc.py` declares `--theta` but never reads `args.theta` | `theta` is effective in `common_recourse.py` close-pair and centroid filters, not in the official generation file. |
| DBSCAN `min_samples` / metric | not specified | `3` / Euclidean defaults | These are pinned-code reproduction fields. |

[`run_experiments.sh`](https://github.com/ssggreg/COMRECGC/blob/122f9341a360e9f06bb58a2f5823bb596021f6bf/run_experiments.sh)
only overrides `theta=0.15` for PROTEINS; it does not override teleportation.

## Project frozen defaults versus actual AIDS run

At integration base `25a6a227`, the source-level full contract is:

```text
generation:
  theta=0.1 teleport=0.1 steps=50000 heads=5
  candidate_capacity=100000 sample_size=10000 seed=0
common recourse:
  theta=0.1 delta=0.02 recourse_size=100
  cf_size=100000 cluster_size=3 seed=0
upstream_commit=122f9341a360e9f06bb58a2f5823bb596021f6bf
```

These are defaults, not proof of what a production payload used. This local,
source-only audit did not open the AutoDL run manifest. The deployment audit
must separately extract and hash the actual AIDS values for `tau`, `theta`,
`delta`, heads, `M`, candidate-cap source/application, predicate, and embedding
checkpoint. An absent actual field remains `UNKNOWN`; it must not be filled
from defaults.

## Required production gates

Before an AIDS exact route may claim original-protocol equivalence:

1. verify the pinned checkout commit and key-file hashes above;
2. count candidates before and after the inclusive classifier predicate and
   state whether a project-side cap changed the count;
3. prove `(col0, col1) = (parent, candidate)` independently;
4. recompute normalized distance and recourse vectors from the same frozen
   NeuroSED/GREED checkpoint;
5. materialize a hashed `<= theta` logical view or bitmap;
6. require `DBSCAN_INPUT_COUNT == logical_close_pair_count`;
7. reproduce exact DBSCAN, centroid, strict-radius, strict-centroid-norm,
   coverage, and greedy decisions;
8. label any enforced downstream cap or output-field repair as a
   `PROJECT_EXTENSION` rather than “official implementation identical”.

Only the source/protocol audit is closed here. Production pair counts,
checkpoint identities, certificates, and full-run PASS markers belong to the
immutable AutoDL execution audit.
