# COMRECGC Baseline

## Scope

This adapter reproduces the pinned ICML 2025 COMRECGC implementation in two
deliberately separate routes:

- `native_reproduction`: official TU datasets, official checkpoints, and
  official label semantics. These outputs are diagnostic and cannot enter
  project paper figures.
- `project_adapted`: project AIDS/HIV or Mutagenicity graphs, frozen project
  GNNs, project GREED/NeuroSED embeddings, common-recourse graph medoids, the
  unified RF oracle, and MolCLR-Node-Wasserstein strict-flip evaluation.

The upstream repository does not publish a clear redistribution license. Its
source is therefore fetched at commit
`122f9341a360e9f06bb58a2f5823bb596021f6bf` into ignored
`external/COMRECGC`; no upstream source is vendored or modified.

## Data Identity Gate

`scripts/baselines/comrecgc/audit_dataset_identity.py` records both identities:

- official AIDS: `torch_geometric.datasets.TUDataset(name="AIDS")` processed
  by upstream code;
- project AIDS: `data/raw/AIDS/HIV.csv` through the frozen project graph and
  source-parent artifacts;
- official Mutagenicity: upstream filtered TU Mutagenicity;
- project Mutagenicity: frozen strict train-source graphs.

The audit also freezes the 1283-parent AIDS and 217-parent Mutagenicity final
evaluation cohorts. Calibration/test data are not loaded during generation.

## Algorithm Boundary

The project imports the clean pinned checkout at runtime. It reuses upstream:

- edit-map neighbor construction;
- transition probabilities and frequency reinforcement;
- dynamic teleportation;
- DBSCAN common-recourse clusters;
- greedy coverage-gain ordering.

Project-owned compatibility code injects project datasets/models, maps project
labels to upstream `source=0,target=1`, batches model calls without changing
their order or formula, and carries source-node lineage. Each embedding cluster
is exported as the real source-to-counterfactual pair nearest its center; an
embedding center is never represented as a fictional graph.

Native smoke is complete only after the actual official random-walk payload is
reloaded, model-counterfactual graphs traverse the official DBSCAN coverage
path, and the official greedy summary serialization completes. An empty
cluster/medoid set is a valid scientific result, not an engineering failure.
The audit artifacts are `native_common_recourse.json` and
`native_representative_counterfactuals.pt`; TU outputs remain ineligible for
project figures.

## Frozen Parameters

| Stage | Smoke | Full |
|---|---:|---:|
| generation steps | 100 | 50000 |
| heads | 2 | 5 |
| candidate capacity | 200 | 100000 |
| neighbor sample size | 128 | 10000 |
| theta / teleport | 0.1 / 0.1 | 0.1 / 0.1 |
| common recourses | 5 | 100 |
| CF pool | 200 | 100000 |
| delta / min cluster | 0.02 / 2 | 0.02 / 3 |
| seed | 0 | 0 |

Smoke values validate interfaces only and are never reported as final results.
Native smoke uses a fixed 64-parent diagnostic cohort for both TU datasets.
The earlier 32-parent native probe was sufficient for random-walk serialization
but produced no AIDS common-recourse cluster; 64 remains within the predefined
smoke interface range and is the only native clustering retry.
The first 50-step/64-sample smoke reached the clustering interface but yielded
only one representative per dataset: the Mutagenicity graph was chemically
invalid and the AIDS graph was RF non-target. The single permitted smoke
budget retry therefore uses the documented upper smoke bounds of 100 steps
and 128 samples; full parameters and all scientific thresholds remain
unchanged.
Full values match the pinned implementation defaults, but they do not match
every value stated in the paper. In particular, the paper reports
`tau=0.05`, while the pinned CLI defaults to `0.1` and the official
`run_experiments.sh` does not override it. The official `--cf_size` option is
also declared but unused; the effective nominal bound originates from
generation-time `--k`, while the project adapter's post-predicate `cf_size`
slice is a project extension. See
`docs/COMRECGC_ORIGINAL_PROTOCOL_AUDIT.md`. Completed generation parameters
remain immutable and must be read from the actual run manifest rather than
silently replaced by either source's defaults.

## Slurm and Automation

All jobs use one A800, at most seven CPUs, and submit through
`scripts/exp_sbatch.sh`. The recoverable driver is:

```bash
python scripts/automation/run_comrecgc_baseline.py \
  --datasets aids,mutagenicity \
  --mode smoke
```

The authorization-scoped recovery driver freezes the complete job DAG before
submission. Full nodes require the matching dataset's completed engineering
smoke Gate and use only Slurm `afterok` dependencies. Under the separately
recorded end-to-end authorization, a successful smoke Gate may promote its own
dataset chain without a second prompt; an engineering failure blocks only that
dataset. The driver writes state, events, evidence paths, job records, and short
reports under `outputs/hpc/automation/comrecgc_recovery/<run_id>/`. Status
refresh is explicit; there is no long-lived local poll loop.

Full project generation writes selected-transition events to bounded JSONL
chunks and stores only a compact candidate lineage index. Chemistry reconstructs
one candidate lineage at a time and retains graph objects only for the immutable
official medoid ranks. Smoke artifacts keep the legacy inline lineage schema for
backward compatibility. Neither format changes RNG calls, neighbor ordering,
candidate payloads, importance values, DBSCAN inputs, or greedy rank.

## Unified Evaluation

Candidate export preserves every official common-recourse rank slot. The
deterministic chemistry policy marks invalid medoids unavailable without
compaction or backfill; only repaired-valid medoids are sent to RF and WNode.
Final metadata uses:

```text
method = COMRECGC
adaptation_mode = common_recourse_cluster_medoid_fullgraph
cf_mode = strict_flip
distance_line = MolCLR-Node-Wasserstein
candidate_set_preselected = true
selection_performed_in_eval = false
```

The existing project evaluator computes all WNode distances and CCRCOV values.
