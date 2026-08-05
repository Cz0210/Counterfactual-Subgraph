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

## Frozen Parameters

| Stage | Smoke | Full |
|---|---:|---:|
| generation steps | 50 | 50000 |
| heads | 2 | 5 |
| candidate capacity | 200 | 100000 |
| neighbor sample size | 64 | 10000 |
| theta / teleport | 0.1 / 0.1 | 0.1 / 0.1 |
| common recourses | 5 | 100 |
| CF pool | 200 | 100000 |
| delta / min cluster | 0.02 / 2 | 0.02 / 3 |
| seed | 0 | 0 |

Smoke values validate interfaces only and are never reported as final results.
Full values match the published implementation unless a future, documented
resource Gate requires a protocol revision.

## Slurm and Automation

All jobs use one A800, at most seven CPUs, and submit through
`scripts/exp_sbatch.sh`. The recoverable driver is:

```bash
python scripts/automation/run_comrecgc_baseline.py \
  --datasets aids,mutagenicity \
  --mode smoke
```

Full chains require `--after-smoke-pass`; Slurm `afterok` dependencies prevent
generation from running when a smoke Gate fails. The driver writes state,
events, evidence paths, job records, and short reports under
`outputs/hpc/automation/comrecgc/<run_id>/`. Status refresh is explicit; there
is no long-lived local poll loop.

## Unified Evaluation

Candidate export preserves official common-recourse rank, filters graph/RDKit
invalid and RF non-target candidates in that order, and never reranks by RF or
WNode. Final metadata uses:

```text
method = COMRECGC
adaptation_mode = common_recourse_cluster_medoid_fullgraph
cf_mode = strict_flip
distance_line = MolCLR-Node-Wasserstein
candidate_set_preselected = true
selection_performed_in_eval = false
```

The existing project evaluator computes all WNode distances and CCRCOV values.
