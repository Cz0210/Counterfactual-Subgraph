# Mutagenicity Dataset

## Canonical task

The project-wide benchmark name is **Mutagenicity**. Its binary labels are:

- `label=1`: mutagenic;
- `label=0`: non-mutagenic.

The main counterfactual direction is `1 -> 0`: start from a mutagenic molecule
and seek recourse toward the non-mutagenic class. Every downstream method must
use the same processed benchmark and split manifest.

## Sources

SMILES files:

- `data/raw/Mutagenicity/smiles/smiles_mutagenicity_raw.csv`
- `data/raw/Mutagenicity/smiles/smiles_mutagenicity_curated.csv`
- `data/raw/Mutagenicity/smiles/smiles_mutagenicity_removed.csv`

TU graph files are under
`data/raw/Mutagenicity/tudataset/Mutagenicity/` and include adjacency, edge
labels, graph indicators, graph labels, and node labels. File integrity is
defined by `data/raw/Mutagenicity/SHA256SUMS` and is checked with Python's
`hashlib`, so validation behaves the same on macOS and Linux.

The raw CSV contains 4,337 rows. The external curated CSV contains 4,247 rows,
with 90 rows recorded in the removed CSV. The curated CSV is the clean
benchmark input; raw CSV and TU files are retained for provenance, graph-order
alignment, and label-mapping audit.

## Label mapping

The CSV uses `mutagenicity=1` for mutagenic and `0` for non-mutagenic. The TU
readme uses the inverse encoding (`0=mutagen`, `1=nonmutagen`). Full row-wise
audit therefore requires:

```text
project_label = 1 - tu_graph_label
```

The source audit tests identity, inverted 0/1, and both -1/1 mappings. It fails
unless one complete mapping is uniquely best.

## Cleaning contract

Each curated SMILES is parsed and sanitized with RDKit. The benchmark stores
canonical isomeric SMILES and uses it as the deduplication key. Canonical
non-isomeric SMILES is retained only for audit. The pipeline drops parse or
sanitize failures, disconnected/multi-component molecules, dummy atoms, empty
molecules, and zero-heavy-atom molecules.

The first version deliberately does **not** neutralize charges, canonicalize
tautomers, or remove stereochemistry. Formal charge, components, atom and bond
types, molecular size, molecular weight, and Bemis-Murcko scaffold are recorded
without changing the molecule.

Same-label canonical duplicates collapse to one stable molecule with all source
row IDs retained. Opposite-label canonical duplicates are written to
`mutagenicity_conflicts.csv` and excluded. `molecule_id` is a deterministic
SHA256-derived identifier and does not depend on DataFrame or CSV sort order.

## Split contract

The split ratios are train/validation/calibration/test = 70/10/10/10 with seed
42. Assignment is deterministic and label-aware, but the indivisible unit is a
Bemis-Murcko scaffold group. The empty Murcko scaffold is treated as one
explicit acyclic group. Exact ratios are not preferred over scaffold isolation.

- train: model fitting;
- validation: training-time model selection;
- calibration: threshold and operating-point calibration;
- test: final held-out reporting.

No molecule ID, canonical SMILES, or scaffold may cross splits. All four splits
must contain both labels and their union must exactly equal the clean benchmark.

## Official graph benchmark distinction

Some GCFExplainer material refers to an official TU Mutagenicity benchmark with
approximately 4,308 graphs. That graph-native version is not silently treated
as this project's final comparison cohort. The project main line uses the
unified clean SMILES benchmark produced here, and Ours, GCFExplainer,
GlobalGCE, CLEAR, and other baselines must be rerun or adapted on this same
benchmark before entering a fair table.

## Commands and outputs

Verify only:

```bash
python scripts/data/verify_mutagenicity_download.py \
  --root data/raw/Mutagenicity \
  --manifest data/raw/Mutagenicity/SHA256SUMS
```

HPC smoke:

```bash
sbatch scripts/slurm/mutagenicity_preprocess_smoke.sh
```

HPC full:

```bash
sbatch scripts/slurm/mutagenicity_preprocess_full.sh
```

Smoke results are isolated under
`outputs/hpc/datasets/mutagenicity_v1_smoke/`. Full run artifacts are under
`outputs/hpc/datasets/mutagenicity_v1_full/`; only after validation succeeds are
the processed files copied to `data/processed/Mutagenicity/v1/`.

The canonical directory contains the master, clean, dropped, duplicate,
conflict and ID-map CSVs; train/val/calibration/test CSVs; the split manifest;
and preprocessing, split, and validation summaries/reports.

