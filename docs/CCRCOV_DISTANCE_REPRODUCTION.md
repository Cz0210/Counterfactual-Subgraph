# CCRCov Distance Reproduction: GREED-GED and MolCLR-Embedding

This document defines the current distance-evaluation workflow for close counterfactual coverage on the canonical AIDS/HIV dataset. The goal is to evaluate Ours and the GT-FullGraph proxy baseline under the same CCRCov protocol without sending all fullgraph pairs to exact NetworkX GED.

## Motivation

The previous close-CF sweep could become intractable when a fullgraph baseline was evaluated with pairwise NetworkX GED. A run with roughly 1283 parents and 2000 fullgraph candidates creates millions of parent-candidate graph edit distance calls. Exact or near-exact NetworkX GED is not suitable for that scale.

The project therefore keeps NetworkX GED only as a small debug option and adds two scalable distance lines:

- **GREED-GED**: train a GREED-style neural graph distance model on AIDS/HIV graph pairs and use its predicted normalized GED inside CCRCov.
- **MolCLR-Embedding**: encode parent, hard-deletion residual, and GT-FullGraph candidate molecules with a pretrained MolCLR GIN encoder; use \(1-\cos(h_G,h_{G^a})\) as distance.

## Why GREED

GCFExplainer cites a GREED-style neural GED approximation for scalable graph distance estimation. We reproduce the same spirit for this repository: a Siamese GIN-style encoder maps each molecule graph to an embedding \(Z_G\), and the predicted normalized GED is:

\[
\hat d_{\mathrm{GREED}}(G_a,G_b)=\|Z_{G_a}-Z_{G_b}\|_2.
\]

The model is trained on AIDS/HIV graph pairs with normalized GED-style labels. Deletion pairs use exact deletion edit cost. Fullgraph/random pairs avoid default NetworkX exact GED; when GEDLIB is unavailable, the current implementation uses a bounded atom/bond count approximation and records the label source.

## AIDS/HIV Dataset Contract

The current workflow targets the canonical AIDS/HIV dataset defined in `docs/DATASET_CONTRACT.md`. In this project, `hiv` and `hiv_quick` are legacy internal names for the same raw CSV, while `aids` is the official graph-baseline dataset key used by CLEAR / GCF-style adapters. They must trace back to:

Canonical raw CSV:

```text
data/raw/AIDS/HIV.csv
```

Canonical columns:

```text
SMILES_COLUMN=smiles
LABEL_COLUMN=HIV_active
TARGET_LABEL=1
```

Project-generated prompt CSVs may use `label` after preprocessing, but final results must still record the canonical source and label column. `ogbg_molhiv` is engineering validation only and must not be reported as the final AIDS/HIV baseline result.

## GREED-GED Data Generation

The GREED line writes:

```text
outputs/hpc/greed_hiv/dataset/graphs.jsonl
outputs/hpc/greed_hiv/pairs/{train,val,test}_pairs.csv
outputs/hpc/greed_hiv/pairs/{train,val,test}_pairs_labeled.csv
outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt
outputs/hpc/greed_hiv/reports/train_metrics.json
outputs/hpc/greed_hiv/reports/test_metrics.csv
```

Pair types:

- `ours_deletion`: \(G\) vs hard-deletion residual \(G \setminus s\).
- `gt_fullgraph`: \(G\) vs selected GT-FullGraph proxy candidate \(C\).
- `random_hiv_pair`: two HIV molecules sampled from the prepared graph pool.

Label sources:

- `deletion_exact`: exact deletion edit cost for hard-deletion residuals.
- `bounded_count_approx`: scalable atom/bond count approximation for fullgraph/random pairs when GEDLIB is unavailable.
- `networkx_timeout_debug`: explicit small debug mode only.
- `failed`: label could not be produced.

## MolCLR-Embedding Line

MolCLR code and checkpoints remain external runtime assets. The project does not download them. Prefer passing:

```bash
MOLCLR_ROOT=/path/to/MolCLR
MOLCLR_CKPT=/path/to/pretrained_gin/checkpoint.pth
```

If these are not provided, `scripts/precompute_molclr_embeddings_for_ccrcov.py` searches common project/HPC locations such as `baselines/MolCLR`, `external/MolCLR`, sibling `MolCLR`, `outputs/hpc`, and `checkpoints`.

The precompute job writes:

```text
outputs/hpc/molclr_ccrcov_embeddings/parents.jsonl
outputs/hpc/molclr_ccrcov_embeddings/ours_residuals.jsonl
outputs/hpc/molclr_ccrcov_embeddings/gt_fullgraph_candidates.jsonl
outputs/hpc/molclr_ccrcov_embeddings/all_embeddings.jsonl
outputs/hpc/molclr_ccrcov_embeddings/summary.json
```

Distance is:

\[
d_{\mathrm{MolCLR}}(G,G^a)=1-\cos(h_{\mathrm{MolCLR}}(G),h_{\mathrm{MolCLR}}(G^a)).
\]

## Smoke Run Order

GREED smoke:

```bash
sbatch scripts/slurm/greed_hiv_prepare_label1.sh
sbatch scripts/slurm/greed_hiv_label_pairs_smoke.sh
sbatch scripts/slurm/greed_hiv_train_smoke.sh
TEACHER_PATH=/path/to/teacher.pkl sbatch scripts/slurm/greed_hiv_eval_ccrcov_smoke.sh
```

MolCLR smoke:

```bash
MOLCLR_ROOT=/path/to/MolCLR MOLCLR_CKPT=/path/to/ckpt.pth sbatch scripts/slurm/molclr_hiv_precompute_embeddings_smoke.sh
TEACHER_PATH=/path/to/teacher.pkl sbatch scripts/slurm/molclr_hiv_eval_ccrcov_smoke.sh
```

## Full Run Order

GREED full:

```bash
sbatch scripts/slurm/greed_hiv_prepare_label1.sh
sbatch scripts/slurm/greed_hiv_label_pairs_full.sh
sbatch scripts/slurm/greed_hiv_train_full.sh
TEACHER_PATH=/path/to/teacher.pkl sbatch scripts/slurm/greed_hiv_eval_ccrcov_full.sh
```

MolCLR full:

```bash
MOLCLR_ROOT=/path/to/MolCLR MOLCLR_CKPT=/path/to/ckpt.pth sbatch scripts/slurm/molclr_hiv_precompute_embeddings_full.sh
TEACHER_PATH=/path/to/teacher.pkl sbatch scripts/slurm/molclr_hiv_eval_ccrcov_full.sh
```

Comparison:

```bash
sbatch scripts/slurm/compare_greed_vs_molclr_ccrcov.sh
```

## Output Interpretation

Each evaluation line reports per-method `details.csv`, `threshold_summary.csv`, `report.md`, and figures.

Important columns:

- `close_only_coverage`: fraction of parents with some valid action under the distance threshold.
- `close_cf_coverage`: CCRCov, requiring threshold satisfaction and teacher flip or sufficient CFDrop.
- `avg_best_distance` / `median_best_distance`: cost among covered parents.
- `flip_rate_among_covered`: teacher flip rate for best covered actions.
- `avg_cf_drop_among_covered`: mean drop in original-label teacher probability.
- `SuppCov`: support coverage proxy for subgraph/deletion methods; not a fullgraph main metric.
- `StructRed`: average Morgan/Tanimoto redundancy among selected action candidates.
- `CovRed`: average Jaccard overlap among candidate coverage sets.
- `ValidRate`: fraction of attempted interventions with valid teacher and distance outputs.

Key reporting thresholds:

- GREED-GED: `0.05`, `0.10`, `0.20`.
- MolCLR-Embedding: `0.02`, `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, `0.30`.

The GT-FullGraph baseline in this workflow is a project proxy baseline. It should not be described as official GCFExplainer.
