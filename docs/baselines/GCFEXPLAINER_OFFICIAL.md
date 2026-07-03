# Official GCFExplainer Baseline

This document records the project-owned wrapper for the official
GCFExplainer repository. The official baseline is a **full counterfactual graph
set** method. It is different from the project proxy named GT-FullGraph, and
GT-FullGraph results must not be reported as official GCFExplainer results.

## Source Layout

Preferred runtime source:

```text
third_party/GCFExplainer
```

If that directory is not present, set:

```bash
GCF_OFFICIAL_REPO=/path/to/GCFExplainer
```

For backward compatibility with older local checkouts, the wrappers also look
at:

```text
baselines/gcfexplainer_official
```

The official repository is treated as a runtime asset. Project adapters live in
`src/`, `scripts/`, and `scripts/slurm/`; large official checkpoints and
outputs are not copied into project source.

## Asset Check

```bash
python scripts/gcf_official_check_assets.py \
  --official-repo "$GCF_OFFICIAL_REPO" \
  --out-dir outputs/hpc/gcfexplainer_official/asset_check
```

The check expects the AIDS GNN and NeuroSED assets used by the official
pipeline, including `data/aids/gnn/model_best.pth`, `preds.pt`, `logits.pt`,
`graph_embeddings.pt`, and `data/aids/neurosed/best_model.pt`.

## Official Reproduction

Smoke:

```bash
sbatch scripts/slurm/gcf_official_aids_smoke.sh
```

Full VRRW for one alpha:

```bash
GCF_ALPHA=0.5 sbatch scripts/slurm/gcf_official_aids_vrrw_full.sh
```

For the paper-facing alpha sweep, submit separate jobs for `GCF_ALPHA=0`,
`GCF_ALPHA=0.5`, and `GCF_ALPHA=1`, then export and evaluate each run:

```bash
GCF_ALPHA=0.5 sbatch scripts/slurm/gcf_official_aids_summary_export_eval.sh
```

All outputs are written under:

```text
outputs/hpc/gcfexplainer_official
```

The wrappers run official `vrrw.py` in an isolated per-run workdir so parallel
alpha jobs do not overwrite `results/aids/runs/counterfactuals.pt`.

## Native Evaluation

Native official evaluation uses:

- `GCF_MODE=official_native`
- `TEACHER_TYPE=official_gnn`
- `DISTANCE_TYPE=official_native`
- `CF_MODE=strict_flip`

The strict counterfactual condition is:

```text
distance(G, C) <= theta and official_gnn(C) != label
```

This path uses the official GNN predictions and official NeuroSED distance. It
does **not** use NetworkX GED for large fullgraph pairwise evaluation.

## Adapted HIVCSV Path

The project also maintains a separate adapted path named
`GCFExplainer-HIVCSV` / `GCFExplainer-adapted-HIVCSV`. This is not the paper
TUDataset AIDS reproduction. It uses the canonical project CSV:

```text
data/raw/AIDS/HIV.csv
```

with `SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`, and
`TARGET_LABEL=1`. The adapted path must not trigger TUDataset AIDS downloads.

The adapted path is:

```text
HIV.csv -> RDKit -> PyG graphs.pt -> HIVCSV GNN teacher -> GCF-style VRRW -> native fullgraph close-CF evaluation
```

Prepare the graph dataset:

```bash
sbatch scripts/slurm/gcf_hiv_csv_prepare_dataset_smoke.sh
sbatch scripts/slurm/gcf_hiv_csv_prepare_dataset_full.sh
```

The HIVCSV GNN teacher is trained by:

```bash
sbatch scripts/slurm/gcf_hiv_csv_train_gnn_smoke.sh
sbatch scripts/slurm/gcf_hiv_csv_train_gnn_full.sh
```

Because `HIV_active` is highly imbalanced, the training script uses
stratified train/validation/test splits and class-weighted cross entropy by
default. The class weights are computed from the training split as inverse
label frequency:

```text
weight_c = total_train / (num_classes * count_c)
```

`gnn_train_summary.json` reports overall accuracy, per-class
precision/recall/F1, macro-F1, balanced accuracy, ROC-AUC, prediction counts,
and a warning if label-1 recall is low. Accuracy alone is not sufficient for
this baseline.

Run the adapted VRRW and native evaluation:

```bash
sbatch scripts/slurm/gcf_hiv_csv_vrrw_smoke.sh
sbatch scripts/slurm/gcf_hiv_csv_summary_eval_smoke.sh
```

Full run:

```bash
sbatch scripts/slurm/gcf_hiv_csv_vrrw_full.sh
sbatch scripts/slurm/gcf_hiv_csv_summary_eval_full.sh
```

The adapted native evaluator records:

- `method=GCFExplainer-HIVCSV`
- `GCF_MODE=hiv_csv_adapted`
- `DATASET_SOURCE=HIV_CSV`
- `TEACHER_TYPE=hiv_csv_gnn`
- `CF_MODE=strict_flip`

It uses a lightweight normalized edit proxy for smoke/full plumbing and does
not use NetworkX GED for large pairwise evaluation. GREED-GED can be connected
later as the learned distance line.

If `GCF_TOP_K_LIST` contains commas in an HPC submission, prefer:

```bash
export GCF_TOP_K_LIST=1,5,10,20,50,100
sbatch --export=ALL scripts/slurm/gcf_hiv_csv_summary_eval_full.sh
```

## Graph-to-SMILES Diagnostic

The official AIDS graph format does not always preserve enough atom/bond
mapping information for safe RDKit reconstruction. The graph-to-SMILES path is
therefore diagnostic only:

```bash
sbatch scripts/slurm/gcf_official_graph_to_smiles_rf_eval.sh
```

It writes conversion rates and reasons such as
`missing_atom_or_bond_mapping` or `rdkit_sanitize_failed`. If enough candidates
convert to valid SMILES, the diagnostic can run the project RF oracle at
`outputs/hpc/oracle/aids_rf_model.pkl`; this does not replace the native
official evaluation.

## GREED-GED and MolCLR Diagnostics

When official fullgraph candidates are available as SMILES diagnostics, they
can be passed through the existing GREED-GED or MolCLR CCRCOV scripts:

```bash
sbatch scripts/slurm/greed_hiv_eval_gcf_official_full.sh
sbatch scripts/slurm/molclr_hiv_eval_gcf_official_full.sh
```

These scripts use the existing distance pipelines and preserve
`CF_MODE=strict_flip`. Low SMILES conversion rate should be reported as a
diagnostic limitation.

## Collect Best Alpha

```bash
sbatch scripts/slurm/gcf_official_aids_collect_best.sh
```

The collector selects the alpha with highest native coverage at `K=10` and
`theta=0.1` and writes:

```text
outputs/hpc/gcfexplainer_official/final/gcf_official_best_summary.json
outputs/hpc/gcfexplainer_official/final/gcf_official_all_runs.csv
```

## Reporting Rules

- Report the method name as official GCFExplainer only for this official
  pipeline.
- Do not call GT-FullGraph official GCFExplainer.
- Final official-native results must record `CF_MODE=strict_flip`.
- Graph-to-SMILES-to-RF is a diagnostic, not the native official result.
- Do not use NetworkX GED for large fullgraph GCFExplainer evaluation.
