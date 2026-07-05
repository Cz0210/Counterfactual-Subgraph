# Dataset Contract: AIDS/HIV

This document is the repository-level source of truth for AIDS/HIV dataset naming, paths, labels, and final baseline reporting. It prevents accidental mixing between legacy internal names, official graph-baseline keys, and engineering validation datasets.

## Canonical Dataset

- Canonical dataset name: AIDS/HIV
- Canonical raw CSV: `data/raw/AIDS/HIV.csv`
- HPC absolute path: `/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv`
- SMILES column: `smiles`
- Label column: `HIV_active`
- Target label: `1`

## Verified Schema And Counts

- CSV columns used by this project:
  - `smiles`
  - `activity`
  - `HIV_active`
- Label distribution:
  - `HIV_active=0`: 39684
  - `HIV_active=1`: 1443
  - total rows: 41127

## Historical Naming

- `hiv` / `hiv_quick`: legacy internal names for the same raw AIDS/HIV CSV at `data/raw/AIDS/HIV.csv`.
- `aids`: official graph-baseline dataset key for CLEAR, GCF-style baselines, GlobalGCE, and other graph-format or pickle-format adapters derived from the same raw CSV.
- `ogbg_molhiv`: engineering validation only. It must not be reported as the final AIDS/HIV baseline result.

## Fairness Rule

All final baseline results must be traceable to `data/raw/AIDS/HIV.csv` unless they are explicitly marked as engineering validation. Different baseline adapters may use different internal dataset keys, but the final paper-facing and group-meeting-facing comparison must share:

- the same canonical raw source;
- the same SMILES column;
- the same label column;
- the same target label;
- the same final split/evaluation subset definition;
- the same teacher/oracle type, or an explicitly documented teacher type when an adapter cannot use the historical SMILES/RF oracle directly.

## Forbidden Reporting

Do not report `ogbg_molhiv` CLEAR results as AIDS/HIV final baseline results. `ogbg_molhiv` runs are useful for wrapper, Slurm, and export smoke validation only.

## Required Metadata For Final Results

Every final CCRCOV / CFDrop / FlipRate / Cost / Redundancy result must record:

```text
DATASET_SOURCE=data/raw/AIDS/HIV.csv
SMILES_COLUMN=smiles
LABEL_COLUMN=HIV_active
TARGET_LABEL=1
BASELINE_DATASET_KEY=<hiv|aids|...>
TEACHER_PATH or TEACHER_KIND
CF_MODE=strict_flip
```

The historical SMILES/RF oracle path is:

```text
outputs/hpc/oracle/aids_rf_model.pkl
```

CLEAR full-graph strict evaluation may use `TEACHER_KIND=clear_graphpred` or another graph teacher adapter when the counterfactual output is a continuous graph tensor rather than a SMILES molecule. That teacher must be explicitly recorded and must not be described as the SMILES/RF oracle.

For the final fair cross-baseline table, CLEAR must additionally provide a
`CLEAR-RF-FullGraph` path whenever its graph arrays can be conservatively
converted back to valid SMILES. That path uses:

```text
TEACHER_PATH=outputs/hpc/oracle/aids_rf_model.pkl
PARENT_DATASET_CSV=outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
SMILES_COLUMN=smiles
LABEL_COLUMN=label
TARGET_LABEL=1
CF_MODE=strict_flip
```

If CLEAR graph arrays cannot be reliably converted to RF-readable SMILES, the
audit must report `rf_oracle_usable=false`; `clear_graphpred` native diagnostics
must not be substituted for the RF-oracle final table.
