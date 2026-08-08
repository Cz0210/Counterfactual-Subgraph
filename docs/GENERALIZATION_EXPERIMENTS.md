# BBBP Generalization Experiments

Status: `FRAMEWORK_ONLY_NOT_RUN`

## Cross-scaffold protocol

`src/data/scaffold_split.py` computes RDKit Bemis-Murcko scaffolds and performs
a deterministic four-way group split. A scaffold may occur in exactly one of
train, validation, calibration, and test. The default policy for acyclic
molecules is `canonical-smiles`, which uses each canonical molecule as its
group key. `--acyclic-policy group` is the explicit alternative.

Teacher fitting uses train and validation only. Candidate/rule discovery uses
train and validation, selector and thresholds use calibration, and final
metrics use unseen-scaffold test only. Scaffold overlap is a hard error, not a
warning. Reports reserve teacher held-out AUROC, CCRCov, cost, CFDrop,
FlipRate, ValidRate, StructRed, CovRed, scaffold counts, and unseen scaffold
rate. The implementation is dataset-agnostic for AIDS, Mutagenicity, BACE,
and BBBP; only the BBBP plan is frozen here.

Planned root:
`outputs/hpc/eval/generalization/bbbp/cross_scaffold_v1/<method>/`.

## Held-out molecule protocol

This is an inductive protocol, not a renamed existing test result:

1. teacher fit and candidate/rule discovery on train/validation;
2. selector tuning and threshold fitting on calibration;
3. freeze the ordered Top-K and threshold manifest;
4. load held-out test only for final metrics.

Test molecules may not influence candidate generation, size matching,
fragment frequency, selector configuration, thresholds, or hyperparameters.
`src/eval/heldout_molecule_protocol.py` and the split/candidate audits enforce
those restrictions fail-closed.

Planned root:
`outputs/hpc/eval/generalization/bbbp/heldout_molecule_v1/<method>/`.
The combined report is `transductive_vs_heldout_summary.csv/json` and compares
standard, held-out-molecule, and cross-scaffold protocols without changing
their metrics.

Both protocols keep `strict_flip`, `MolCLR-Node-Wasserstein`, calibration-only
thresholds, and the existing Figure 3/4/Table 2 schemas.
