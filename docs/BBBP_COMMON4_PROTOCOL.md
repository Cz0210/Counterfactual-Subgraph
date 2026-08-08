# BBBP Common4 Protocol

Status: `FRAMEWORK_ONLY_NOT_RUN`

## Inputs and identity

- Raw data: `data/raw/BBBP/bbbp.csv` (never downloaded implicitly).
- Supported SMILES aliases: `smiles`, `mol`, `canonical_smiles`.
- Supported label aliases: `p_np`, `label`, `target`.
- Ambiguous aliases are fatal. Use `--smiles-col`/`RAW_SMILES_COL` and
  `--label-col`/`RAW_LABEL_COL` to select an explicit source column.
- Normalized data live under `data/processed/BBBP/` and include deterministic
  molecule IDs, canonical SMILES, graph records, split manifests, and leakage
  audits.

The main protocol uses a deterministic four-way train/validation/calibration/
test split. Candidate discovery is limited to train and validation, selector
tuning and thresholds are calibration-only, and test is final evaluation only.
Molecule ID and canonical-SMILES overlap across splits is a hard failure.

## Frozen scientific contract

- Source label: 1; target label: 0.
- Counterfactual mode: `strict_flip`.
- Distance: `MolCLR-Node-Wasserstein` with the existing project evaluator.
- Candidate selection in evaluation: false.
- Threshold fitting on test: false.
- Figure 3: one frozen order, nested prefixes `K=1..20`.
- Figure 4: the shared calibration-derived threshold grid.
- Table 2: `K=10`.

The plotting schemas are written directly by evaluation:

- Figure 3: `method,k,coverage,cost`
- Figure 4: `method,threshold,coverage`
- Table 2: `method,k,coverage,cost,flip_rate,cf_drop`

## Four method DAGs

1. Ours: prepare -> teacher -> ChemLLM generation -> lineage persistence ->
   candidate audit -> calibration selector -> WNode evaluation -> audit.
2. GlobalGCE: prepare -> official native run -> ordered rule/candidate export
   -> frozen frequency summary -> WNode evaluation -> audit.
3. GCFExplainer: prepare -> official GNN -> official VRRW -> official native
   greedy summary/export -> WNode evaluation -> audit.
4. COMRECGC: prepare -> official native generation -> transition gate ->
   common-recourse export -> WNode evaluation -> audit.

Global/fullgraph methods remain fullgraph methods. They are never relabeled as
Ours-style delete-only actions. The COMRECGC plan imports the reviewed bounded
transition cache, chunking, streaming trace, and memory diagnostics from commit
`c0fcfb16381a88f1f67956fbd7cb644764f0f9ad`; it does not claim unsafe random-
walk resume.

## Planned artifacts

The future root is `outputs/hpc/eval/paper/bbbp_common3_standardized_v1/`.
Despite the historical root name, the manifest requires all four methods:
`Ours`, `GlobalGCE`, `GCFExplainer`, and `COMRECGC`. Each method must provide
the three plotting CSVs plus `summary.json`, `run_manifest.json`,
`protocol_manifest.json`, `split_manifest.json`, `split_leakage_audit.json`,
`candidate_lineage_audit.json`, and `final_artifact_audit.json`.

No formal BBBP artifact has been produced by this framework change.
