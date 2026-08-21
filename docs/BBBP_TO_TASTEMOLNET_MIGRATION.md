# BBBP to TasteMolNet migration

Date: 2026-08-22

## Scope

The active four-dataset experiment matrix is now intended to be:

1. AIDS;
2. Mutagenicity;
3. BACE;
4. TasteMolNet.

This is an additive registry migration. Historical BBBP data, manifests, logs,
paper drafts, and Git history are not deleted or rewritten. At the migration
base commit there was no tracked runtime BBBP registry; the only tracked BBBP
mention was historical. Any untracked paper material is treated as pre-existing
user work and must be preserved while its active tables are updated separately.

## TasteMolNet contract

- internal dataset ID: `tastemolnet`;
- aliases: `taste`, `bst`, `bitter_sweet_tasteless`;
- display name: `TasteMolNet`;
- task: three-class graph classification;
- labels: `0=Bitter`, `1=Sweet`, `2=Tasteless`;
- source class: `1=Sweet`;
- counterfactual mode: untargeted strict flip.

For a correctly classified Sweet parent, either `1 -> 0` or `1 -> 2` is a
valid strict flip. TasteMolNet must not be reduced to Sweet versus non-Sweet.
Every evaluation records the destination label and the destination
distribution.

## Data provenance status

No user-local TasteMolNet CSV was present. The approved fallback is the public
repository `MujeebOnawole/Taste_Prediction_RGCN`, fixed at commit
`16af8ead8a17b6bd3941d9eb5879c5be75c14114`. Its processed scaffold CSV is an
`upstream_processed` input, not a prediction-site scrape. The repository has no
standalone license file, so the data must remain untracked and the foundation
state is `LICENSE_REVIEW_REQUIRED` until reuse terms are confirmed. No raw or
processed TasteMolNet records are committed to this repository.

## Split and leakage policy

The project creates its own deterministic 70/10/10/10 scaffold-disjoint
train/validation/calibration/test split from the single-label canonicalized
input. Cross-label canonical duplicates are excluded rather than majority
voted. Test is never used for checkpoint selection, temperature fitting,
threshold calibration, selector fitting, or prefix ordering.

## Model-role boundary

TasteMolNet uses a task-specific frozen three-class GNN classifier. ChemLLM is
only a proposal generator; MolCLR is only the WNode distance encoder. Neither
may be presented as the TasteMolNet classifier. RF backends are forbidden for
the active BACE and TasteMolNet routes.
