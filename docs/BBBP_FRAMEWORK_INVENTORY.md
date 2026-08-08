# BBBP Framework Inventory

Status: `FRAMEWORK_ONLY_NOT_RUN`

This inventory was prepared before implementation and updated after framework
validation. It records the committed code used as the baseline, the modules
now implemented on the feature branch, and dataset-specific inputs that still
have to be frozen before a future experiment can run.

## Frozen development inputs

- Base branch: `baseline/bace-common4`
- Base commit: `5e542a31e422b7c9309492db0ac992eb6d0ac740`
- Latest reviewed COMRECGC memory-fix commit:
  `c0fcfb16381a88f1f67956fbd7cb644764f0f9ad`
- Development branch:
  `feature/bbbp-common4-generalization-ablations-20260808-233949`
- Local worktree: `/private/tmp/bbbp-framework-20260808-233949`
- HPC worktree:
  `/share/home/u20526/czx/worktrees/bbbp-framework-20260808-233949`

The running BACE, AIDS, and Mutagenicity worktrees are not development inputs
and must remain untouched.

## Directly reusable modules

### Dataset and teacher

- `src/data/bace_adapter.py` provides deterministic molecular normalization,
  explicit graph serialization, stable molecule/graph hashes, and four-way
  scaffold-group splitting.
- `src/models/bace_rf_teacher.py` reuses the established Morgan-RF primitives,
  trains on train only, selects on validation only, and materializes
  teacher-consistent views after model selection.
- `src/models/mutagenicity_rf_teacher.py` owns the shared feature extraction,
  model-selection, prediction, and metric primitives.

### Ours

- `scripts/generate_full_candidate_pool.py` remains the ChemLLM candidate
  generator. BBBP must inject checkpoint, cohort, decoding, and seed settings;
  no AIDS path may be embedded in BBBP code.
- `src/data/bace_candidate_lineage.py` demonstrates the required lineage-only
  enrichment: preserve row order and scientific fields, add stable IDs and
  hashes, and fail on parent disagreement.
- Existing candidate audit and `src/eval/class_counterfactual_selector.py`
  remain the scientific implementations.
- `scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py` remains the shared
  strict-flip evaluator.

### Paper artifacts

- `src/eval/bace_paper_artifacts.py` fixes the current plotting schemas:
  Figure 3 `method,k,coverage,cost`; Figure 4
  `method,threshold,coverage`; Table 2
  `method,k,coverage,cost,flip_rate,cf_drop`.
- `scripts/audit_bace_paper_artifacts.py` demonstrates fail-closed method,
  parent-universe, threshold, teacher, MolCLR, monotonicity, and schema gates.
- BBBP will use a dataset-specific exporter/auditor with those exact schemas;
  no plotting compatibility conversion is permitted after evaluation.

### GCFExplainer

- The committed BACE route is the canonical six-stage template:
  prepare -> official GNN -> official VRRW -> native greedy summary/export ->
  shared WNode evaluation -> artifact audit.
- `src/baselines/gcfexplainer_bace_adapter.py` and
  `src/baselines/gcfexplainer_bace_runtime.py` keep project data adaptation
  outside `baselines/gcfexplainer_official/`.
- BBBP requires its own frozen atom/channel schema and checkpoint compatibility
  audit. The BACE nine-channel NeuroSED projection must not be assumed valid
  for BBBP before the BBBP train/validation vocabulary is known.

### GlobalGCE

- `scripts/baselines/globalgce/run_globalgce_wrapper.py` copies the pinned
  official source into an isolated run root and records provenance.
- `scripts/baselines/globalgce/export_globalgce_outputs.py` exports native
  rule/counterfactual artifacts without modifying their order.
- The current graph-to-molecule label maps in
  `src/baselines/globalgce_adapter.py` are AIDS-specific. BBBP therefore needs
  an explicit dataset conversion and label-map manifest before native output
  can enter the common evaluator; silently reusing AIDS maps is forbidden.

### COMRECGC

- The project route is dataset preparation -> native full generation ->
  transition gate -> common-recourse/export -> shared WNode evaluation ->
  artifact audit.
- The reviewed memory-safe implementation at `c0fcfb16` supplies exact action
  replay, compact numeric transition storage, a bounded expanded-graph LRU,
  chunked model calls, streaming trace artifacts, and memory diagnostics.
- BBBP will expose chunk size, batch size, checkpoint interval, resume intent,
  and memory cap in its config. It must not claim cross-job random-walk resume
  unless all RNG and transition state can actually be restored.

## Implemented framework modules

- BBBP raw CSV normalization with explicit alias resolution, reason-coded
  invalid rows, deterministic IDs, and separate main/cross-scaffold protocols.
- Shared split, scaffold, candidate-lineage, and threshold-source leakage gates.
- BBBP RF wrapper and validate-only plan.
- BBBP-specific Ours lineage wrapper and four-method stage wrappers.
- GlobalGCE BBBP dataset conversion/label manifest.
- GCFExplainer BBBP schema/profile validation layer.
- COMRECGC BBBP dataset identity/config layer using the committed bounded-cache
  mechanism.
- Dataset-agnostic held-out and cross-scaffold protocol manifests.
- Candidate-source generators, selector variants, nested budget prefixes,
  multi-seed aggregation, and parent-level bootstrap confidence intervals.
- A plan-only CLI that emits registered future commands but cannot submit.

All items above are implemented in the isolated BBBP worktree. Native
GlobalGCE, GCFExplainer, and COMRECGC stage execution remains deliberately
fail-closed until their BBBP-specific input/checkpoint manifests are supplied;
the wrappers and full dependency DAGs are present and validate without running.

## Stage DAGs

### Ours

`prepare -> teacher -> candidate generation -> lineage persistence -> candidate audit -> top-20 selector -> WNode strict-flip evaluation -> artifact audit`

### GlobalGCE

`prepare -> native run -> native rule/candidate export -> frozen frequency summary -> WNode strict-flip evaluation -> artifact audit`

### GCFExplainer

`prepare -> official GNN -> official VRRW -> official native summary/export -> WNode strict-flip evaluation -> artifact audit`

### COMRECGC

`prepare -> native full generation -> transition gate -> common-recourse/export -> WNode strict-flip evaluation -> artifact audit`

## Scientific semantics that must not change

- `cf_mode=strict_flip`
- `distance_line=MolCLR-Node-Wasserstein`
- uniform node mass, cosine node cost, exact EMD2, and zero size penalty
- candidate/rule selection is complete before test evaluation
- thresholds are fit on calibration only and frozen for every method
- Figure 3 uses nested prefixes of one frozen order
- Figure 4 uses the shared frozen threshold grid
- Table 2 uses K=10
- evaluator-side selection is false
- fullgraph baselines remain fullgraph; they are not converted into Ours
  delete-only actions

## Current blockers for future execution

- `data/raw/BBBP/bbbp.csv` has not been asserted present and will not be
  downloaded by this framework.
- Raw BBBP column names must be resolved explicitly if more than one supported
  alias is present.
- BBBP-specific GCFExplainer feature vocabulary and NeuroSED compatibility are
  unknown until the prepared train/validation data are audited.
- The official GlobalGCE repository must support, or be adapted through a
  project-owned converter for, the frozen BBBP graph representation.
- ChemLLM SFT/PPO checkpoint paths remain configuration inputs; the framework
  will not silently fall back to an AIDS-specific checkpoint.
- Slurm resources are intentionally `null`/`NOT_RUN` in framework configs until
  future preflight resolves them from proven cluster jobs.

No experiment, model, candidate pool, distance computation, registry write, or
Slurm submission is performed by this inventory or the framework task.
