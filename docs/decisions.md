# Decisions Log

This file records major design decisions for the counterfactual subgraph v3 project.

It should be updated whenever a meaningful implementation, algorithmic, or interface decision is made.

---

## [2026-07-23] Continue exactly one existing PEFT adapter on Mutagenicity

### Background

The Mutagenicity continued-SFT smoke run completed successfully, but PEFT
warned that the model already had a `peft_config` and could receive multiple
adapters. The previous runtime check only required at least one trainable
parameter, so it could not prove that the ChemLLM base was unwrapped or that
exactly one existing AIDS adapter was active.

### Decision

Load the adapter configuration from the AIDS `checkpoint-500`, verify that its
declared ChemLLM base matches the requested base model, load that base without
an adapter, and invoke `PeftModel.from_pretrained(..., is_trainable=True)`
exactly once. Before training, require exactly one configured and active LoRA
adapter, positive trainable LoRA parameters, and zero trainable non-LoRA base
parameters. Persist this result as `adapter_audit.json` and include it in the
training log and report.

### Consequences

- Continued SFT updates the recovered AIDS LoRA weights rather than creating a
  second random adapter.
- Existing or ambiguous adapter state on the requested base fails before
  Trainer construction.
- Data, optimization hyperparameters, PPO, selectors, WNode, baselines, and
  evaluation semantics are unchanged.

### Status

Accepted

---
## [2026-07-24] Isolate stable-PPO validation generation RNG

### Background

The ChemLLM/InternLM2 PEFT stack rejects a `torch.Generator` passed through
`model.generate()`, because this Transformers version validates it as an
unused model keyword. Mutagenicity stable-PPO therefore failed during its
first validation pass after completing a successful rollout and update.

### Decision

Do not forward `generator` to validation `generate()`. Derive a deterministic
seed from the run seed, validation step, and batch index, then run only that
validation batch inside `torch.random.fork_rng`. Seed CPU and available CUDA
generators inside the context so their previous states are restored afterward.
Leave PPO rollout generation and all training, reward, teacher, sampling, and
optimization logic unchanged.

### Consequences

- Validation remains reproducible for the same seed, step, and batch order.
- Validation sampling does not advance the RNG state used by later PPO
  rollouts.
- The fix is shared by Mutagenicity and AIDS/HIV stable-PPO validation.

### Status

Accepted

---
## [2026-07-23] Adapt the shared stable PPO loop to Mutagenicity 1 -> 0

### Background

The AIDS stable decoded-chemistry PPO loop already implements generation,
chemistry validation, parent projection, deletion-based RF scoring, PPO
clipping, adaptive KL, and value-head updates. Its update counter is
DataLoader-batch based, however, and its generic counterfactual scorer defines
a flip only from the post-intervention prediction. Mutagenicity requires an
auditable source-label `1` to target-label `0` direction and one complete
no-replacement pass over 1,448 train parents.

### Decision

Keep `run_stable_decoded_chem_ppo_loop()` as the only PPO algorithm and add an
optional run observer plus stable `molecule_id` propagation. The
Mutagenicity adapter deterministically orders the teacher-correct train view,
uses a nominal rollout batch of 64, derives `ceil(1448 / 64) = 23` updates,
and stops after the first exhausted DataLoader pass. Load the pure ChemLLM
base plus exactly one continued-SFT `checkpoint-200` LoRA for both policy and
frozen reference policy. Use a Mutagenicity-specific scorer with strict flip
`pred_before == 1 and pred_after == 0` and `cf_drop = p1_before - p1_after`.

### Consequences

- Existing AIDS callers do not pass the observer and retain their current
  sampling and reward behavior.
- The decoded stable loop still does one optimizer update per PPO epoch for
  each rollout batch; its legacy mini-batch and gradient-accumulation CLI
  values do not subdivide this local loop.
- Parent coverage, update metrics, candidates, validation samples, model
  freezing, teacher direction, and checkpoint selection are persisted as
  first-class run artifacts.
- Calibration and test data are rejected before model loading.

### Status

Accepted

---

## [2026-07-15] Plot and report theta-covered conditional FGW cost by default

### Background
The GCF-style report CSV already contained the correct theta-covered
conditional median cost, but Figure 3 still plotted the unconditional
`median_cost`. This made the plotted GlobalGCE K=10 cost exceed the displayed
theta even though the correctly conditioned value was available.

### Decision
Make `theta_covered_conditional_median_cost` the explicit default for Figure 3
and final Table 2. Record the selected Figure 3 value as `plotted_cost`, leave
zero-coverage points as NaN, and assert that every finite default plotted/table
cost is no greater than theta. Emit separate K=1..10 and K=1..20 Figure 3 files.
The final `table2_global_recourse` artifact contains only method, coverage, and
theta-covered conditional median cost; legacy audit-oriented fields remain in
the compatibility CSV.

### Consequences
- Figure 3 and Table 2 now use the same strict-close conditioning event as
  coverage at the stated theta.
- Applicable-parent and unconditional medians remain available only through
  explicit metric parameters or audit columns.
- Figure 4 coverage, strict flip, parent cohort, candidate order, and all saved
  Node-FGW distances are unchanged.

### Status
Accepted

---
## [2026-07-23] Continue the AIDS SFT-v3 adapter on Mutagenicity

### Background

Mutagenicity now has fixed teacher-consistent SFT train/validation data, but
the repository had no training entrypoint for it. The stable AIDS
`checkpoint-500` is a PEFT LoRA adapter; treating it as a complete base model
or resuming its completed optimizer/global step would give incorrect
continued-training semantics.

### Decision

Load the same 4-bit ChemLLM base used by AIDS SFT-v3, attach the stable AIDS
adapter with `is_trainable=True`, and start a fresh Mutagenicity Trainer state.
Retain the AIDS learning rate, batch/accumulation, scheduler, warmup, bf16, and
500-step full schedule. Make the previously implicit prompt/completion
supervision explicit: prompt tokens are masked with `-100`, while completion
and retained EOS tokens participate in causal-LM loss. Validate the complete
1,317/250 train/validation contract before any smoke sampling and never load
calibration or test.

### Consequences

- Continued SFT inherits learned AIDS adapter weights without silently falling
  back to another model or random LoRA initialization.
- Tokenization, truncation, parent coverage, checkpoint selection, and
  generation sanity checks are persisted as auditable artifacts.
- Validation loss selects checkpoints; calibration and test cannot influence
  training or selection.
- PPO, reward, selector, teacher, WNode, baselines, and unified evaluation code
  remain unchanged.

### Status

Accepted

---

## [2026-07-15] Require explicit parent-cohort inputs in saved FGW audits

### Background
The GlobalGCE Frequency-Top20 audit populated an empty `--comparison-run`
list with production output paths. As a result, an otherwise self-contained
unit test could discover an unrelated 1283-parent Ours run and fail before it
audited its two temporary parents.

### Decision
Treat comparison runs and reference cohorts strictly as data inputs: only open
them when supplied explicitly through `--reference-parent-ids`,
`--comparison-run Ours=...`, `--reference-ours-run`, or
`--auto-reference-from-ours`. Without one of these inputs, audit the current
run as an all-label-parent diagnostic and emit a warning. Explicit reference
CSVs are already in the current GlobalGCE ID namespace; explicit Ours-run
references are mapped into that namespace by a one-to-one canonical-SMILES
crosswalk before missing-ID validation.

### Consequences
- Unit tests and ad hoc audits no longer depend on files under `outputs/hpc`.
- Final 1283-parent audits retain their exact explicit crosswalk behavior.
- No MolCLR embedding, FGW distance, strict-flip, ranking, coverage, or Table 2
  calculation changes.

### Status
Accepted

---

## [2026-07-14] Use one explicit parent-ID cohort and corrected strict flip for final FGW reports

### Background
The raw GlobalGCE Frequency-Top20 run contains 1443 label parents, while the
final Ours reference cohort contains 1283 parents. Historical GlobalGCE pair
details also recorded the old weak flip (`pred_after != target_label`). Counting
mismatches before filtering by method mixed unrelated rows into the audit.

### Decision
Define the final comparison cohort by the exact `parent_id` set from the final
Ours run's `details/pair_details.csv`, or by an explicit reference-parent CSV.
Filter every method by this set before strict-flip, cost, prefix-K, threshold,
or bootstrap aggregation. Extra raw parents may be discarded by ID; missing
reference IDs are fatal, even when the raw parent count happens to match.

Recompute teacher-strict flip from saved `label`, `pred_before`, and
`pred_after`, retaining the old weak field only for audit. This correction is
post-processing only and reuses every saved Node-FGW distance. Final Table 2
reports only coverage and theta-covered conditional median cost at the exact
requested theta; the latter is asserted not to exceed theta.

### Consequences
- Raw 1443-parent GlobalGCE output remains an all-label-parent diagnostic.
- Final figures and tables use the same 1283 parent IDs across all methods.
- Historical weak-flip pair details can be corrected without loading MolCLR,
  running POT, or changing the distance cache.
- Applicable-parent median cost remains available as an audit metric but is not
  presented as theta-conditional cost in the final table.

### Status
Accepted

---

## [2026-07-14] Keep strict-flip confusion summaries self-contained and backward compatible

### Background
The GlobalGCE Frequency-Top20 FGW audit computed the historical strict-flip
confusion matrix correctly, but downstream automation could report a null
mismatch count when a JSON reader expected one exact top-level field name.
Older audit files may also contain the complete four-cell matrix without the
redundant totals.

### Decision
Write the four confusion cells, `recorded_true_pairs`,
`expected_strict_pairs`, and `mismatch_rows` at the top level of
`strict_flip_confusion.json`. Preserve the previous field aliases, and allow
readers to infer missing redundant totals from an arithmetically consistent
four-cell matrix. Such legacy inputs are `PASS_WITH_WARNINGS`; contradictory
provided totals remain failures.

### Consequences
- Automated checks no longer interpret a missing redundant field as a failed
  core experiment.
- The mismatch count is always `TF + FT` and is guarded by explicit arithmetic
  assertions.
- Corrected pair details, parent cohorts, FGW distances, coverage, candidate
  ranking, and corrected Table 2 metrics are unchanged.

### Status
Accepted

---

## [2026-07-14] Audit GlobalGCE Frequency-Top20 from saved Node-FGW artifacts

### Background
GlobalGCE Frequency-Top20 shows plateaus and jumps in prefix-K coverage. Such a
shape can be caused by candidate marginal coverage, but it can also expose a
stale weak-flip artifact, rank drift, or inconsistent post-processing. The
saved pair details already contain all required Node-FGW distances, so an audit
must not recompute MolCLR embeddings or FGW transport.

### Decision
Add a read-only audit that verifies teacher-strict flip, candidate order,
frequency provenance, prefix/threshold monotonicity, exact-theta consistency,
fullgraph evaluation semantics, and per-rank marginal coverage. It reads the
external Frequency-Top20 order and never ranks candidates by FGW.

The report metric previously called `Conditional median cost` is the median
best strict-recourse distance over parents with any finite strict recourse; it
is not conditioned on `distance <= theta`. Keep its compatibility field, but
label it `Applicable-parent median cost`, add a distinct
`Covered-parent median cost`, and assert that the latter cannot exceed theta.
Likewise, label applicability as `Strict-recourse applicable rate` so it is not
confused with subgraph match applicability or teacher-target parent rate.

### Consequences
- Existing evaluator outputs, Node-FGW distances, caches, and candidate ranks
  remain unchanged.
- Historical tables using the ambiguous labels remain reproducible but are
  identified by the audit as reporting-label issues.
- Frequency-ranked candidates may legitimately produce coverage plateaus;
  they are accepted as data-driven only after strict-flip, order, and summary
  consistency checks pass.

### Status
Accepted

---

## [2026-07-14] Generate paper-style recourse reports without reevaluating candidates

### Background
The four final MolCLR-Node-FGW runs already contain teacher-strict pair details
for externally selected Top20 candidate sets. Reusing evaluator summaries alone
cannot produce prefix-K curves, and ranking candidates by their measured FGW
distance during reporting would leak the evaluation metric back into selection.

### Decision
Add a read-only GCFExplainer-style reporting entrypoint that restores each
candidate order from its recorded external selector file, validates exactly 20
unique ranked candidates and no evaluator-side selection, and aggregates Ours
match instances to the minimum finite strict-flip distance for each
parent-candidate pair. The report uses all 1283 parents as the denominator,
represents unavailable unconditional recourse cost as positive infinity, and
uses one shared absolute-threshold grid and paired parent-bootstrap indices for
all four methods.

### Consequences
- Existing MolCLR embeddings, FGW distances, caches, strict-flip semantics, and
  candidate sets remain untouched.
- Prefix-K and threshold curves are reproducible from final artifacts alone.
- Candidate selection remains external to evaluation and reporting; FGW values
  never determine candidate rank.
- The same reporting implementation can be reused for another distance line by
  supplying four run paths, a distance label, and a table prefix.

### Status
Accepted

---

## [2026-07-12] Filter legal GCF-HIVCSV molecules before greedy Top-K export

### Background
The original HIVCSV summary export ranked all generated graphs before the
graph-to-SMILES legality audit. Invalid graphs could therefore consume ranks
and update the covered-parent set. Its fallback expression also interpreted a
real `min_distance_seen=0.0` as missing.

### Decision
Keep the historical export unchanged as an experiment artifact, but add a
validity-first export path. Convert and sanitize the complete raw candidate
pool, discard illegal or empty molecules, and then apply the existing greedy
key `(marginal_coverage_gain, frequency, -min_distance_seen)`. Preserve one
shared order across metadata, graph tensors, and FGW-ready SMILES.

### Consequences
- Invalid candidates cannot influence native coverage selection.
- A real zero distance wins the expected tie-break instead of becoming 999.
- The valid Top-K export is deterministic and written beside, rather than over,
  the historical `summary_export` files.

### Status
Accepted

---

## [2026-07-12] Treat Ours Top20 as externally preselected in Node-FGW

### Background
One selected fragment can match a parent molecule at several atom mappings.
The Node-FGW evaluator expands those mappings into multiple detail rows, but
that expansion is evaluation work and is not a new candidate-selection step.

### Decision
When `PRESELECTED_TOPK` is enabled, validate Ours selector directories using
`selected_subgraphs.csv` or `selected_subgraphs.json`, preserve their rank
order, and record the external selector identity from `selector_summary.json`
or directory metadata. Ours evaluation rows use
`evaluation_row_unit=match_instance`, while candidate provenance remains
`candidate_set_preselected=true` and `selection_performed_in_eval=false`.

### Consequences
Run summaries separately report unique parent-candidate pairs, detail rows, and
valid match instances. Multiple `match_index` values no longer imply that the
evaluator optimized or reordered the selected Top20. Fullgraph preselection
validation remains unchanged.

### Status
Accepted

---

## [2026-07-12] Require a teacher transition for strict-flip CCRCOV

### Background
The MolCLR-Node-FGW evaluator treated every candidate whose post-intervention
prediction differed from the dataset target label as a flip. This counted
parents that the teacher already classified as non-target before intervention,
inflating CCRCOV without an actual prediction transition.

### Decision
Define the main strict-flip condition consistently across shared CCRCov pair
generation, MolCLR-Node-FGW aggregation, and baseline comparison as:

`pred_before == target_label and pred_after != target_label`.

The earlier condition, `pred_after != target_label`, is retained only as an
explicit `old_weak_flip` audit field. Main CCRCOV continues to use all evaluated
parents as its denominator, while `num_teacher_target_parents` is reported
separately.

### Consequences
- Parents already predicted as non-target cannot create strict-flip coverage.
- Pair details and summaries expose both definitions without mixing them.
- Existing FGW distances, caches, thresholds, and candidate selection remain
  unchanged; affected evaluations must be rerun to refresh their metrics.

### Status
Accepted

---

## [2026-07-12] Audit absolute Node-FGW radii across methods

### Background
Method-local `auto_quantile` sweeps can use the same quantile labels while
producing different absolute MolCLR-Node-FGW thresholds. Coverage at equal
quantile labels is therefore not necessarily coverage at an equal distance
radius.

### Decision
Add a read-only Node-FGW threshold consistency audit. Each `run_dir + method`
is treated as a distinct run. The audit compares FGW definition, teacher and
parent protocol, quantile grid, and absolute thresholds independently. A pair
is directly comparable only when FGW configuration, evaluation protocol,
parent count, and absolute thresholds all match.

### Consequences
Auto-quantile remains suitable for method-local diagnostics. Final fair tables
must use shared explicit absolute FGW thresholds. Ours is never selected as an
implicit reference; reference comparison requires an explicit run id.

### Status
Accepted

---

## [2026-07-12] Add CLEAR Parent-Frequency Top20 as a parallel selector

### Background
CLEAR full-molecule generation yields repeated canonical candidates across
source parents and experiment repetitions. Greedy-MMR is one valid global
selection protocol, but a direct frequency baseline is useful for separating
generation recurrence from coverage-proxy optimization.

### Decision
Add `selection_mode=parent_frequency` to the shared CLEAR selector. It reuses
RDKit validation, canonical deduplication, and the AIDS/HIV RF strict-flip
filter. For each canonical candidate it records raw row frequency, distinct
`source_instance_index` frequency, distinct experiment frequency, minimum
action cost, and mean action cost. Parent frequency intentionally excludes
`source_exp_id` from its key.

The exact ranking is parent frequency descending, raw frequency descending,
RF label-0 probability descending, minimum total action cost ascending, then
canonical SMILES ascending. Node-FGW, GED, MolCLR embeddings, and iterative
coverage gain do not participate in this ranking.

### Consequences
`CLEAR Parent-Frequency Top20` is reported separately from CLEAR Greedy-MMR.
Its Top20 CSV is a preselected candidate set: Node-FGW preserves row order,
requires exactly 20 unique RDKit-valid strict-flip molecules, and records
`selection_method=parent_frequency`, `selection_performed_in_eval=false`, and
`candidate_set_preselected=true`.

### Status
Accepted

---

## [2026-07-12] Preselect CLEAR Top20 before MolCLR Node-FGW evaluation

### Background
CLEAR produces 9,184 RDKit-valid full-molecule candidates after RF-unified
conversion. Evaluating all label-1 parents against the entire pool with
MolCLR-Node-FGW is computationally impractical and would also conflate global
candidate selection with final distance evaluation.

### Decision
Add a dedicated CLEAR fullgraph selector with the following pipeline:

```text
CLEAR candidate generation
-> RDKit validation and canonical deduplication
-> shared RF strict-flip filter
-> Morgan/Tanimoto greedy MMR Top20
-> MolCLR-Node-FGW final evaluation
```

The selector reuses the accepted coverage-heavy Ours weights
(`w_cf=0.8`, `w_cov=20.0`, `w_cost=0.3`, `w_red=0.7`) and the same weighted MMR
score helper. Full-molecule coverage is represented by packed Morgan/Tanimoto
parent bitsets; candidate redundancy is Morgan Tanimoto. Because the Ours
selector uses exact fragment support and defines no full-molecule similarity
threshold, `COVERAGE_THRESHOLD` must be supplied explicitly.

Node-FGW preselected mode requires exactly 20 unique RDKit-valid candidates,
preserves CSV order, performs no in-evaluator selection, and evaluates every
target parent against those 20 candidates. It records
`selection_performed_in_eval=false` and `candidate_set_preselected=true`.

### Consequences
The final CLEAR Node-FGW workload decreases from `parents x 9184` to
`parents x 20`. Node-FGW remains an evaluation-only distance and never enters
the CLEAR selector. CLEAR native total-action-cost ordering is diagnostic and
does not replace RF strict-flip greedy MMR selection in the fair table.

### Status
Accepted

---

## [2026-07-10] Select GlobalGCE fullgraphs by strict-flip MolCLR Node-FGW coverage

### Background
The MolCLR Node-FGW evaluator can write pair-level distance and teacher-flip
details for both `ours_selected_subgraphs` and `globalgce` in one CSV. A
GlobalGCE top2000 fullgraph pool needs a project-owned top-K selection step
that reflects its actual strict-flip explanatory coverage, rather than a
frequency-only or arbitrary first-K choice.

### Decision
Add `scripts/select_fullgraph_candidates_by_fgw_coverage.py`. The selector
filters the input detail table by exact `method=globalgce` before computing any
distance quantile or coverage. A candidate covers a parent only when:

```text
cf_flip == true and Node-FGW distance <= threshold
```

It greedily maximizes marginal parent coverage, breaking ties by lower mean
distance on newly covered parents, shorter SMILES, and earlier source-candidate
order. The resulting `selected_top20_for_eval.csv` uses:

```text
method = GlobalGCE
fullgraph_method = globalgce_selected20
```

and can be supplied as a fullgraph candidate input to the Node-FGW evaluator.

### Consequences
Ours rows are never used to select GlobalGCE fullgraph candidates. The
selection remains evaluation-only: it does not modify GlobalGCE, the teacher,
PPO, selector training, or the existing Node-FGW distance calculation.

### Status
Accepted

---

## [2026-07-07] Add MolCLR Node-FGW as evaluation-only CCRCOV distance line

### Background
Graph-level MolCLR cosine distance can be too coarse for some AIDS/HIV CCRCOV
threshold sweeps. A node-level distance can preserve more local molecular
structure without changing the generator, selector, or training objective.

### Decision
Add `molclr_node_fgw` as an auxiliary CCRCOV evaluation distance line. The line
uses the existing MolCLR pretrained GIN checkpoint but extracts node-level
embeddings before graph pooling. Molecules are compared with Fused
Gromov-Wasserstein distance using normalized unweighted shortest-path structure
matrices and cosine node feature cost:

```text
distance_line = MolCLR-Node-FGW
distance_type = node_fgw
FGW_LAMBDA = 0.5
```

The implementation caches both SMILES-level node artifacts and pairwise FGW
distances:

```text
outputs/hpc/cache/molclr_node_embeddings/
outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite
```

Thresholds default to auto quantiles instead of graph-level MolCLR cosine
thresholds because FGW has a different scale.

### Consequences
`molclr_node_fgw` is evaluation-only. It does not modify loss functions, PPO,
candidate generation, or selector logic. It also skips `StructRed`, `CovRed`,
and pairwise candidate redundancy. GREED-GED remains the main GED-style
distance line; MolCLR Node-FGW is an embedding-matrix auxiliary CCRCOV line.

### Status
Accepted

---

## [2026-07-05] Evaluate CLEAR full-graph pools with explicit graphPred teacher adapter

### Background
CLEAR `export_test` and candidate-pool conversion now produce AIDS full-graph
records with `original_adj`, `cf_adj`, `original_x`, and `cf_x`. The converted
pool also preserves CLEAR official prediction diagnostics, but those fields are
not the final unified metrics. The historical AIDS RF oracle is SMILES-based and
cannot directly score CLEAR's continuous graph tensors.

### Decision
Extend `scripts/baselines/clear/evaluate_clear_candidate_pool.py` with
`TEACHER_KIND=clear_graphpred`. This adapter loads the CLEAR graph prediction
checkpoint:

```text
baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt
```

and recomputes predictions for each original/counterfactual graph pair. The
evaluator records `strict_flip_eval`, `strict_flip_vs_original_label_eval`, and
`cf_drop_eval`, while keeping `official_flip` only as a diagnostic comparison.
`TEACHER_KIND=none` / `action_only` remain cost-only diagnostics and must not be
reported as final CLEAR FlipRate, CFDrop, or CCRCov.

### Consequences
Final CLEAR AIDS reporting must explicitly record `TEACHER_KIND=clear_graphpred`
or another documented unified teacher path. Official CLEAR flip/validity values
are never used as the final strict-flip condition.

### Status
Accepted

---

## [2026-07-05] Add CLEAR-RF-FullGraph path for final fair CCRCOV tables

### Background
The CLEAR AIDS pipeline now produces a full-graph candidate pool and can be
evaluated with `TEACHER_KIND=clear_graphpred`. That native diagnostic uses
CLEAR's own graph prediction checkpoint and action-distance costs. It is not
directly comparable to Ours and GT-FullGraph when those methods are evaluated
with the shared AIDS/HIV RF oracle and learned/embedding distances.

### Decision
Add a separate `CLEAR-RF-FullGraph` adaptation path. The path first audits
whether CLEAR's `original_adj`, `cf_adj`, `original_x`, and `cf_x` arrays can be
conservatively converted into valid RDKit SMILES. If conversion is feasible, it
writes RF-readable fullgraph candidates and evaluates them through the same
GREED-GED / MolCLR CCRCov pipeline used by Ours and GT-FullGraph:

```text
parent set = outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
teacher = outputs/hpc/oracle/aids_rf_model.pkl
CF_MODE = strict_flip
method = CLEAR-RF-FullGraph
```

If conversion is not reliable, the audit records `rf_oracle_usable=false` and
the native `clear_graphpred` result remains diagnostic only.

The converter treats CLEAR `cf_x` as a continuous decoder tensor rather than an
atom vocabulary. It must never use a raw float value as an atom-type key. For
the current AIDS pickles, atom identity is recovered from the AIDS descriptor
slot `original_x[:, 2] = atomic_num / 100`; true one-hot or soft categorical
rows, if introduced later, are decoded by row-wise argmax over the fixed AIDS
atom vocabulary (`C`, `N`, `O`, `F`, `S`, `Cl`). Counterfactual topology comes
from symmetrized/thresholded `cf_adj`, with single-bond valence checks used to
avoid obviously invalid RDKit molecules. The converter records feature-schema
statistics, argmax distributions, decode-mode counts, `cf_adj` statistics, and
CSV-level RDKit validation. It enforces a quality gate
(`MIN_VALID_CANDIDATES=20`, `MIN_VALID_RATE=0.001` by default); failing either
the conversion gate or CSV validation exits non-zero and blocks downstream fair
evaluation.

MolCLR-Node-FGW can run CLEAR fullgraph candidates without re-running ours by
setting:

```text
RUN_OURS=0
RUN_FULLGRAPH=1
RUN_GT_FULLGRAPH=0
CLEAR_FULLGRAPH_CANDIDATES_PATH=<clear_rf_fullgraph_candidates.csv>
FULLGRAPH_METHOD_NAME=CLEAR-RF-FullGraph
```

### Consequences
The final fair table must not substitute CLEAR native graphPred/action-distance
metrics for RF-oracle CCRCov. CLEAR can enter the final table only through
`CLEAR-RF-FullGraph` or another explicitly documented adapter that shares the
same parent set, teacher, distance system, thresholds, and strict-flip mode as
the other methods.

### Status
Accepted

---

## [2026-07-03] Train GCFExplainer-HIVCSV GNN with imbalance-aware metrics

### Background
The adapted GCFExplainer-HIVCSV path uses the canonical project source
`data/raw/AIDS/HIV.csv` with `LABEL_COLUMN=HIV_active`. The label distribution
is highly imbalanced (`0: 39684`, `1: 1443`), so overall accuracy can be
misleading and can hide majority-class collapse.

### Decision
Add the adapted HIVCSV scaffold:

- `scripts/gcf_hiv_csv_prepare_dataset.py` converts the canonical CSV into
  RDKit/PyG `graphs.pt` without external graph benchmark downloads;
- `scripts/gcf_hiv_csv_train_gnn.py` trains the adapted HIVCSV GNN teacher;
- `scripts/gcf_hiv_csv_run_vrrw.py` runs a project-owned lightweight
  GCF-style VRRW over the HIVCSV graphs;
- `scripts/gcf_hiv_csv_export_summary.py` and
  `scripts/evaluate_gcf_hiv_csv_native.py` produce top-K summaries and native
  close-CF coverage;
- `scripts/convert_gcf_hiv_csv_graphs_to_smiles.py` is a diagnostic conversion
  path only.

The training script uses deterministic stratified train/validation/test splits
and enables class-weighted `CrossEntropyLoss` by default:

```text
weight_c = total_train / (num_classes * count_c)
```

Checkpoint selection defaults to macro-F1, and `gnn_train_summary.json`
records overall accuracy, per-class precision/recall/F1, macro-F1, balanced
accuracy, ROC-AUC, prediction counts, class weights, and split label counts.
If test label-1 recall is below the configured threshold, the summary includes
a warning.

### Consequences
The adapted HIVCSV path is separate from the official AIDS graph-benchmark
reproduction and must be reported as `GCFExplainer-HIVCSV` or
`GCFExplainer-adapted-HIVCSV`. Accuracy alone is not accepted as evidence that
the HIVCSV GNN teacher is usable. The adapted path does not invoke external
graph benchmark downloads; it reads only the project CSV-derived `graphs.pt`.

### Status
Accepted

---

## [2026-07-03] Add official GCFExplainer native fullgraph baseline path

### Background
The project already contains a GT-FullGraph proxy baseline, but that proxy is
not the official GCFExplainer reproduction. Official GCFExplainer outputs a set
of complete counterfactual graphs and writes to fixed relative paths inside its
repository, which makes alpha sweeps unsafe unless each run is isolated.

### Decision
Add project-owned official GCFExplainer adapters and Slurm entrypoints without
modifying the official source. The new path resolves the official checkout from
`GCF_OFFICIAL_REPO`, `third_party/GCFExplainer`, or the legacy
`baselines/gcfexplainer_official` directory. VRRW runs execute inside an
isolated per-run workdir and write results under
`outputs/hpc/gcfexplainer_official`. Native evaluation uses official GNN
predictions, official NeuroSED distance, `GCF_MODE=official_native`,
`TEACHER_TYPE=official_gnn`, `DISTANCE_TYPE=official_native`, and
`CF_MODE=strict_flip`.

### Consequences
GT-FullGraph remains a project proxy and must not be named official
GCFExplainer. Graph-to-SMILES-to-RF evaluation is available only as a diagnostic
because official graph artifacts may not preserve safe atom/bond mapping.
NetworkX GED is not used for large fullgraph GCFExplainer evaluation; GREED-GED
and MolCLR diagnostics reuse the existing distance pipelines only when valid
SMILES candidates are available.

### Status
Accepted

---

## [2026-07-03] Audit GlobalGCE AIDS/HIV edge-label conversion modes

### Background
The first `native-cf-fullgraph` GlobalGCE AIDS/HIV evaluation ran successfully,
but graph-to-SMILES conversion produced very low validity and many sanitized
SMILES with implausible cumulene-like double-bond chains. This indicated that
the exported GlobalGCE edge labels may not always be raw zero-based bond labels;
for example, an internal label value of `1` can mean a single bond rather than
a double bond.

### Decision
Keep GlobalGCE official source unchanged and make the project adapter
edge-label interpretation explicit and auditable. The converter now supports
`raw_zero_based`, `internal_one_based`, `adjacency_only_single`, and default
`auto`. In `auto`, each graph tries internal one-based labels, raw zero-based
labels, and adjacency-only single bonds, then selects the first RDKit-sanitized
result by that priority. The evaluator records raw conversion ok/fail counts,
unique valid candidates before top-K, selected candidates after top-K, edge
label values seen, and conversion success/failure by edge-label mode.

### Consequences
GlobalGCE `native-cf-fullgraph` remains a diagnostic fullgraph candidate
evaluation. Strict CCRCOV now requires teacher-strict flipping:
`distance <= threshold`, `pred_before == target_label`, and
`pred_after != target_label`. The old weaker condition
`pred_after != target_label` is retained only as `old_weak_CCRCOV` /
`old_weak_flip` audit output. If `distance_mode=tanimoto`, reports continue to
label the distance as `tanimoto_fingerprint`; it must not be presented as GED.

### Status
Accepted

---

## [2026-07-03] Use weighted CLEAR graphPred training for AIDS/HIV imbalance

### Background
The canonical AIDS/HIV source is `data/raw/AIDS/HIV.csv` with
`SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`, and `TARGET_LABEL=1`.
The raw distribution is strongly imbalanced (`HIV_active=0: 39684`,
`HIV_active=1: 1443`). The CLEAR AIDS max100 x10 dataset preserves this
natural imbalance, and the initial CLEAR graph prediction run degenerated into
an almost majority-class predictor.

An audit found no existing balanced parent molecule classification dataset that
can be directly used for CLEAR `pred`. Existing balanced or label-conditioned
artifacts are SFT/PPO prompt files, label-specific candidate pools, selector
outputs, or `hiv_quick` evaluation outputs. They are not CLEAR graphPred
training data and must not be used as a substitute for the prepared AIDS
pickles.

### Decision
Keep the prepared CLEAR AIDS dataset:

```text
baselines/clear_official/dataset/aids_full.pickle
baselines/clear_official/dataset/aids_datasplit.pickle
```

Add `patches/clear_official/004_aids_weighted_graphpred.patch` so the official
CLEAR `train_pred.py` runtime copy uses class-weighted cross entropy only for
`dataset=aids`. The class weights are computed from the current training split:

```text
weight_c = total_train / (num_classes * count_c)
```

The patch also changes graphPred metrics to be computed over the full
validation/test split instead of averaging batch-level AUC/F1. AIDS pred logs
now report training label counts, class weights, `y_true_counts`,
`y_pred_counts`, `positive_pred_rate`, `balanced_accuracy`, F1, and ROC-AUC.
For AIDS pred, checkpoint selection prefers validation F1 rather than validation
loss alone.

### Consequences
The canonical AIDS/HIV raw dataset and CLEAR max100 x10 pickles remain the
source for CLEAR AIDS pred. No new balanced evaluation dataset is introduced.
SFT/PPO prompt files, label1 candidate pools, and `hiv_quick` evaluation
outputs remain forbidden as CLEAR graphPred training data. The fix is isolated
to the CLEAR patch workflow and does not change PPO, selector, candidate pool,
or unified evaluation logic.

### Status
Accepted

---

## [2026-07-03] Evaluate GlobalGCE on canonical AIDS/HIV labels

### Background
GlobalGCE official AIDS top30 reproduction and export produce
`globalgce_rules.jsonl` and `globalgce_cfs_graphs.jsonl`. The official AIDS
graph format has its own preprocessing and label alignment caveats, so its
internal graph labels must not be treated as final project labels.

### Decision
Add a project-facing GlobalGCE evaluator path for the canonical AIDS/HIV
dataset:

- final dataset display name is `AIDS/HIV`;
- raw source is `data/raw/AIDS/HIV.csv`;
- labels come from `HIV_active`;
- target label is `1`;
- GlobalGCE official graph outputs are treated as baseline-generated candidate
  artifacts;
- `native-cf-fullgraph` converts GlobalGCE CF graphs to RDKit molecules and
  canonical SMILES, then evaluates them with the project teacher;
- strict CCRCOV uses `distance <= threshold`,
  `pred_before == target_label`, and `pred_after != target_label`;
- smoke distance is explicitly named `distance_type=tanimoto_fingerprint` and
  must not be reported as GED.

### Consequences
SuppCov is skipped for `native-cf-fullgraph` because complete CF graph
candidates are not support rules. `native-cf-delta-action` and `rule-action`
remain safety/audit modes until reliable source-parent atom mapping and
attachment-aware LHS/RHS replacement are available.

### Status
Accepted

---

## Decision: AIDS/HIV is the canonical main dataset

### Background
Project scripts and baseline adapters historically use several names around the
same benchmark: `hiv`, `hiv_quick`, `aids`, and `ogbg_molhiv`. This creates a
risk that engineering validation runs are mixed with final AIDS/HIV baseline
results.

### Decision
AIDS and HIV are not two separate main datasets in this project. The canonical
main dataset is the AIDS/HIV dataset backed by the single raw CSV:

```text
data/raw/AIDS/HIV.csv
```

The canonical columns are `SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`,
and `TARGET_LABEL=1`. Different modules may keep different internal dataset
keys:

- `hiv` / `hiv_quick` are legacy internal names for the same raw CSV;
- `aids` is the official graph-baseline dataset key for CLEAR and GCF-style
  graph-format adapters;
- `ogbg_molhiv` is engineering validation only.

Final comparison must be unified to `data/raw/AIDS/HIV.csv` and must record the
metadata required by `docs/DATASET_CONTRACT.md`.

### Consequences
Do not report `ogbg_molhiv` CLEAR results as final AIDS/HIV baseline results.
All final CCRCOV, CFDrop, FlipRate, Cost, and Redundancy tables must be
traceable to the canonical CSV, label column, target label, baseline dataset
key, teacher/oracle path or teacher kind, and `CF_MODE=strict_flip`.

### Status
Accepted

---

## [2026-07-02] Add CLEAR AIDS dataset support

### Background
The AIDS/HIV main experiment uses `data/raw/AIDS/HIV.csv` with
`smiles` as the molecule column and `HIV_active` as the binary label. Previous
CLEAR engineering smoke runs used `ogbg_molhiv`, but that dataset is not the
AIDS/HIV main-result dataset. CLEAR official source only supports
`community`, `ogbg_molhiv`, and `imdb_m` out of the box.

### Decision
Add a project-owned AIDS preparation and patch workflow:

- `scripts/baselines/clear/prepare_clear_aids_dataset.py` converts
  `data/raw/AIDS/HIV.csv` into CLEAR-compatible `aids_full.pickle` and
  `aids_datasplit.pickle`, using `max_num_nodes=100` and `10` deterministic
  stratified split repetitions by default;
- `scripts/slurm/prepare_clear_aids_dataset.sh` provides the HPC sbatch
  entrypoint for deterministic stratified CLEAR-internal split preparation;
- `patches/clear_official/003_support_aids_dataset.patch` adds
  `dataset=aids` support to CLEAR official loaders, CLI choices, molecular
  evaluation branches, and graph prediction model behavior through the existing
  idempotent patch mechanism;
- CLEAR wrappers now recognize `aids` dataset files and keep all generated
  pickles/checkpoints/exports under ignored runtime paths.

### Consequences
`ogbg_molhiv` remains only a CLEAR engineering validation dataset. AIDS
baseline runs should use `dataset=aids`, `SMILES_COLUMN=smiles`, and
`LABEL_COLUMN=HIV_active`. CLEAR official flip/validity remain diagnostic only.
The historical AIDS RF oracle at `outputs/hpc/oracle/aids_rf_model.pkl` is a
SMILES/Morgan-fingerprint oracle and cannot directly consume CLEAR continuous
graph counterfactual tensors; final strict-flip CCRCov requires a full-graph
candidate pool and an explicitly documented unified teacher/adapter path.

### Status
Accepted

---

## [2026-07-02] Add CLEAR candidate/action pool unified evaluation entrypoint

### Background
The CLEAR reproduction pipeline now produces a converted candidate/action pool
under `outputs/hpc/baselines/clear/<dataset>/candidate_pool/`. The converted
pool preserves CLEAR official prediction diagnostics, but final baseline
comparison must use the project's unified teacher/oracle and native-action
CCRCov convention.

### Decision
Add `scripts/baselines/clear/evaluate_clear_candidate_pool.py` and
`scripts/slurm/evaluate_clear_candidate_pool.sh`. The evaluator reads CLEAR
action-pool JSONL files, computes action-distance costs, writes
`per_candidate_eval.jsonl`, `summary.json`, `summary.csv`, `threshold_summary.csv`,
and `report.md`, and exposes `strict_flip`, `drop_or_flip`, and `drop_only`
counterfactual modes. CLEAR official `official_flip` is diagnostic only and is
never used as final strict flip. If the candidate pool lacks SMILES,
precomputed unified-teacher fields, or full graph arrays needed by a graph
teacher adapter, the evaluator fails clearly unless `--allow-action-only` is
used for smoke diagnostics.

### Consequences
CLEAR official source and training remain unchanged. The current default CLEAR
candidate pool can be smoke-checked for cost/action summaries, but final
`FlipRate`, `CFDrop`, and `CCRCov` require a unified teacher prediction source
for CLEAR original/counterfactual graph pairs.

### Status
Accepted

---

## [2026-07-02] Convert CLEAR exports into unified candidate/action pools

### Background
CLEAR `export_test` now produces per-instance original/counterfactual graph
pairs under `outputs/hpc/baselines/clear/<dataset>/test_exports/`. These files
preserve full graph arrays and CLEAR official prediction diagnostics, but they
are not yet in the action-pool format consumed by the project's unified
CCRCov/action-rule evaluation.

### Decision
Add `scripts/baselines/clear/convert_clear_exports_to_candidate_pool.py` to
convert CLEAR export pickles into a project-owned JSONL candidate/action pool.
The conversion keeps official CLEAR flip and target-success diagnostics but
does not filter non-flips by default, because final `FlipRate`, `CFDrop`, and
`CCRCov` must be recomputed by the unified frozen teacher/oracle. Each
candidate records edge additions/deletions and continuous node-feature changes
from the original graph to the CLEAR counterfactual graph. A Slurm wrapper,
`scripts/slurm/convert_clear_exports_to_candidate_pool.sh`, provides the HPC
entrypoint.

### Consequences
CLEAR official source, model structure, training, and export logic remain
unchanged. Runtime candidate pools stay under
`outputs/hpc/baselines/clear/<dataset>/candidate_pool/` and must not be
committed. The resulting JSONL can feed the next CLEAR adapter/evaluator stage
for unified SuppCov, CCRCov, CFDrop, FlipRate, Cost, StructRed, and CovRed.

### Status
Accepted

---

## [2026-07-02] Add CLEAR per-instance counterfactual export

### Background
CLEAR official `test` loads trained CFE checkpoints and reports aggregate
metrics, but it does not persist per-instance original/counterfactual graph
outputs. The official entrypoint also maps `experiment_type == test` to
`test_small`, so the default printed metrics cover a small test subset. Unified
CCRCov/action-rule evaluation needs per-instance counterfactual graph records.

### Decision
Add a second project-owned CLEAR patch:

- `patches/clear_official/002_export_test_counterfactuals.patch` adds the
  marker `CLEAR_WRAPPER_EXPORT_TEST_COUNTERFACTUALS`;
- the patch adds opt-in CLI flags `--export_counterfactuals`,
  `--export_full_test`, `--export_max_items`, and `--export_dir`;
- official aggregate test behavior is preserved unless export flags are
  explicitly passed;
- `scripts/baselines/clear/run_clear.sh` adds `export_test` for full test
  split export and `export_test_small` for debugging;
- export files are written under
  `outputs/hpc/baselines/clear/<dataset>/test_exports/` as pickle arrays plus
  JSONL metadata.

### Consequences
CLEAR model structure, loss, optimizer, training logic, dataset loading, and
official aggregate metrics remain unchanged. The exported per-instance graph
records can be converted into a CLEAR candidate/action pool for unified
SuppCov, CCRCov, CFDrop, FlipRate, Cost, StructRed, and CovRed evaluation.

### Status
Accepted

---

## [2026-07-01] Patch CLEAR to save CFE checkpoints for test

### Background
CLEAR `pred` successfully saves the graph prediction model, but the official
`train` path in `baselines/clear_official/src/main.py` passes
`save_model=False`, while the official `test` path always loads
`../models_save/weights_graphCFE_CLEAR_<dataset>_exp<i>_epoch900.pt`. This can
make a completed CLEAR train run unusable for test because no CFE generator
checkpoint exists.

### Decision
Keep CLEAR algorithm code isolated and add a project-owned patch workflow:

- `patches/clear_official/001_save_cfe_checkpoints.patch` enables CFE
  checkpoint saving without changing model structure, losses, optimizer,
  dataset loading, or metrics;
- train now saves epoch 900 and final-epoch CFE `state_dict()` files with
  `[CLEAR_CKPT_SAVE]` logs;
- `scripts/baselines/clear/apply_clear_patches.sh` applies the patch
  idempotently by checking the `CLEAR_WRAPPER_SAVE_CFE_CHECKPOINT` marker;
- `scripts/hpc_pull_clear.sh`, `scripts/baselines/clear/slurm_clear.sbatch`,
  and `scripts/baselines/clear/run_clear.sh` apply the patch before CLEAR runs;
- wrappers check for exp0/exp1/exp2 epoch-900 CFE checkpoints and create an
  epoch-900 symlink from the highest available epoch when needed.

### Consequences
The submodule does not need to be committed dirty as the sole source of the
fix. Runtime artifacts remain ignored. CLEAR test fails early with a clear
checkpoint error if train has not produced any usable CFE checkpoint.

### Status
Accepted

---

## [2026-07-01] Make GREED/MolCLR CCRCov default to strict flip

### Background
GREED-GED and MolCLR-Embedding CCRCov smoke outputs could report
`close_cf_coverage > 0` while `flip_rate_among_covered = 0` because the
counterfactual condition allowed probability-drop coverage through
`cf_drop >= min_cf_drop`.

### Decision
For GREED/MolCLR distance-based CCRCov evaluation, add explicit `cf_mode`
support:

- `strict_flip`: `distance <= theta` and `pred_after != label`;
- `drop_or_flip`: `distance <= theta` and either strict flip or
  `cf_drop >= min_cf_drop`;
- `drop_only`: `distance <= theta` and `cf_drop >= min_cf_drop`.

The default is now `strict_flip`. `min_cf_drop` remains recorded and is used
only by drop-based modes. Slurm wrappers expose `CF_MODE` and `MIN_CF_DROP`,
and threshold summaries/reports record the selected mode.

### Consequences
The main GREED/MolCLR CCRCov result now matches the paper-facing
`phi(G^a) != y` strict flip definition by default. GREED training, MolCLR
encoding, PPO, selector, and candidate generation remain unchanged.

### Status
Accepted

---

## [2026-06-29] Add CLEAR official baseline HPC wrappers

### Background
CLEAR / GraphCFE is kept as an official baseline under
`baselines/clear_official`. Its official code relies on relative paths such as
`../dataset` and `../models_save`, so project-owned wrappers must run from the
official `src` directory while keeping datasets, checkpoints, logs, and outputs
out of ordinary Git.

### Decision
Add project-owned CLEAR workflow files:

- shared shell helpers for dataset checks, runtime directory creation, and
  environment diagnostics;
- a stage wrapper for `pred`, `train`, `test`, CLEAR baselines, and `all`;
- a Slurm wrapper that activates the HPC default `smiles_pip118` conda
  environment by default, allows `CLEAR_CONDA_ENV` overrides, requests the A800
  GPU queue with one `gpu:a800:1` allocation, and delegates to the stage
  wrapper;
- an HPC pull helper that syncs submodules and prepares runtime directories
  without downloading data;
- documentation and `.gitignore` rules for CLEAR datasets, checkpoints, and
  logs.

### Consequences
`baselines/clear_official/src/` remains untouched. CLEAR can be launched from
HPC via `sbatch` after `git pull`, while large runtime artifacts remain outside
normal Git history.

### Status
Accepted

---

## [2026-06-29] Add GREED-GED and MolCLR distance lines for CCRCov

### Background
Fullgraph CCRCov evaluation cannot scale if every parent-candidate pair is sent
to NetworkX GED. The current HIV comparison needs a distance protocol that can
evaluate Ours and the GT-FullGraph proxy baseline without blocking on exact GED.

### Decision
Add two evaluation-only distance lines:

- GREED-GED prepares HIV graph pairs, labels deletion pairs exactly, labels
  fullgraph/random pairs with a scalable bounded approximation unless an
  explicit debug mode is requested, trains a Siamese GIN-style distance model,
  and evaluates CCRCov with predicted normalized GED;
- MolCLR-Embedding precomputes parent, hard-deletion residual, and GT-FullGraph
  candidate embeddings with an explicit runtime MolCLR checkpoint and evaluates
  CCRCov using `1 - cosine_similarity`;
- NetworkX GED remains only a small debug option and is not the default
  fullgraph distance path;
- GT-FullGraph is treated as a project proxy baseline, not as official
  GCFExplainer.

### Consequences
Training PPO, selector logic, and candidate generation remain unchanged. The
new files provide sbatch-first workflows for smoke/full GREED, smoke/full
MolCLR, and final comparison plots under the native-action CCRCov convention.

### Status
Accepted

---

## [2026-06-26] Add GlobalGCE baseline reproduction and unified evaluation wrappers

### Background
GlobalGCE is a relevant global counterfactual explanation baseline, but its
official outputs and metrics are not directly comparable to the project's
native-action CCRCov protocol. The official code should remain isolated under
`baselines/globalgce_official`, while project-owned wrappers should control
HPC execution, artifact export, and unified re-evaluation.

### Decision
Add GlobalGCE support without modifying official source code:

- layout checking for `baselines/globalgce_official`;
- a wrapper that copies official `src` into `outputs/hpc/globalgce/...` and runs
  `main.py` from the copied tree;
- an exporter that records official metrics, introspects rules/CF pickles, and
  writes project-owned JSON/JSONL artifacts;
- a `src.baselines.globalgce_adapter` module for AIDS label maps, CF graph
  conversion, rule descriptors, structural redundancy, coverage redundancy, and
  label-alignment warnings;
- a unified evaluator that supports first-stage native-CF CCRCov and a
  rule-action audit mode that reports SuppCov/StructRed/CovRed while explicitly
  marking safe RHS replacement as unsupported;
- Slurm wrappers for smoke, official top30, export, and label-specific CCRCov
  evaluation;
- baseline documentation in `docs/BASELINE_GLOBALGCE.md`.

### Consequences
GlobalGCE official code remains untouched. All generated GlobalGCE run outputs
live under `outputs/hpc/globalgce/...`, and unified evaluation outputs live under
`outputs/hpc/eval/globalgce/...`. Official validity/proximity metrics remain
reproduction diagnostics, while final comparison metrics are recomputed by the
project's frozen teacher and CCRCov protocol.

### Status
Accepted

---

## [2026-06-26] Add Slurm experiment tracking entrypoint

### Background
The project has many Slurm jobs for PPO, candidate-pool generation, selectors,
baseline evaluation, CCRCov sweeps, and visualization. Direct `sbatch`
submission makes it easy to lose the job id, command, output root, git commit,
and notes needed to reconstruct a result later.

### Decision
Add a lightweight submission and tracking layer:

- `scripts/exp_sbatch.py` calls the real `sbatch` without `shell=True`, records
  successful and failed submissions, and supports dry-run inspection;
- `scripts/exp_sbatch.sh` provides a repository-root shell wrapper;
- `scripts/sync_experiment_status.py` appends Slurm status snapshots using
  `sacct` with `squeue` fallback;
- `docs/EXPERIMENT_LOG.md` stores append-only human-readable records;
- `outputs/hpc/experiment_registry/jobs.jsonl` is the runtime machine-readable
  registry path;
- `docs/EXPERIMENT_TRACKING.md` documents the standard workflow and optional
  shell alias.

### Consequences
Training, selector, and evaluation logic remain unchanged. Existing Slurm
scripts are not modified; future submissions should prefer the tracking wrapper
so that experiment provenance is preserved.

### Status
Accepted

---

## [2026-06-26] Add close counterfactual coverage evaluation workflow

### Background
The selected-subgraph method needs to be evaluated under a GCFExplainer-style
close counterfactual graph protocol while preserving the existing PPO,
candidate-pool, audit, overlap, and selector workflows. The comparison must
support both our selected fragments, which become counterfactual graphs only
after hard deletion from a parent, and GCF-style baselines that already provide
full counterfactual graph candidates.

### Decision
Add evaluation-only code:

- a close counterfactual coverage module that computes hard-deletion residuals,
  normalized deletion-GED, optional NetworkX GED, teacher prediction deltas, and
  teacher-embedding distance when the teacher exposes an embedding API;
- CLI entrypoints for single-mode evaluation, four-way ours/GCF by GED/embedding
  evaluation, and matplotlib threshold-sweep visualization;
- a label-1 Slurm wrapper for the VSCode -> git push -> HPC git pull -> sbatch
  workflow;
- focused tests for deletion, threshold de-duplication, embedding-distance
  semantics, and GCF no-deletion behavior.

### Consequences
- Training logic, reward logic, candidate-pool generation, selector scoring, and
  overlap analysis remain unchanged.
- Our selected fragments are evaluated by hard deletion with any-match
  semantics; GCF candidates are evaluated as full counterfactual graphs.
- GED defaults to the fast deletion-cost upper bound for our hard-deletion
  residuals, while GCF GED uses NetworkX graph edit distance because it has no
  deletion action.
- Embedding distance is available only when the supplied teacher/model exposes a
  graph-embedding method; otherwise rows record `embedding_ok=false` with an
  explicit error.

### Status
Accepted

---

## [2026-06-22] Add no-GNN GT-fullgraph Tanimoto baseline trajectory for Pareto plots

### Background
MolCLR-GNN selector sweeps provide one trajectory for the current method, but
the Pareto frontier plot also needs a comparable baseline trajectory that does
not use GNN embeddings. The available clean GT-fullgraph action motif pool is
`camc_gt_fullgraph_motif_pool.csv`; it can be re-selected with the same
top-k/gamma sweep idea using Morgan/Tanimoto redundancy and then evaluated by
the unchanged legacy HIV quick CAMC evaluator.

### Decision
Add evaluation-only scripts:

- a GT-fullgraph motif-pool selector that aggregates action motifs, scores them
  with CF/support/size proxies, and applies greedy MMR with RDKit Morgan
  Tanimoto redundancy;
- a Slurm wrapper for gamma/beta sweeps that writes legacy-evaluator-compatible
  `selected_subgraphs.csv` and `selected_subgraphs.json`;
- a manifest-driven plotting script that reads legacy evaluator
  `camc_comparison_table.csv` outputs for Ours-MolCLR-GNN and
  Baseline-noGNN-Tanimoto trajectories, marks three-objective Pareto points, and
  exports PNG/PDF figures.

### Consequences
- The legacy evaluator remains unchanged.
- The new baseline does not use GNN embeddings; redundancy is the original
  Morgan/Tanimoto structural similarity.
- Selected motif outputs are compatible with
  `scripts/eval/compare_hiv_recourse_baselines.py --ours-selected-dir`.

### Status
Accepted

---

## [2026-06-21] Make MolCLR-GNN skip policy produce selector-ready pools

### Background
The MolCLR-GNN embedding job can encode most candidate fragments while a small
number of invalid fragment SMILES fail RDKit/PyG graph construction. The
original `invalid_policy=skip` behavior omitted those SMILES from the embedding
map but still wrote their original rows to the output JSONL, leaving rows
without `final_fragment_gnn_embedding` and causing the selector's
`--embedding-missing-policy error` mode to fail.

### Decision
Keep selector behavior unchanged and fix only the embedding preparation layer:

- have the MolCLR encoder expose invalid-SMILES details alongside successful
  embeddings;
- make `scripts/add_candidate_pool_molclr_embeddings.py` skip failed rows from
  the output JSONL when `--invalid-policy skip` is used;
- retain zero-vector behavior only for `--invalid-policy zero`, with an explicit
  `molclr_embedding_status` marker;
- expand the summary with input/output row counts, skipped rows, zero rows, and
  missing-embedding checks.

### Consequences
- `INVALID_POLICY=skip` now creates selector-ready JSONL files whose retained
  rows all contain `final_fragment_gnn_embedding`.
- Failed rows remain auditable through `molclr_gnn_embedding_failed_rows.jsonl`.
- Existing ChemLLM text embeddings, Morgan/Tanimoto redundancy, and selector
  logic remain unchanged.

### Status
Accepted

---

## [2026-06-15] Add MolCLR-GNN embedding redundancy workflow for selector experiments

### Background
The selector already supports embedding-cosine redundancy through
`--sim-metric embedding`, but the existing learned embedding field is generated
from the ChemLLM text model. The current experiment needs a fragment-level graph
embedding from a pretrained MolCLR GIN/GCN encoder while keeping Morgan/Tanimoto
and ChemLLM embedding paths unchanged.

### Decision
Add an evaluation/selection workflow only:

- introduce `src.embeddings.molclr_gnn_embedding`, which converts fragment
  SMILES to MolCLR/PyG molecular graphs and loads MolCLR code/checkpoints from
  explicit runtime paths;
- add `scripts/add_candidate_pool_molclr_embeddings.py`, which writes
  `final_fragment_gnn_embedding` into a derived candidate-pool JSONL;
- add Ours and GT seed-13 Slurm wrappers for MolCLR-GNN embedding generation,
  embedding-redundancy selection, and legacy HIV quick CAMC re-evaluation;
- document that MolCLR code/checkpoints are external assets and must not be
  downloaded implicitly on HPC.

### Consequences
- Selector defaults remain unchanged; Morgan/Tanimoto remains available unless
  scripts explicitly pass `--sim-metric embedding`.
- The new `final_fragment_gnn_embedding` field affects only the redundancy
  similarity term, not coverage gain, counterfactual scoring, size penalties,
  training, rewards, or selected-subgraph generation.
- Ours and GT can be compared with the same MolCLR checkpoint and the same
  selector redundancy semantics.

### Status
Accepted

---

## [2026-06-11] Re-evaluate embedding selector sets with the legacy HIV quick CAMC evaluator

### Background
The candidate-pool sanity check can show protocol drift, but it still does not
answer whether the embedding selector top20 sets perform poorly under the exact
legacy CAMC evaluator that produced the old PPT table. The old reference table
was generated by `scripts/eval/compare_hiv_recourse_baselines.py` using the HIV
CSV, RF teacher, seed 13, and CAMC top-k/delta settings.

### Decision
Add an evaluation-only Slurm workflow that runs the unchanged legacy HIV quick
CAMC evaluator three times:

- old Morgan-MMR selector top20 as a reproduction control;
- embedding conservative wide-grid selector top20;
- embedding low-redundancy wide-grid selector top20.

Add a small summary script that reads each run's `camc_comparison_table.csv`,
extracts `method=ours_selected_subgraph, k=20`, compares embedding rows against
old Morgan, and checks whether old Morgan reproduces the old PPT values.

### Consequences
- The legacy evaluator metrics and implementation remain unchanged.
- The result directly distinguishes true embedding-selector coverage loss from
  the earlier candidate-pool-evidence protocol difference.
- All long-running work remains submitted through Slurm rather than executed on
  a login node.

### Status
Accepted

---

## [2026-06-11] Add selected-set sanity check for CAMC protocol drift diagnosis

### Background
The legacy HIV quick CAMC table reported much higher Ours coverage than the
new embedding-selector final table. The new table is computed from selected
top20 fragments and candidate-pool evidence, while the legacy CAMC evaluator
uses full target inputs plus RF-teacher deletion evaluation. A dedicated sanity
check is needed to distinguish true coverage loss from evaluator/protocol
differences.

### Decision
Add an evaluation-only selected-set sanity check that:

- reads multiple selector directories, including the old Morgan-MMR set and new
  embedding-MMR wide-grid sets;
- evaluates all selected sets under one shared candidate-pool evidence protocol;
- dumps each selected fragment with selector score, support evidence,
  cf-drop/flip evidence, atom-ratio evidence, and redundancy diagnostics;
- records the legacy CAMC generator location
  (`scripts/eval/compare_hiv_recourse_baselines.py`) and explains why this
  lightweight sanity command does not rerun teacher-based CAMC unless that full
  evaluator is launched separately.

### Consequences
- The check can reveal whether the embedding selector truly sacrificed coverage
  relative to the old Morgan selected set under the same evidence source.
- If the old Morgan set is also much lower under candidate-pool evidence than
  in the legacy CAMC table, the observed drop is mainly due to evaluator or
  evidence-source differences.
- Training code, selector defaults, selected-subgraph artifacts, and candidate
  pools remain unchanged.

### Status
Accepted

---

## [2026-06-10] Formalize embedding-cosine redundancy selector wide-grid CAMC workflow

### Background
The gamma-only embedding selector sweep showed that both ours and the relaxed
GT-fullgraph proxy baseline can run with `--sim-metric embedding`, but the
first pass did not find an ours gamma that simultaneously preserved
coverage/flip/cf-drop and reduced embedding redundancy below the GT mean. The
experiment therefore needs a wider coverage-vs-redundancy search and a final
CAMC-style table computed from the selected top20 fragments.

### Decision
Keep selector defaults and all training code unchanged, and add explicit
evaluation-only Slurm workflows:

- run Ours and GT-fullgraph relaxed selectors with embedding-cosine redundancy
  over a beta/gamma grid;
- summarize the grid without requiring identical beta/gamma alignment, while
  also reporting same-parameter deltas;
- identify Ours Pareto candidates by maximizing coverage, keeping flip high,
  preferring higher cf-drop, and minimizing embedding cosine redundancy;
- write conservative, balanced, and low-redundancy recommended Ours configs;
- compute the final selected-top20 CAMC-style table from selector outputs and
  candidate-pool evidence, explicitly flagging GT `cf_drop=0.0` rows as proxy
  filled rather than teacher re-evaluated strength.

### Consequences
- The official experiment scripts now make embedding cosine similarity the
  redundancy term by passing `--sim-metric embedding` explicitly.
- Morgan/Tanimoto remains available as the selector default and as a reporting
  diagnostic, but it is no longer the redundancy objective in this formal
  comparison workflow.
- Final CAMC table generation is reproducible from selected selector artifacts
  and candidate pools, with clear warnings about proxy GT cf-drop and
  theta-coverage fallback sources.

### Status
Accepted

---

## [2026-06-10] Relax GT-fullgraph embedding selector flow after candidate-pool diagnosis

### Background
The first GT-fullgraph embedding selector sweep produced zero coverage and null
selected metrics. The converted GT CAMC motif pools can lack teacher-recomputed
`cf_drop` / `cf_flip` fields, while the strict selector wrapper required
`--require-cf-flip` and the selector default `min_cf_drop=0.2`; this can filter
all GT proxy candidates before MMR selection.

### Decision
Keep selector defaults and training code unchanged, but add an explicit GT proxy
diagnosis and relaxed evaluation path:

- add a selector-pool diagnosis script that reports required-field coverage,
  embedding availability, strict-filter reasons, and relaxed-GT-filter reasons;
- make CAMC-to-candidate-pool conversion write `cf_drop=0.0` with
  `cf_drop_missing=true` when the CAMC motif pool lacks true `cf_drop`;
- make missing `cf_flip` default to `true` with `cf_flip_missing=true` for this
  GT proxy candidate-pool conversion;
- add a relaxed GT embedding sweep wrapper that removes `--require-cf-flip` and
  sets `--min-cf-drop -999` while keeping embedding redundancy,
  final-substructure filtering, deduplication, and the shared MMR weights;
- add a relaxed summary wrapper that compares ours against the relaxed GT sweep
  root.

### Consequences
- The original strict GT sweep script remains available for auditing.
- The relaxed path treats GT CAMC motifs as an already constructed baseline
  action pool rather than as teacher-rescored generated fragments.
- Ours and GT still use the same embedding redundancy selector once their
  candidate pools pass the minimum structural/embedding filters.

### Status
Accepted

---

## [2026-06-10] Add embedding-MMR comparison workflow for GT-fullgraph CAMC motifs

### Background
The embedding-based selector is available for our merged stable300 candidate
pool, but the GT-fullgraph CAMC baseline currently exists as action-motif pool
CSV files rather than selector-readable candidate pools with learned fragment
embeddings. A fair redundancy comparison requires both ours and GT-fullgraph to
run the same class-level MMR selector with `sim_metric=embedding`.

### Decision
Add evaluation/Slurm workflow code only:

- convert the three clean GT-fullgraph CAMC motif pools
  `label1_1594411`, `label1_1594412`, and `label1_1594413` into
  selector-readable JSONL pools, explicitly excluding the older `1593189` run;
- reuse `scripts/add_candidate_pool_embeddings.py` to add
  `final_fragment_embedding` to the converted GT pools;
- run embedding-redundancy gamma sweeps for both ours merged and GT-fullgraph
  proxy pools with identical MMR selector weights except for gamma;
- summarize ours-vs-GT sweep results by gamma, including GT mean/std over the
  three clean seeds and a simple pass/fail recommendation rule.

### Consequences
- Existing SFT, PPO, reward, selector defaults, selected-subgraph artifacts, and
  original candidate pools remain unchanged.
- The default selector redundancy metric remains Morgan/Tanimoto unless an
  experiment script explicitly passes `--sim-metric embedding`.
- GT-fullgraph CAMC motif pools can now participate in selector-level embedding
  redundancy comparisons after an explicit conversion and embedding-generation
  preparation step.

### Status
Accepted

---

## [2026-06-02] Add offline candidate-pool embedding generation for embedding-MMR selection

### Background
The class-level selector now supports `--sim-metric embedding`, but the current
stable300 merged candidate pool does not yet contain `final_fragment_embedding`.
The embedding selector should therefore consume a derived JSONL with learned
fragment embeddings instead of mutating or regenerating the original
`candidate_pool.jsonl`.

### Decision
Add an evaluation/inference utility layer only:

- introduce `scripts/add_candidate_pool_embeddings.py`, which reads an existing
  candidate pool, embeds each resolved fragment SMILES with the same ChemLLM
  base-model plus optional SFT/PPO PEFT adapter loading path used by
  `src.eval.full_candidate_pool`, and writes a new
  `candidate_pool_with_embeddings.jsonl`;
- keep `final_fragment` as the primary text source, with fallbacks through
  `core_fragment`, `final_fragment_smiles`, `candidate_smiles`, and
  `raw_fragment`;
- use attention-mask-aware mean pooling by default over the last hidden state,
  L2-normalize the vector, and record summary/failed-row sidecar files;
- add stable300 and generic Slurm wrappers for HPC embedding generation, then
  point the embedding-MMR selector wrapper at the derived embedded pool.

### Consequences
- Existing SFT, PPO, reward, selector training, selected-subgraph artifacts, and
  original candidate pools remain unchanged.
- The embedding selector now has a reproducible upstream preparation step before
  `--embedding-missing-policy error` is used.
- The default Morgan/Tanimoto selector path remains unaffected.

### Status
Accepted

---

## [2026-06-02] Add embedding-based redundancy mode for class-level selector

### Background
The class-level counterfactual subgraph selector currently uses greedy MMR with
Morgan/Tanimoto similarity as the redundancy penalty. For candidate pools that
also contain learned subgraph embeddings, we need an optional way to compute
redundancy from embedding cosine similarity while preserving the default Morgan
behavior and the deletion-based counterfactual objective.

### Decision
Extend only the selector/evaluation layer:

- keep `sim_metric=morgan` as the default and preserve the existing Morgan
  fingerprint Tanimoto redundancy path;
- add `sim_metric=embedding`, where redundancy is
  `max(0, cosine(candidate_embedding, selected_embedding))`;
- parse embeddings from `final_fragment_embedding` by default, with fallback
  fields for `embedding`, `fragment_embedding`, `subgraph_embedding`, and
  `graph_embedding`;
- add `embedding_missing_policy={error,skip}` so missing embeddings either fail
  clearly or are filtered explicitly;
- record MMR component diagnostics in selected outputs and add pairwise
  embedding cosine statistics to selector summaries/reports;
- add an HPC Slurm wrapper for the stable300 label-1 merged pool embedding
  selector and a lightweight embedding-field checker.

### Consequences
- Existing SFT, PPO, reward, selector training, and selected-subgraph artifacts
  remain unchanged.
- CF score, coverage gain, size penalty, and candidate filters remain the same;
  only the redundancy similarity source changes when explicitly requested.
- Existing Morgan/Tanimoto reports remain present for backward compatibility,
  with embedding statistics reported separately.

### Status
Accepted

---

## [2026-05-29] Replace eval Morgan fingerprint calls and add CAMC motif overlap diagnostics

### Background
The HIV quick comparison plus CAMC run completed, but the Slurm log contained a
large number of RDKit deprecation warnings:
`DEPRECATION WARNING: please use MorganGenerator`. These warnings came from
legacy Morgan fingerprint APIs used by evaluation and similarity helpers, not
from a training objective change.

### Decision
Keep the change evaluation/helper scoped:

- replace legacy Morgan bit-vector calls in HIV comparison, selector/audit
  similarity helpers, selected-subgraph overlap, and chemistry projection/repair
  Tanimoto helpers with cached `rdFingerprintGenerator.GetMorganGenerator`
  instances;
- keep `src/rewards/reward_calculator.py` unchanged because reward code is out
  of scope and already prefers the newer generator when available;
- add `--suppress-rdkit-warnings` / `--no-suppress-rdkit-warnings` to the HIV
  comparison script as a fallback log-control option;
- add CAMC `motif_overlap_diagnostics` comparing ours and GT selected motifs by
  exact overlap, max Tanimoto, atom counts, aromatic motifs, and hetero-atom
  motifs;
- make the HIV quick Slurm wrapper count MorganGenerator deprecation warnings in
  `progress.log` at the end of the job.

### Consequences
- Existing SFT, PPO, reward, selector training, and selected-subgraph artifacts
  remain unchanged.
- CAMC metrics are unchanged; the new motif-overlap block is diagnostic only.
- Future HIV quick comparison logs should show zero MorganGenerator deprecation
  warnings unless a separate code path still uses RDKit's legacy API.

### Status
Accepted

---

## [2026-05-28] Add theta-aware recourse coverage diagnostics and CAMC action-motif comparison

### Background
The HIV quick comparison ran end to end, but long runs lacked enough progress
logging for Slurm monitoring. The ours recourse coverage could also show
`k=20 < k=10`, which indicated that the evaluator was choosing one action before
applying theta instead of computing theta-aware existential coverage over the
top-k action set. The current analysis also needs a second, method-aligned view
that compares shared counterfactual action motifs while remaining applicable to
fullgraph baselines.

### Decision
Update only the evaluation and Slurm layers:

- rebuild `scripts/eval/compare_hiv_recourse_baselines.py` around explicit
  ours action candidates and theta-aware existential aggregation;
- keep the original recourse-level outputs while adding
  `ours_action_candidates.csv`, `diagnostic_counts.json`, and `progress.log`;
- add CAMC output files that evaluate action motifs from our selected fragments
  and MCS-deleted motifs extracted from GT or extra selected fullgraph SMILES;
- add flushed logging, tqdm progress, MCS timing diagnostics, and recourse/CAMC
  monotonicity warning lists;
- update `scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh`
  to tee logs to the run directory and pass CAMC/progress controls.

### Consequences
- Existing SFT, PPO, reward, selector training, and selected-subgraph artifacts
  remain unchanged.
- Recourse coverage is now computed as `exists feasible action` under each
  theta, so it should be monotone in both `k` and theta.
- CAMC is more aligned with class-level counterfactual subgraph selection, but
  fullgraph methods can still participate when they provide selected fullgraph
  SMILES. Official graph benchmark outputs still require a graph-level CAMC
  evaluator or a SMILES adapter.

### Status
Accepted

---

## [2026-05-28] Add HIV quick recourse-level comparison evaluator

### Background
The official GCFExplainer AIDS reproduction is now available as a sanity check,
but it is not directly comparable to the current HIV/SMILES counterfactual
fragment system. The next practical need is a fast recourse-level comparison in
the current RF-teacher setting that can compare our selected subgraphs with a
simple opposite-label full-molecule baseline without changing training code.

### Decision
Add an evaluation-only quick comparison path:

- `scripts/eval/compare_hiv_recourse_baselines.py` evaluates
  `ours_selected_subgraph` by deleting selected fragments from each target
  molecule and evaluates `gt_fullgraph_greedy` by greedily choosing
  opposite-label full molecules;
- both methods are normalized to per-input recourse candidates `G_i'` and scored
  with the same RF teacher for `p_before`, `p_after`, `cf_drop`, and `cf_flip`;
- both methods use the same RDKit MCS proxy distance for approximate recourse
  cost;
- `scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh` provides
  an HPC wrapper with `smiles_pip118`, environment diagnostics, HIV CSV
  auto-discovery, and explicit failure when the CSV is ambiguous;
- `docs/baselines/hiv_quick_recourse_comparison.md` documents the scope,
  limitations, metrics, and commands.

### Consequences
- Existing SFT, PPO, reward, selector, and candidate-pool training code remains
  unchanged.
- The comparison is explicitly a quick HIV/SMILES RF-teacher analysis, not an
  official GCFExplainer reproduction and not a learned/exact GED evaluation.
- Ours-style subgraph match coverage is retained as an internal diagnostic, but
  headline comparison uses recourse-level coverage, cost, drop, and flip metrics.

### Status
Accepted

---

## [2026-05-28] Harden official GCFExplainer conda activation and add AIDS GNN training wrapper

### Background
The official GCFExplainer AIDS reproduction needs a temporary upstream GNN
checkpoint before `vrrw.py` can run. The first Slurm attempt failed before
training because `source ~/.bashrc` transitively loaded `/etc/bashrc`, where the
HPC shell referenced `BASHRCSOURCED` while `set -u` behavior could treat it as
an unbound-variable error.

### Decision
Keep the fix isolated to the official GCFExplainer scaffold:

- add `scripts/slurm/gcfexplainer/train_aids_gnn.sh` to train upstream
  `gnn.py --dataset aids` in the separate `gcfexplainer_py38` environment;
- source `/share/home/u20526/anaconda3/etc/profile.d/conda.sh` directly in all
  GCFExplainer Slurm wrappers instead of `~/.bashrc`;
- enable `set -u` only after conda activation in these wrappers;
- fail fast in the `vrrw` and all-in-one wrappers when
  `baselines/gcfexplainer_official/data/aids/gnn/model_best.pth` is missing,
  with a direct pointer to the GNN training sbatch command.

### Consequences
- Existing SFT, PPO, reward, selector, and candidate-pool code paths remain
  unchanged.
- The GCFExplainer baseline no longer depends on user shell startup files for
  conda activation.
- Missing official AIDS GNN checkpoints now produce a clear remediation command
  instead of a later opaque upstream failure.

### Status
Accepted

---

## [2026-05-26] Add isolated official GCFExplainer AIDS reproduction scaffold

### Background
The project needs a reproducible way to run the official GCFExplainer AIDS
baseline from the frozen third-party source under
`baselines/gcfexplainer_official/` while preserving the current v3 SMILES
counterfactual objective and avoiding dependency contamination in the main
`smiles_pip118` environment.

### Decision
Add an isolated HPC reproduction scaffold:

- `scripts/slurm/gcfexplainer/reproduce_aids_vrrw.sh` runs upstream
  `vrrw.py --dataset aids` and syncs official `results/aids/` artifacts into a
  per-job output directory.
- `scripts/slurm/gcfexplainer/reproduce_aids_summary.sh` runs upstream
  `summary.py --dataset aids`, then parses coverage/cost text into JSON and CSV.
- `scripts/slurm/gcfexplainer/reproduce_aids_all.sh` runs both stages in one
  Slurm job.
- `scripts/eval/collect_gcf_official_results.py` parses known and partially
  unknown summary formats without failing silently.
- `docs/baselines/gcfexplainer_reproduction.md` documents local checks, separate
  conda environment setup, Slurm commands, and result inspection.

### Consequences
- Existing SFT, PPO, reward, selector, and candidate-pool code paths remain
  unchanged.
- Official GCFExplainer dependencies stay in a separate `gcfexplainer_py38`
  environment by default, with `CONDA_ENV` override support.
- The official AIDS run is recorded as a sanity check, not as a fair comparison
  with the current HIV/SMILES method.
- Missing upstream `vrrw.py` or `summary.py` fails fast with explicit errors, and
  unparseable summary logs still produce machine-readable failure artifacts.

### Status
Accepted

---

## [2026-05-22] Add label=1 Base/SFT/PPO ablation wrappers

### Background
The current stable300 label=1 model is selector-ready, but the project still
needs a controlled ablation to separate the contribution of SFT from the
additional contribution of PPO. The comparison must keep the label=1 parent set,
teacher/oracle, generation count, projection settings, audit tooling, and
selector settings fixed so the measured differences are attributable to model
stage rather than sampling or downstream configuration changes.

### Decision
Add a label=1 ablation layer that reuses the existing full-pool generator,
candidate-pool audit, and class-level selector:

- `scripts/generate_full_candidate_pool.py` can now treat `--sft-lora-path NONE`
  as a base-model-only inference run, skipping PEFT adapter loading while
  preserving existing SFT-only and PPO adapter paths.
- Three Slurm wrappers generate and audit comparable `n=4` candidate pools for
  Base ChemLLM, SFT-only, and SFT+PPO stable300.
- Three selector Slurm wrappers apply the same coverage-heavy MMR selector
  settings to each pool.
- `scripts/export_candidate_pool_audit_artifacts.py` materializes audit sidecar
  artifacts expected by ablation bookkeeping.
- `scripts/summarize_label1_sft_ppo_ablation.py` combines audit and selector
  summaries into CSV and Markdown reports.

### Consequences
- The ablation does not modify PPO training code, stable PPO training code, or
  label=0 logic.
- Base ChemLLM can now be evaluated through the same generation/audit path as
  SFT and PPO checkpoints.
- The high-temperature merged pool remains reserved for the final complete
  system and is excluded from this main ablation to avoid confounding sampling
  diversity with training-stage effects.

### Status
Accepted

---

## [2026-05-21] Add a same-source label0 PPO prompt builder for unified label01 runs

### Background
The existing label=1 PPO prompt CSV in the SFT v3 HIV dataset directory is a
minimal `smiles,label` file and is consumed by downstream PPO, pool generation,
and selector jobs. The corresponding label=0 CSV was missing, which prevented
the unified label01 prompt build from proceeding. Copying label=1 rows and
changing the label would violate the counterfactual data contract because
label=0 prompts must come from genuine label=0 parent molecules.

### Decision
Add a small reusable prompt-CSV builder around the existing PPO prompt dataset
loader:

- `scripts/build_label_ppo_prompt_csv.py` reads a shared source CSV/JSONL,
  resolves SMILES and label columns with existing fallbacks, filters by
  `--target-label`, and writes only `smiles,label`;
- `scripts/slurm/build_sft_v3_hiv_ppo_prompts_label0_same_as_label1.sh`
  builds the missing label=0 file from the same SFT v3 train split source and
  also emits the stratified `shuffle_seed13` variant;
- `scripts/slurm/build_unified_ppo_prompts_label01.sh` now uses the same
  minimal builder when label0 is missing, and merges label0/label1 into a
  shuffled two-column unified CSV without requiring a separate source CSV when
  both label-specific prompt files already exist.

### Consequences
- The existing label=1 prompt file and downstream label=1 pipeline remain
  untouched.
- Unified label-conditioned PPO can build its input from genuine label=0 and
  label=1 parents while preserving the minimal prompt CSV schema expected by
  current generation/training loaders.
- The label0 build is reproducible from Slurm and produces the same style of
  stratified shuffle companion file used by stable label=1 PPO runs.

### Status
Accepted

---

## [2026-05-21] Harden unified label01 prompt Slurm input resolution

### Background
The unified label-conditioned PPO prompt build wrapper previously required an
original source training CSV even when label0 and label1 PPO prompt CSVs already
existed. On HPC this caused `build_unified_ppo_prompts_label01.sh` to fail with
an unresolved `SOURCE_INPUT_CSV`, even though the intended next step could have
been completed by merging the existing label-specific prompt files.

### Decision
Keep the Python prompt builders unchanged and harden only the Slurm wrapper:

- preserve explicit `SOURCE_INPUT_CSV` support when the user supplies a valid
  source CSV;
- if `SOURCE_INPUT_CSV` is empty and both label-specific prompt CSVs already
  exist, merge them directly into the unified label01 prompt CSV;
- otherwise, auto-discover an original training CSV under the dataset directory
  while excluding derived prompt, audit, candidate, balance, and summary files;
- emit structured `[UNIFIED_PROMPT_*]` diagnostics and a richer unified summary
  JSON with source mode and label counts.

### Consequences
- Existing label=1 prompt artifacts are not rebuilt unless explicitly requested
  with `FORCE_REBUILD_PROMPTS=true`.
- HPC users can recover from missing source-path configuration without manual
  intervention when label0 and label1 prompt CSVs are already available.
- Failure logs now include dataset CSV listings and column previews, making
  source CSV selection issues much easier to diagnose.

### Status
Accepted

---

## [2026-05-17] Add a dedicated Slurm wrapper for auditing the stable300 candidate pool with the existing selector-facing audit script

### Background
The repository already has a selector-facing candidate-pool audit entrypoint in
`scripts/audit_candidate_pool.py`, and the stable300 run
`decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500`
has completed. The immediate need is to audit its saved
`candidate_pool.jsonl` on HPC without introducing a new audit implementation or
changing any PPO or stable-PPO training code.

The existing audit CLI was confirmed to support the required arguments:

- `--pool_jsonl`
- `--out_json`
- `--out_txt`
- `--group_by_label`
- `--sim_sample_size`
- `--topk_show`

### Decision
Add only a thin Slurm wrapper:

- `scripts/slurm/audit_candidate_pool_stable300.sh`

This wrapper:

- targets the stable300 candidate pool path directly;
- prints environment and path diagnostics;
- checks that both the candidate pool and the audit script exist;
- writes outputs into
  `outputs/hpc/audits/<RUN_NAME>_candidate_pool_audit/`;
- reuses the existing `scripts/audit_candidate_pool.py` unchanged.

### Alternatives considered
1. Write a new stable300-specific audit script.
2. Hardcode the stable300 path inside the existing audit Python entrypoint.
3. Run the audit manually from an interactive shell without a Slurm wrapper.

### Consequences
- Stable300 can now be audited with a single `sbatch` command.
- The result remains directly comparable with earlier candidate-pool audits
  because the same Python audit implementation is reused.

### Status
Accepted

---

## [2026-05-20] Add unified label-conditioned PPO prompt, training-wrapper, and overlap-analysis pipeline

### Background
The stable300 label=1 pool already became selector-ready, so the next question
is no longer whether more PPO steps help. Instead, we now need to test whether
one shared policy can condition on the parent label and produce useful
counterfactual fragments for both label=0 and label=1 parents, while keeping
the existing selector and pool-audit tooling intact.

That requires:

- explicit label-conditioned prompt construction for label0, label1, and mixed
  label01 training sets;
- a unified stable PPO submission path that keeps per-sample labels visible in
  logs;
- separate full-pool generation and selection for unified label0 and label1
  outputs;
- a final overlap analysis over the selected category-level fragment sets.

### Decision
Extend the repository with a unified label-conditioned PPO experiment layer
without rewriting selector/merge tooling or the legacy PPO entrypoint:

- add `src/data/unified_ppo_prompts.py` plus thin CLIs for:
  - building label-specific PPO prompts,
  - building balanced unified label01 prompts,
  - checking unified prompt balance by 50-row blocks;
- keep `scripts/train_ppo_stable.py` as the stable PPO entrypoint, but add an
  opt-in `[UNIFIED_PPO_SAMPLE]` per-sample logging path so unified runs can be
  audited label-wise without creating a parallel trainer;
- extend `scripts/analyze_stable_ppo_log.py` to parse the new sample logs and
  report label0/label1 metrics by training block while preserving old log
  behavior when the new tag is absent;
- reuse the existing full-pool generator and selector for unified label0 and
  label1 pools through new Slurm wrappers only;
- add `src/eval/selected_subgraph_overlap.py` and a thin CLI that compares
  selected fragment sets by exact canonical overlap, soft Morgan overlap, and
  Murcko scaffold overlap.

### Alternatives considered
1. Fork `train_ppo_stable.py` into a second unified trainer.
2. Compare full-pool overlap only and skip selector-level overlap.
3. Build unified prompts with a new schema that drops compatibility anchors like
   `ORIGINAL_LABEL`, `MOLECULE_SMILES`, and `FRAGMENT_SMILES`.

### Consequences
- Unified PPO can now be tested with minimal risk to the existing label=1
  stable300 path because the main trainer is reused rather than duplicated.
- Selector-level overlap becomes a first-class analysis artifact instead of an
  ad hoc notebook step.
- The unified prompt format explicitly conditions on the original label while
  still preserving compatibility anchors expected by current PPO data loaders
  and generation helpers.

### Status
Accepted

---

## [2026-05-20] Add class-level MMR selector and diversity-side pool merge tooling

### Background
Stable PPO training has already converged far enough for offline pool building,
and the stable300 full candidate pool audit now says the pool is suitable for a
selector. At the same time, the audit still flags high mode-collapse risk and
recommends sampling tuning rather than more PPO steps. That means the next phase
needs two things:

- a class-level selector that can turn a large candidate pool into a shared,
  low-redundancy fragment set;
- a safe way to compare the current base pool against a higher-temperature pool
  and optionally merge them without touching any PPO training code.

### Decision
Add a selector-and-merge layer around the existing full-pool generation and
audit pipeline:

- `src/eval/class_counterfactual_selector.py` implements a greedy MMR selector
  over filtered counterfactual candidates, scoring fragments by shared
  counterfactual strength, marginal parent coverage, redundancy penalty, and
  atom-ratio size regularization.
- `scripts/select_class_counterfactual_subgraphs.py` exposes that selector as a
  thin CLI and writes JSON, CSV, summary JSON, and TXT report artifacts.
- `src/eval/candidate_pool_merge.py` and
  `scripts/merge_candidate_pools.py` merge multiple pool JSONLs while
  deduplicating by `(final_fragment, parent_smiles)` and keeping the
  higher-scoring candidate.
- New Slurm wrappers cover:
  - base-pool selector runs,
  - high-temperature stable300 full-pool generation plus audit,
  - merged-pool creation plus audit,
  - merged-pool selector runs.

The implementation explicitly reuses the existing candidate-pool normalization
contract from `src/eval/candidate_pool_audit.py` so selector filtering stays
compatible with current and future pool field aliases.

### Alternatives considered
1. Continue PPO training to chase diversity improvements.
2. Pick fragments only by raw `cf_drop` without redundancy control.
3. Rebuild the pool schema specifically for selector consumption.

### Consequences
- Selector development is now decoupled from PPO training and can iterate on
  existing stable300 pools.
- Diversity recovery can be tested through higher-temperature sampling and pool
  merging without changing stable300 checkpoints.
- Shared field normalization between audit and selector reduces schema drift
  risk when pools come from slightly different generation paths.

### Status
Accepted

## 2026-05-17 Stable300 Full Candidate Pool Wrapper

### Background
The stable PPO diagnostic path has already converged on
`decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500`
as the current selector-ready checkpoint. We do not want to continue PPO
training, rewrite the existing pool audit, or fork new reward logic just to
prepare the full label=1 candidate pool.

### Decision
Reuse the existing full-pool generation and candidate-pool audit entrypoints:

- `scripts/generate_full_candidate_pool.py`
- `scripts/audit_candidate_pool.py`

and add a single Slurm wrapper:

- `scripts/slurm/generate_and_audit_full_pool_stable300_label1_n4.sh`

The wrapper generates a 4-candidate-per-parent full pool for the complete
label=1 PPO prompt CSV, skips regeneration when a non-empty pool already
exists, and then audits the resulting JSONL with the existing
selector-facing audit script.

### Alternatives considered
1. Add a new stable-specific Python generation pipeline.
2. Re-run PPO or tune stable300 before building the full pool.
3. Write a second full-pool audit implementation tailored to selector prep.

### Consequences
- Stable300 full-pool generation stays aligned with the same chemistry,
  projection, oracle, and candidate-pool schema already used elsewhere.
- The workflow remains `git pull` + `sbatch`, with no checkpoint mutation and
  no PPO code changes.
- Existing `candidate_pool.jsonl` runs can be resumed safely because the Slurm
  wrapper only regenerates when `FORCE_REGEN=true` or the pool is missing /
  empty.

### Status
Accepted

---

## [2026-05-17] Add a parallel conservative stable-PPO path without modifying the existing PPO entrypoint

### Background
The repository already had a working decoded-chem PPO path and a best short-run
checkpoint from the original shuffled prompt order, but longer 150/200/300-step
runs showed drift symptoms:

- later-step reward and `cf_flip` dropped;
- `approx_kl` rose in the back half of longer runs;
- short shuffled PPO remained more reliable than simply extending ordinary PPO.

The user explicitly required that the original PPO code remain untouched so the
team could run apples-to-apples control experiments between:

1. the existing PPO path, and
2. a new conservative / stable PPO path with stronger KL control and lower
   update aggressiveness.

### Decision
Add a new stable-only training and analysis path that stays fully parallel to
the original PPO entrypoint:

- `scripts/train_ppo_stable.py` is a new decoded-chem PPO entrypoint that
  reuses existing loaders / model builders / reward helpers but implements its
  own conservative PPO update behavior;
- `src/rewards/reward_wrapper_stable.py` adds a stable-only post-processing
  layer around the existing reward wrapper so teacher-confidence gating can be
  applied without changing default reward behavior in ordinary PPO;
- the stable PPO path now supports optional environment / CLI overrides for:
  lower learning rate, smaller clip range, fewer PPO epochs, explicit gradient
  clipping, reward clipping, optional reward / advantage normalization, target
  and hard KL monitoring, adaptive KL penalty, teacher-confidence gating, and
  validation-based best-checkpoint / early-stop logic;
- new Slurm wrappers were added for:
  - a 5-step smoke run, and
  - a 200-step conservative shuffled-label1 run;
- `scripts/make_stratified_ppo_prompts.py` adds a new stratified shuffle tool
  for PPO prompt CSVs;
- `scripts/analyze_stable_ppo_log.py` adds a stable-PPO segment analyzer for
  1-50 / 51-100 / 101-150 / 151-200 blocks.

### Alternatives considered
1. Modify `scripts/train_ppo.py` in place to add stable flags.
2. Change the default reward wrapper behavior globally.
3. Keep using only the original shuffled short-run PPO path and skip a
   conservative long-run branch entirely.

### Consequences
- The old PPO entrypoint and its paired Slurm script remain unchanged, which
  preserves backward compatibility and protects current baselines.
- Conservative long-run PPO can now be tested as a parallel branch rather than
  as a behavior change hidden behind new flags in the original script.
- Teacher-confidence gating and stable-KL logic are isolated to the new stable
  path, which keeps the default reward semantics unchanged for existing runs.

### Status
Accepted

---

## [2026-05-17] Add full-dataset candidate-pool generation and selector-facing audit for original shuffle100 before any further PPO training

### Background
By this point the project had already completed:

- SFT v3;
- decoded PPO diagnostics;
- teacher-confidence filtering;
- ordered-vs-shuffle control experiments.

The working conclusion shifted from "keep extending PPO" to "treat the current
best shuffled short-run checkpoint as a candidate generator and measure whether
its full label=1 pool is good enough for the downstream class-level selector."

The current best checkpoint is the original shuffled 100-step run
`decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500`.
The next priority is therefore not more PPO optimization, but:

1. generate a full label=1 candidate pool with multiple candidates per parent,
   without any PPO updates;
2. compare that pool against an SFT-only baseline;
3. audit legality, counterfactual utility, diversity, redundancy, and parent
   coverage in a selector-facing format.

### Decision
Add a separate full-pool inference and audit path that stays outside SFT and
PPO training logic:

- `src/data/ppo_prompt_dataset.py` now provides one normalized loader for PPO
  prompt CSV / JSONL files, including fallbacks for `parent_smiles`, `smiles`,
  prompt-text SMILES extraction, and prompt reconstruction when needed;
- `src/eval/full_candidate_pool.py` now provides:
  - checkpoint inspection helpers for adapter-root vs `checkpoint-*` layouts,
  - adapter load-path resolution,
  - offline multi-candidate generation over the full prompt pool,
  - reward/evaluator reuse through `ChemRLRewarder.compute_rewards_from_decoded(...)`,
  - JSONL row enrichment so the output is compatible with existing
    `candidate_pool.jsonl` consumers while also exposing new selector-facing
    aliases such as `final_fragment`, `projection_used`, `final_substructure`,
    and `cf_oracle_called`;
- `scripts/generate_full_candidate_pool.py` is the thin CLI entrypoint for
  full-dataset inference with either:
  - SFT-only mode: base model + SFT LoRA, or
  - PPO mode: base model + resolved PPO adapter;
- `src/eval/full_candidate_pool_audit.py` now computes a richer audit than the
  earlier checkpoint-level candidate-pool audit, including:
  - pool scale,
  - legality,
  - counterfactual-oracle usage,
  - size statistics,
  - diversity / redundancy,
  - selector-facing fragment coverage over the full label=1 parent set,
  - explicit failure-case export;
- `scripts/audit_full_candidate_pool.py` is the thin CLI wrapper for that
  selector-facing full-pool audit;
- new Slurm wrappers now support one-command HPC generation / audit for:
  - PPO original shuffle100,
  - optional chained generate+audit,
  - SFT-only baseline;
- the shuffle200 path is kept as a future template only; it is not executed by
  default.

### Alternatives considered
1. Wait for shuffle200 to finish before building any full-pool generation path.
2. Reuse only partial `candidate_pool.jsonl` artifacts from training steps
   instead of running dedicated full-dataset inference.
3. Build a separate reward implementation for inference-time pool scoring.

### Consequences
- The repository now has a dedicated offline candidate-generation path that
  reuses the existing decoded-chem reward/evaluator instead of duplicating the
  reward logic.
- Selector readiness can now be judged from saved full-pool artifacts rather
  than from short PPO logs or early-step candidate traces alone.
- The checkpoint-inspection helper makes the saved-adapter assumption explicit:
  decoded-chem PPO saves the final adapter at the run root when adapter files
  are present there; only if those files are missing do we need to fall back to
  a `checkpoint-*` subdirectory.
- The work remains completely outside SFT data construction, SFT training, PPO
  optimization logic, and reward semantics.

### Status
Accepted

---

## [2026-05-17] Add a teacher-confidence filter for PPO prompt pools before continuing long decoded-chem PPO runs

### Background
The reward/teacher audit on the current best short-run checkpoint
`decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_failfix_ckpt500`
showed a split conclusion:

- decoded PPO itself still looked structurally healthy, with
  `cf_oracle_called_rate` near 1.0 and no obvious projection or size loophole;
- but teacher reliability on the label=1 parent pool was only moderately
  trustworthy, with `teacher_correct_rate≈0.855`, `low_confidence_rate≈0.144`,
  and `very_low_confidence_rate≈0.104`.

That means the next priority is not to keep lengthening PPO blindly, nor to
rewrite reward shaping immediately. The more controlled next step is to reduce
teacher-side noise in the PPO prompt pool and compare short filtered runs
against the current baseline.

### Decision
Add a standalone teacher-confidence filtering path for PPO prompt CSV files:

- new filtering logic now lives in `src/data/teacher_confidence_filter.py`;
- `scripts/filter_ppo_prompts_by_teacher_confidence.py` provides a thin CLI
  wrapper that:
  - reads a PPO prompt CSV,
  - resolves `parent_smiles` with fallback to `smiles` and prompt text,
  - scores each parent with `TeacherSemanticScorer`,
  - keeps only rows satisfying target-label, `teacher_result_ok`,
    optional teacher-correctness, and minimum `p_label`;
- `scripts/slurm/filter_ppo_prompts_teacher_p05_label1.sh` hardcodes the
  current label=1 PPO prompt CSV, the AIDS RF teacher, and the
  `p_label >= 0.5 && teacher_correct` filter so HPC usage stays one-command
  simple via
  `sbatch scripts/slurm/filter_ppo_prompts_teacher_p05_label1.sh`;
- `scripts/slurm/train_ppo.sh` now accepts an optional `DATASET_PATH`
  environment variable and forwards it to `--dataset-path`, so filtered prompt
  files can be used without changing PPO training code.

### Alternatives considered
1. Continue training 150/200/300-step PPO on the unfiltered prompt pool first.
2. Immediately redesign the reward function despite the audit not showing a
   clear reward loophole yet.
3. Create a separate custom PPO training wrapper instead of teaching the
   existing Slurm wrapper to accept a dataset override.

### Consequences
- The project can now run a cleaner apples-to-apples experiment:
  unfiltered PPO-100/150 versus teacher-filtered PPO-100/150.
- Teacher-side uncertainty is reduced before spending more A800 time on longer
  decoded-chem PPO runs.
- The change remains outside SFT dataset construction, SFT training, and PPO
  reward logic; it only filters prompt inputs and improves Slurm parameter
  plumbing.

### Status
Accepted

---

## [2026-05-17] Add an independent reward/teacher audit entrypoint for diagnosing PPO-100 vs long-run degradation

### Background
The current decoded PPO workflow reached a point where `PPO-100` looked better
than 150/200/300-step checkpoints on reward, direct-substructure rate, and
parse/core usability, but that trend alone could not distinguish between:

- benign PPO policy drift after longer optimization;
- data-order / prompt-difficulty effects from short sequential training;
- an actual problem in the reward function or teacher/oracle behavior.

The team therefore needed a standalone audit path that could inspect teacher
reliability on original parents and analyze `candidate_pool.jsonl` for
counterfactual-oracle coverage, reward alignment, projection shortcuts, and
size shortcuts, without changing SFT v3 data construction, SFT training, or
PPO main training logic.

### Decision
Add a separate reward/teacher diagnosis entrypoint that stays fully outside the
training loop:

- new audit logic now lives in `src/eval/reward_teacher_audit.py`;
- `scripts/audit_reward_teacher.py` provides a thin CLI for dataset-backed
  teacher reliability plus candidate-pool reward auditing;
- `scripts/slurm/audit_reward_teacher.sh` hardcodes the current best 100-step
  candidate pool, the label=1 PPO prompt dataset, and the AIDS random-forest
  teacher so HPC usage stays one-command simple via
  `sbatch scripts/slurm/audit_reward_teacher.sh`;
- the audit explicitly tolerates candidate-pool schema drift by accepting
  compatibility aliases such as `p_before/teacher_p_before`,
  `p_after/teacher_p_after`, `cf_drop/counterfactual_drop/teacher_cf_drop`,
  `cf_flip/counterfactual_flip`, and
  `counterfactual_reason/cf_reward_skipped_reason`;
- final outputs now separate six questions the team cares about most:
  teacher reliability, cf-oracle skip/deletion failure pressure, reward-to-cf
  alignment, projection loopholes, size loopholes, and whether current
  degradation looks more like PPO drift or reward/teacher trouble.

### Alternatives considered
1. Reuse only PPO training logs and avoid any new structured audit.
2. Fold reward/teacher diagnosis into `train_ppo.py`, making the training
   entrypoint even more coupled to analysis logic.
3. Jump directly to shuffle-100 or multi-seed-100 experiments before first
   checking whether the current reward/teacher pipeline is itself suspicious.

### Consequences
- PPO checkpoint diagnosis becomes reproducible from saved artifacts instead of
  depending on partial training logs.
- Teacher/oracle reliability on original parents can now be audited directly,
  which helps separate model-side label noise from PPO optimization effects.
- Projection and oversized-fragment shortcuts become explicit audit outputs
  instead of vague hypotheses during long-run PPO comparisons.
- The change remains evaluation-only and does not alter SFT data, SFT training,
  or PPO optimization behavior.

### Status
Accepted

---

## [2026-05-14] Add a selector-facing candidate-pool audit entrypoint for checkpoint-level PPO evaluation

### Background
The current decoded PPO workflow has already reached the stage where the
important question is no longer "can PPO run?" but "is a short-run checkpoint a
good candidate-pool generator for the downstream class-level selector?" The
team identified the 100-step run
`decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_failfix_ckpt500` as a
better candidate than the longer 300-step run, which showed more policy drift.
At that point, continuing long PPO training was less urgent than auditing the
quality, diversity, projection dependence, and counterfactual utility of the
already generated `candidate_pool.jsonl`.

### Decision
Add a dedicated audit path for PPO candidate pools without touching SFT v3
dataset construction, SFT training, PPO loss, teacher/oracle logic, or
projection search:

- new selector-facing audit logic now lives in
  `src/eval/candidate_pool_audit.py`;
- `scripts/audit_candidate_pool.py` provides a thin CLI wrapper for JSON/TXT
  reports over `candidate_pool.jsonl`;
- `scripts/slurm/audit_candidate_pool.sh` is pinned to the current 100-step run
  so HPC usage stays one-command simple via
  `sbatch scripts/slurm/audit_candidate_pool.sh`;
- the audit uses field-compatibility fallbacks across reward-trace schema
  variants instead of assuming one rigid JSONL shape, because the pool rows
  evolved during recent projection-penalty and distance-reward debugging;
- final recommendations are driven by selector-oriented heuristics such as
  final-substructure rate, projection-used rate, cf-flip rate, diversity, and
  Morgan-Tanimoto redundancy.

### Alternatives considered
1. Reuse training logs only and skip a structured `candidate_pool.jsonl` audit.
2. Fold candidate-pool analysis into `train_ppo.py`, mixing evaluation logic
   back into the training entrypoint.
3. Continue toward 150-step/200-step/1000-step PPO runs first and postpone
   selector-oriented auditing.

### Consequences
- Short PPO runs can now be evaluated as generator checkpoints before spending
  more compute on longer RL training.
- Candidate-pool readiness for the downstream selector becomes explicit and
  reproducible through saved JSON and TXT reports.
- The audit remains decoupled from training so the class-level selector can
  iterate independently of PPO runtime changes.

### Status
Accepted

---

## [2026-05-14] Make decoded PPO failure traces tolerate forward-compatible reward-debug fields

### Background
After enabling both projection penalty and substructure distance reward in the
decoded PPO diagnostic, normal reward paths worked for the first few samples,
but a later parseable non-direct fragment crashed inside
`ChemRLRewarder._fail(...)` with:

`TypeError: ChemRLRewarder._fail() got an unexpected keyword argument 'projection_penalty_config'`

The reward path had already been extended to merge richer debug dictionaries
containing fields such as `projection_penalty_config`,
`projection_penalty_applied`, `reward_before_projection_penalty`,
`reward_after_projection_penalty`, and `subdist_contribution`. Successful
traces handled those fields, but failure branches still routed everything
through an older `_fail(...)` signature with a closed keyword list. As soon as a
failure path inherited a new trace field, PPO aborted instead of returning a
penalized failure trace.

### Decision
Keep reward semantics unchanged and only harden the failure-trace plumbing:

- `_fail(...)` now accepts the currently expanded projection-penalty fields
  explicitly and also captures future trace extensions through
  `**extra_trace_fields`;
- failure trace construction now uses a small merge helper that appends only
  real `RewardTrace` dataclass fields and never lets unknown logging keys crash
  PPO;
- explicit/core `_fail(...)` arguments still have priority, while future fields
  can populate newly added `RewardTrace` slots without requiring every old call
  site to be rewritten immediately;
- parse-failed, core-unusable, and parseable-but-not-direct branches now all
  keep returning structured failure traces even when projection/distance debug
  fields are present.

### Alternatives considered
1. Add only `projection_penalty_config` to the `_fail(...)` signature and leave
   the rest of the trace field flow unchanged.
2. Strip all non-core debug fields before failure handling and accept reduced
   observability on error paths.
3. Move failure-trace construction out of `_fail(...)` entirely and duplicate
   trace assembly across call sites.

### Consequences
- PPO no longer aborts when a failure branch inherits newer reward-debug fields.
- Failure rows in `CHEM_REWARD_COMPONENTS` and `candidate_pool.jsonl` retain the
  same projection and distance diagnostics as success rows, which keeps reward
  debugging consistent through bad generations.
- Future reward-trace extensions are less likely to require emergency fixes in
  `_fail(...)`, as long as they also become `RewardTrace` fields.

### Status
Accepted

---

## [2026-05-14] Apply projection penalty inside decoded PPO reward breakdown whenever a non-direct fragment needs a successful parent projection

### Background
After enabling both projected-cf reward and substructure distance reward, the
decoded PPO diagnostic logs correctly showed
`[PROJECTED_CF_REWARD_CONFIG] enabled=True`,
`[SUBSTRUCTURE_DISTANCE_REWARD_CONFIG] enabled=True`, and non-zero
`subdist_contribution`. However, `projection_penalty` still stayed at `0.0` in
`[CHEM_REWARD_COMPONENTS]` even when `PROJECTION_PENALTY=1.0` was passed through
the Slurm environment and parseable non-direct fragments were being rescued by a
nearest parent subgraph or by the projected counterfactual path.

The core issue was that projection diagnostics were being carried only as trace
metadata, while reward aggregation never subtracted the configured penalty from
`reward_total`. In the main non-direct branch, later trace merges also replaced
the projection-debug fields with a distance-reward trace that defaulted the
penalty back to `0.0`.

### Decision
Keep the projection search algorithm, teacher/oracle calls, and PPO loss logic
unchanged, and fix only the reward aggregation and observability layer:

- decoded reward breakdowns now carry explicit
  `projection_penalty_config`, `projection_penalty_applied`,
  `reward_before_projection_penalty`, and
  `reward_after_projection_penalty` fields;
- `reward_total` is now the post-penalty value:
  `reward_after_projection_penalty =
  reward_before_projection_penalty - projection_penalty_applied`;
- the penalty is applied whenever a fragment is not a direct parent
  substructure and the nearest-parent projection path succeeded
  (`projection_attempted=True` and `projection_success=True`);
- direct-substructure examples keep `projection_penalty_applied=0.0`;
- parse-failed / core-unusable examples that never reached a successful
  projection path also keep `projection_penalty_applied=0.0`;
- `scripts/train_ppo.py` now also accepts `PROJECTION_PENALTY` as an env-backed
  default for direct local launches, while `scripts/slurm/train_ppo.sh`
  continues forwarding `--projection-penalty`.

### Alternatives considered
1. Apply projection penalty only when
   `used_projected_subgraph_for_reward=True`, leaving other successful nearest
   parent projections unpenalized.
2. Push the penalty into PPO loss code instead of the chemistry reward
   breakdown.
3. Keep penalty logging separate and ask users to post-process logs to estimate
   the effective reward.

### Consequences
- Projection dependence now affects `reward_total` directly instead of being a
  logging-only diagnostic.
- Logs and `candidate_pool.jsonl` rows now show both the configured penalty and
  the actually applied deduction, making it easier to audit whether projected
  rescue is being overused.
- The fix remains local to reward composition and runtime config plumbing; it
  does not touch SFT datasets, PPO update math, teacher/oracle scoring, or the
  parent-projection retrieval algorithm itself.

### Status
Accepted

---

## [2026-05-14] Make substructure distance reward a first-class PPO shaping term with explicit runtime config and contribution logging

### Background
The decoded PPO diagnostic path already computed nearest-parent-subgraph
distance fields such as `substructure_similarity`,
`substructure_distance_reward`, and the shorter `subdist_*` aliases, but recent
projected-cf reward runs still surfaced `subdist_weight=0.0` inside
`[CHEM_REWARD_COMPONENTS]`. In practice that meant the dense distance reward was
being logged without actually contributing to `reward_total`, mostly because the
generic HPC wrapper `scripts/slurm/train_ppo.sh` never forwarded the distance
reward enable/weight knobs even though `scripts/train_ppo.py` and the rewarder
already knew about them.

### Decision
Keep the decoded PPO loss flow, SFT pipeline, teacher/oracle stack, and
projection semantics unchanged, and fix only the reward configuration,
parameter plumbing, and diagnostics:

- `scripts/train_ppo.py` now resolves the canonical runtime switch
  `--enable-substructure-distance-reward` /
  `--no-enable-substructure-distance-reward` explicitly, rejects a CLI conflict,
  supports the env alias `SUBDIST_WEIGHT` alongside
  `SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT`, and logs the resolved runtime state via
  `[SUBSTRUCTURE_DISTANCE_REWARD_CONFIG]`;
- when the feature is disabled, the effective runtime weight is forced to
  `0.0`, so logs and reward traces cannot silently show a stale positive weight;
- `src/rewards/reward_wrapper.py` now surfaces
  `subdist_contribution = subdist_weight * subdist_reward` explicitly in both
  breakdowns and decoded reward logs, while preserving the legacy
  `subdist_weighted_r` key for compatibility;
- `scripts/slurm/train_ppo.sh` now exports, echoes, and forwards the full
  distance-reward knob family so local Codex edits and HPC `sbatch` runs stay in
  sync;
- the recommended default shaping weight is now `0.3`, which keeps distance
  reward active as a conservative continuous constraint without overriding hard
  failure penalties or projected-cf reward behavior.

### Alternatives considered
1. Leave the reward wrapper unchanged and only document that users must switch
   to a dedicated `train_decoded_chem_ppo_subdist_reward.sh` script.
2. Broaden the change into a larger reward refactor that changes PPO loss
   semantics or projection behavior.
3. Introduce a second negative CLI spelling such as
   `--disable-substructure-distance-reward`.

### Consequences
- Enabling distance reward now makes `subdist_weight > 0` and
  `subdist_contribution != 0` visible in reward traces when the fragment earns
  non-zero dense similarity reward.
- Disabled runs remain explicit and unambiguous because both the config log and
  the trace fields resolve to `weight=0.0`.
- Projected counterfactual reward and distance reward can now be diagnosed
  together: projected fragments may still power deletion-based cf reward, while
  the raw/core fragment keeps its own nearest-parent-subgraph shaping term.

### Status
Accepted

---

## [2026-05-10] Add an explicit enable switch for projected counterfactual reward and route successful projections into the deletion teacher

### Background
The decoded PPO stack already surfaced projection diagnostics such as
`projection_attempted`, `projection_success`, and `projected_fragment`, but the
projected counterfactual reward path never actually became live in practice.
Two gaps caused that:

- the CLI/HPC workflow only exposed the legacy negative flag name
  `disable_projected_cf_reward`, so omitting the flag still left the feature
  disabled by default;
- `ChemRLRewarder` logged projected subgraphs for non-direct fragments, but the
  non-direct reward branch still skipped the counterfactual teacher
  unconditionally and never consumed the projected fragment even when a legal
  parent subgraph was available.

### Decision
Keep projected counterfactual reward disabled by default, preserve the old
negative flag for compatibility, and add one explicit positive path:

- `scripts/train_ppo.py` now accepts `--enable-projected-cf-reward` and resolves
  it against the legacy `--disable-projected-cf-reward` flag with an explicit
  conflict error;
- `scripts/slurm/train_ppo.sh` now forwards
  `ENABLE_PROJECTED_CF_REWARD=true/false` and the legacy disable env override;
- `ChemRLRewarder` now uses a projected parent subgraph for deletion-based
  counterfactual reward only when all gating conditions are satisfied:
  projected-cf reward enabled, parent projection enabled, projection attempted
  and successful, and the projected fragment revalidates as a legal parent
  substructure;
- the raw model output stays unchanged in logs and candidate-pool rows; the
  projected fragment is tracked separately through
  `projected_fragment_smiles` / `used_projected_subgraph_for_reward`.

### Alternatives considered
1. Keep the old negative-only flag and rely on `--no-disable-...` style usage.
2. Treat projection diagnostics as logging-only forever and never reuse them for
   counterfactual reward.
3. Replace the model output with the projected fragment downstream.

### Consequences
- Default behavior remains unchanged: projected counterfactual reward is off
  unless the user explicitly enables it.
- When enabled, parseable non-direct fragments can now receive deletion-based
  counterfactual reward through a legal projected parent subgraph without
  pretending that the model originally emitted that projected fragment.
- Logs now make the resolved runtime state explicit via
  `[PROJECTED_CF_REWARD_CONFIG]`, and relevant reward traces surface
  `used_projected_subgraph_for_reward=True` when the projected path is actually
  used.

### Status
Accepted

---

## [2026-05-10] Deduplicate decoded PPO failure-trace kwargs before calling `_fail`

### Background
The decoded chemistry PPO reward path correctly handled early success cases, but
some reward-failure branches accumulated debug fields from multiple trace
dictionaries and then passed them into `ChemRLRewarder._fail(...)` alongside
explicit keyword arguments. When a failure dict already contained fields such as
`direct_substructure`, `projection_attempted`, or
`substructure_distance_reward`, Python raised
`TypeError: ... got multiple values for keyword argument ...` instead of
returning a penalized failure trace.

### Decision
Keep the decoded PPO objective, reward semantics, projection behavior, and
training CLI unchanged, and harden only the failure-trace assembly:

- add a dedicated `_merge_failure_fields(...)` helper in
  `src/rewards/reward_wrapper.py` so `_fail(...)` kwargs are merged into one
  dict before the call;
- update every `_fail(...)` call site that mixed explicit kwargs with trace-dict
  expansion to use the merged call pattern;
- add regression tests covering the public
  `compute_rewards_from_decoded(...)` path for parseable-but-not-direct
  fragments, plus a direct `_fail(...)` assembly test where the extra trace dict
  already contains `direct_substructure=False`.

### Alternatives considered
1. Remove only the current duplicated `direct_substructure` field from one
   failing branch.
2. Drop projection/subdistance diagnostics from failure traces entirely.
3. Broaden the change into a larger PPO reward refactor.

### Consequences
- Reward failure branches such as parseable non-substructures now return logged
  negative/low-value traces instead of aborting PPO with a keyword-collision
  exception.
- Existing diagnostics remain visible in `[CHEM_REWARD_COMPONENTS]`, including
  `failure_tag`, `invalid_detail`, `direct_substructure`,
  `projection_attempted`, `projection_success`, `projection_method`, and
  `reward_total`.
- The fix stays local to reward-trace assembly and does not change SFT data,
  PPO loss flow, teacher scoring logic, or HPC launch semantics.

### Status
Accepted

---

## [2026-05-10] Standardize decoded PPO initialization on SFT_LORA_PATH with INIT_LORA_PATH alias and explicit init logging

### Background
The decoded chemistry PPO path already expected an SFT-initialized LoRA policy,
but the repository exposed that checkpoint through a mix of names depending on
which layer a user looked at: `--sft-lora-path` in `scripts/train_ppo.py`,
`SFT_LORA_PATH` in the generic Slurm wrapper, and several older diagnostic
scripts that still hardcoded a specific checkpoint path. This made it too easy
to launch PPO from the wrong adapter when switching from an older checkpoint to
the newer SFT v3 HIV runs.

### Decision
Keep `SFT_LORA_PATH` / `--sft-lora-path` as the canonical decoded PPO
initialization path, and add one narrow compatibility alias plus stronger
runtime logging:

- `scripts/train_ppo.py` now accepts `--init-lora-path` as a compatibility
  alias, with precedence `--sft-lora-path` > `--init-lora-path` >
  `--sft-adapter-path`;
- the PPO runtime manifest and logs record both the raw init-path arguments and
  the final resolved checkpoint path plus its source field;
- `scripts/slurm/train_ppo.sh` and
  `scripts/slurm/train_decoded_chem_ppo_full.sh` now accept `INIT_LORA_PATH`
  as an environment-variable alias, resolve one final init LoRA path, echo it,
  and warn if both alias names are set to different values.

### Alternatives considered
1. Leave the existing `SFT_LORA_PATH` support unchanged and rely on users to
   inspect each Slurm script manually.
2. Rename every PPO entrypoint to one new variable and break the older wrappers.
3. Broaden the compatibility layer to ambiguous names such as `CHECKPOINT_PATH`.

### Consequences
- The canonical answer for decoded PPO initialization remains:
  use `SFT_LORA_PATH` / `--sft-lora-path`.
- Existing workflows keep working, while `INIT_LORA_PATH` can be used as a
  clear compatibility alias in shared shell snippets.
- PPO logs now make it obvious which LoRA checkpoint was actually loaded for
  policy/reference initialization.

### Status
Accepted

---

## [2026-05-10] Normalize legacy SFT JSONL columns to TRL prompt-completion format at train time

### Background
The rebuilt SFT v3 datasets and some older SFT exports were still centered on
`instruction` / `output` audit fields, and the current HIV builder also emitted
`prompt` plus `response` without a `completion` alias. On HPC, TRL's
`SFTTrainer` entered prompt-completion tokenization mode and failed early with
`KeyError: 'completion'`, even though the train/validation splits themselves
were valid and readable.

### Decision
Keep the existing SFT data objective, candidate generation, filtering, and PPO
logic unchanged, and add a narrow compatibility layer for SFT-only training:

- `scripts/train_sft.py` now normalizes loaded JSONL rows before constructing
  `SFTTrainer`, preserving legacy audit fields while materializing
  `prompt` / `completion` when possible;
- the normalization supports three input shapes:
  direct `prompt` / `completion`, legacy `instruction` / `output` with optional
  `input`, and the current builder's `prompt` / `response` alias;
- completions are prefixed with a separator newline when synthesized so prompt
  and fragment text do not concatenate silently;
- train/eval startup now logs normalized column names plus prompt/completion
  previews, and raises a clear `ValueError` with available columns when a split
  still lacks the required fields;
- the SFT v3 builder now writes a `completion` alias in new JSONL outputs so
  future datasets are directly compatible with TRL prompt-completion mode.

### Alternatives considered
1. Require every existing SFT dataset to be rebuilt before retraining.
2. Patch TRL usage to rely only on a concatenated `text` column and skip
   prompt-completion compatibility entirely.
3. Broaden the fix into a larger SFT data-schema refactor.

### Consequences
- Existing `instruction` / `output` JSONL files can be trained directly without
  a dataset rebuild.
- Current builder outputs are now safer for TRL because `completion` is written
  explicitly in addition to the preserved legacy fields.
- SFT failures around missing text columns now surface with actionable dataset
  diagnostics instead of an internal TRL `KeyError`.

### Status
Accepted

---

## [2026-05-09] Add nearest-parent-subgraph distance reward and stop using projected fragments for counterfactual reward

### Background
The decoded PPO path previously had one high-risk mismatch with the v3
counterfactual objective: when a parseable core fragment was not a strict
parent substructure, the reward wrapper could retrieve one projected legal
parent subgraph and then continue deletion-based reward computation on that
projected fragment instead of on the model's own output. This hid the true
failure mode and turned the projection module into an answer-rewriting path
rather than a structural-distance diagnostic.

### Decision
Keep the strict exact-substructure reward, but separate it from a new dense
auxiliary reward based on the nearest legal connected parent subgraph:

- direct parent matches keep the exact binary substructure reward and remain the
  only cases allowed to call the deletion-based counterfactual teacher;
- parseable but non-direct fragments now compute
  `substructure_similarity / substructure_distance / substructure_distance_reward`
  against the nearest legal parent subgraph built from the existing
  parent-derived candidate pool;
- the nearest legal parent subgraph is logged for debugging only and is never
  substituted back into the reward path for counterfactual deletion scoring;
- non-direct fragments explicitly log
  `used_projected_subgraph_for_reward=False` and
  `cf_reward_skipped_reason=not_direct_substructure`;
- the decoded PPO CLI and Slurm path expose dedicated knobs for enabling the
  new dense reward and tuning its candidate window and MCS settings.

### Alternatives considered
1. Keep the projection-retrieval path as the effective reward fragment and only
   add more logging around it.
2. Treat all parseable non-substructure outputs as hard failures with zero
   dense structural feedback.
3. Replace the distance reward with fragment-only teacher semantics.

### Consequences
- Reward logs now distinguish exact direct substructure success from
  non-direct-but-similar fragments.
- Deletion-based counterfactual reward is again aligned with the model output
  instead of a projected replacement fragment.
- PPO can still receive a dense structural signal for near-miss fragments
  without allowing reward leakage through projection.

### Status
Accepted

---

## [2026-05-09] Replace the SFT v3 HIV scaffold split with a label-stratified scaffold holdout

### Background
The rebuilt raw-HIV SFT v3 dataset was already structurally healthy, but the
existing `scaffold_group_greedy` split could severely distort validation label
balance. In the observed full build, validation drifted toward almost all
positives (`{'0': 24, '1': 404}`), making the split unrepresentative even
though scaffold overlap remained zero.

### Decision
Keep candidate generation, fragment filtering, oracle ranking, and text target
format unchanged, and only replace the train/val split logic in
`src/data/sft_v3_builder.py`:

- make the default split objective explicitly label-stratified at the scaffold
  group level, so validation selection optimizes per-label target counts before
  raw total-count closeness;
- preserve scaffold-level holdout by assigning each effective scaffold group to
  exactly one split;
- treat missing/acyclic scaffolds as stable per-example pseudo-scaffolds during
  splitting so they do not collapse into one oversized group;
- surface split diagnostics such as total/target/actual label counts,
  per-label target error, actual val ratio, and per-label val ratio.

### Alternatives considered
1. Keep the old global scaffold greedy split and only tweak weights slightly.
2. Fall back to pure label-stratified random splitting and give up scaffold
   holdout.
3. Rebuild the dataset again with different parent sampling instead of fixing
   the split itself.

### Consequences
- Validation should remain much closer to the global 2:1 label mix while still
  keeping scaffold overlap at zero in normal cases.
- Large scaffold groups can still introduce small count error, but the error is
  now explicit in the split summary/report instead of being silent.
- The rebuild path stays fully compatible with the existing SFT build, audit,
  train, and eval scripts.

### Status
Accepted

---

## [2026-05-08] Restore and harden the raw HIV -> SFT v3 rebuild path as a first-class workflow

### Background
The repository already had a scaffold-aware SFT v3 rebuild pipeline, but the
active worktree had lost `src/data/hiv_dataset_utils.py`, which broke the raw
HIV.csv normalization and parent-sampling path outright. At the same time, the
HPC build wrapper and human-readable reports needed a bit more explicit
bookkeeping so the negative-pool sampling behavior remained easy to audit when
rebuilding a larger SFT initializer for later PPO runs.

### Decision
Keep the existing raw-HIV -> parent-derived reference build strategy, and make
the following operational hardening changes instead of inventing a new data
objective:

- restore `src/data/hiv_dataset_utils.py` as the source of truth for flexible
  HIV column detection, label normalization, scaffold extraction, and
  scaffold+size diversity sampling;
- preserve the current parent-derived candidate path in
  `src/data/sft_v3_builder.py`, but expose clearer selection/report metadata
  for positive/negative queue sizes, stratum counts, and raw label tokens;
- standardize the HPC default path layout under
  `outputs/hpc/sft_v3_hiv_runs/<RUN_NAME>/...` so dataset build, audit, train,
  and eval can share one experiment name without hand-editing multiple paths;
- add a login-node submission helper that emits the full Slurm dependency graph
  from one command, keeping build as the root stage, launching audit and train
  after build, and launching eval after train;
- keep the paired Slurm builder wrapper in sync by ensuring the warn-log parent
  directory is created before stderr teeing;
- extend tests so the rebuild path is checked against both numeric HIV labels
  and string-valued class aliases.

### Alternatives considered
1. Re-implement the builder around the older `prepare_sft_data.py` flow.
2. Patch only the import error and leave the reporting/slurm path unchanged.
3. Introduce a new sampling objective before the current v3 path was even
   operational again.

### Consequences
- The repo once again has a working, auditable `HIV.csv -> SFT v3` builder
  compatible with the current training/eval scripts.
- Sampling summaries now make it clearer how many negative parents are
  available versus how many are actually targeted for successful selection.
- The default HPC workflow is now easier to operate because `RUN_NAME` can
  identify the full dataset/audit/train/eval artifact tree.
- The HPC workflow is now one-command submit friendly without giving up the
  stage-specific Slurm wrappers.
- The HPC `sbatch` path is less brittle because warn-log teeing no longer
  depends on the log directory already existing.

### Status
Accepted

---

## [2026-04-27] Rebuild SFT v3 from raw HIV.csv with scaffold-aware parent sampling and parent-derived reference ranking

### Background
The repository's earlier SFT data path was built from a much smaller weak-target
pool and inherited two distribution problems: too many near-parent large
fragments and a non-trivial tail of trivial tiny fragments. For the current v3
counterfactual objective, the SFT initializer should expose the model to more
strict parent substructures whose deletion leaves a non-empty residual and
whose size distribution is centered in a usable mid-range before PPO starts.

### Decision
Add a new `scripts/build_sft_v3_from_hiv.py` pipeline backed by
`src/data/sft_v3_builder.py` and `src/data/hiv_dataset_utils.py`:

- read raw `HIV.csv` directly with flexible field-name detection for SMILES and
  labels;
- normalize parent molecules with RDKit canonicalization and build scaffold +
  parent-size metadata;
- keep positive-class parents aggressively while downsampling negatives with a
  scaffold-and-size-diversity-first round-robin strategy instead of raw
  full-retention or rigid 1:1 balancing;
- generate reference candidates only from parent-derived connected fragments,
  primarily through the existing projection candidate pool
  (ring systems, functional-group neighborhoods, BRICS components, atom/bond
  k-hop fragments) plus a lightweight Murcko-like scaffold path;
- filter candidates by strict parent-substructure matching, non-empty deletion
  residual, non-full-parent status, and a default mid-size atom-ratio window of
  `[0.10, 0.55]`;
- if an oracle bundle is provided, weak-rank filtered candidates by
  `cf_flip -> cf_drop -> size closeness`; otherwise fall back to a size-aware
  heuristic ranking;
- keep the final text target as core-only fragment SMILES while preserving the
  recovered dummy-capped explanation fragment in metadata.

### Alternatives considered
1. Keep extending the old `prepare_sft_data.py` weak-target path.
2. Rebuild SFT references from arbitrary fragment-only teacher semantics rather
   than deletion-based parent-derived candidates.
3. Preserve all negatives and rely on later SFT/PPO reweighting to correct the
   data imbalance.

### Consequences
- The project now has a direct raw-HIV -> SFT-v3 rebuild path that is aligned
  with the residual-graph counterfactual objective.
- SFT train/val JSONL outputs remain compatible with `scripts/train_sft.py`,
  `scripts/eval_sft_fragment_quality.py`, and
  `scripts/analyze_sft_fragment_distribution.py`.
- HPC workflows can rebuild, train, and evaluate the new dataset through
  dedicated Slurm wrappers without changing the local VSCode -> `git push` ->
  HPC `git pull` -> `sbatch` loop.

### Status
Accepted

---

## [2026-04-26] Switch SFT and decoded PPO text targets to core-only fragments

### Background
Decoded PPO diagnostics showed repeated failure buckets around raw dummy-atom
targets: `parse_failed`, `invalid_or_not_substructure`, and
`core_fragment_unusable_after_dummy_normalization`. The project objective is
still deletion-based counterfactual subgraph generation, but requiring the LLM
to emit capped `*...*` fragments was unnecessarily expanding the text search
space and entangling text generation with RDKit attachment-point bookkeeping.

### Decision
Adopt `v3_core` / `core_no_dummy` as the default text target for the current
SFT and decoded PPO path:

- SFT dataset responses now store no-dummy `core_fragment` strings while
  preserving the original dummy-bearing fragment as metadata;
- PPO prompts and core-mode prompt builders now instruct the model to emit only
  connected core-fragment SMILES without `*`;
- RDKit remains responsible for strict parent-substructure matching,
  parent-constrained projection, boundary-bond detection, and optional recovery
  of an explanation fragment with dummy attachment markers;
- deletion-based teacher-oracle scoring continues to operate on the strict or
  projected parent subgraph, not on fragment-only teacher semantics;
- decoded PPO keeps projection and repair scaffolding, but dummy output is now
  treated as a warning plus light penalty in core mode instead of being the
  desired text format.

### Alternatives considered
1. Keep dummy-bearing targets and only patch parse/salvage heuristics.
2. Remove dummy handling from the codebase entirely.
3. Switch to a graph-only generator and bypass SMILES decoding altogether.

### Consequences
- New `data/sft_v3_core_train.jsonl` and `data/sft_v3_core_val.jsonl` datasets
  can coexist with legacy dummy-target datasets.
- Core-only eval summaries now report dummy-output and stripped-core recovery
  metrics explicitly.
- Decoded PPO candidate pools now retain both core fragments and RDKit-restored
  explanation fragments with dummy attachment points, so diagnostics remain
  available without making dummy atoms part of the model target.

### Status
Accepted

---

## Template

```md
## [YYYY-MM-DD] Decision title

### Background
Why was this decision needed?

### Decision
What was decided?

### Alternatives considered
What other options were considered?

### Consequences
What changes because of this decision?

### Status
Proposed / Accepted / Deprecated / Superseded
```

---

## [2026-04-26] v4 minimal patch for decoded PPO repair, salvage semantics, and size-window reward

### Background
The `decoded_chem_diag50_parsefix_connectfix_v3` diagnose run kept the
projection-v1 retrieval path alive and activated the tiny-fragment hard guard,
but three problems remained: minimal syntax repair still failed mostly at
`repair_candidate_parse_failed`, component salvage logs still mixed raw/core
disconnects with core-unusable cases, and the policy could still oscillate
between tiny fragments and near-parent fragments.

### Decision
Keep projection-v1, the decoded PPO main loop, and the existing Slurm argument
chain intact, and apply only a narrow v4 repair-path patch:

- upgrade minimal syntax repair from single accepted candidate to
  multi-candidate generation with staged diagnostics
  (parse/core/strict-parent/projection);
- let repair candidates prove they are either strict parent subgraphs or
  projectable through the existing parent-constrained retrieval path before
  counting as `repair_success=True`;
- restrict component salvage to true raw/core disconnected fragments, and label
  core-unusable normalization failures explicitly instead of routing them
  through `fragment_not_connected`;
- add a soft size-window reward on the final accepted fragment atom ratio while
  preserving tiny-fragment, near-parent, and tiny-residual hard fails;
- add a dedicated v4 diagnose Slurm wrapper with all arguments encoded in the
  script.

### Alternatives considered
1. Add a global nearest-valid-molecule repairer after parse failure.
2. Rewrite projection-v1 instead of keeping the existing retrieval path.
3. Rework the entire reward framework instead of patching the decoded PPO
   reward wrapper in place.

### Consequences
- Repair logs now distinguish whether failure happened at parse, core
  normalization, strict-parent validation, or repair-time projection.
- Component salvage logs now distinguish raw/core disconnected inputs from
  non-salvageable core normalization failures.
- The final accepted fragment now carries explicit size-window diagnostics
  without weakening existing hard guards.

### Status
Accepted

---

## [2026-04-26] Tighten decoded PPO parse/connect diagnostics and tiny-fragment guard

### Background
The `decoded_chem_diag50_parsefix_connectfix_v2` diagnose run showed that
minimal syntax repair was attempted but never surfaced successful repaired
candidates, disconnected fragments were not reliably entering component
salvage, and the policy was beginning to exploit very small fragments such as
`O`, `S`, and `N=O`.

### Decision
Keep projection-v1 and the existing decoded PPO Slurm parameter chain intact,
and make a narrow reward-path patch:

- expand minimal syntax repair diagnostics with candidate counts, acceptance,
  and fine-grained failure reasons;
- detect raw and core disconnected components before strict substructure checks
  and run component salvage on the disconnected representation;
- add a diagnose-configurable tiny-fragment hard fail after strict/projected
  fragment resolution, so projection success cannot bypass the minimum atom
  constraint;
- add a v3 50-step Slurm wrapper that enables the new guard with
  `MIN_FRAGMENT_ATOMS=3` and `TINY_FRAGMENT_HARD_FAIL_PENALTY=-6.0`.

### Alternatives considered
1. Rewrite parse-failed outputs into nearest valid molecules.
2. Move tiny-fragment suppression into the prompt only.
3. Rebuild projection-v1 instead of preserving the existing retrieval path.

### Consequences
- Repaired parseable candidates now continue into the same strict parent
  subgraph / retrieval projection path rather than being treated as a separate
  unconstrained repair objective.
- Diagnose logs distinguish raw/core component counts and salvage failure
  stages.
- Very small strict or projected fragments receive a fixed hard-fail reward in
  the new diagnose script, preventing positive reward terms from masking tiny
  fragment collapse.

### Status
Accepted

---

## [2026-04-25] Add parent-constrained retrieval projection for decoded PPO non-substructure fragments

### Background
Decoded-chem PPO now surfaces parse failures separately from parseable fragments
that are not valid parent substructures. The latter failure bucket is actionable:
the raw fragment already has a chemically parseable shape, but the exact graph
does not occur in the parent molecule. For the current instance-level candidate
generator, these cases should be repaired by projecting onto strict
parent-derived candidates before reward and oracle scoring.

### Decision
Add an optional parent-constrained candidate retrieval projection path for
decoded PPO:

- keep parse-failed raw fragments on the existing failure path;
- mark parseable strict parent substructures as `projection_method=identity`;
- when a parseable connected core is not a parent substructure, build a
  parent-derived candidate pool from ring systems, SMARTS functional-group
  neighborhoods, atom-centered k-hop neighborhoods, bond-centered k-hop
  neighborhoods, and stable parent-index BRICS components;
- filter candidates to connected, RDKit-parseable, non-full-parent subgraphs
  whose deletion leaves a non-empty residual;
- score candidates with Morgan Tanimoto, MCS atom coverage, functional-group
  overlap, atom-count difference, and a large-fragment penalty;
- if the best score passes the configured threshold, continue reward and
  deletion-oracle scoring on the projected fragment and subtract a projection
  penalty from `reward_total`;
- expose projection controls through `scripts/train_ppo.py`, decoded PPO logs,
  and Slurm wrappers.

### Alternatives considered
1. Reuse the existing parent-aware repair path without adding k-hop candidates,
   strict deletion filtering, or projection-specific logs.
2. Penalize all parseable non-substructure outputs without attempting a
   parent-constrained projection.
3. Rewrite raw fragments directly with string heuristics instead of retrieving
   from parent atom-index subgraphs.

### Consequences
- The dominant `parse_ok_but_not_substructure` failure bucket can now produce
  rewardable parent-derived fragments without changing parse-failed behavior.
- Logs now record projection attempt/success, retrieval score/source, projected
  fragment, atom statistics, candidate count, and applied penalty.
- HPC diagnosis can run fixed 50-step and 200-step projection jobs through
  dedicated Slurm scripts without hand-written `sbatch --export` arguments.

### Status
Accepted

---

## [2026-04-25] Wire parent-aware repair controls through decoded PPO CLI and Slurm wrappers

### Background
The decoded PPO rewarder already supported one optional parent-aware repair
attempt for broken decoded fragments, but the training entrypoint and the HPC
Slurm wrappers did not expose those controls. As a result, users could set
environment variables such as `ENABLE_PARENT_AWARE_REPAIR=true` in `sbatch`
commands without the settings ever reaching `ChemRLRewarder`.

### Decision
Expose the existing repair controls end-to-end without changing the reward
objective:

- add `--enable-parent-aware-repair`,
  `--repair-min-similarity`, and `--repair-max-candidates` to
  `scripts/train_ppo.py`;
- pass the parsed values into `ChemRLRewarder`;
- log the resolved repair configuration at startup;
- forward the corresponding environment variables through
  `scripts/slurm/train_decoded_chem_ppo_full.sh` and
  `scripts/slurm/train_ppo.sh`.

### Alternatives considered
1. Leave repair available only as a code-level option in `reward_wrapper`.
2. Ask users to edit the Slurm shell scripts manually for each experiment.
3. Rework repair behavior inside the rewarder without exposing it through the
   CLI.

### Consequences
- `sbatch --export=ALL,ENABLE_PARENT_AWARE_REPAIR=...` style launches now
  actually affect decoded PPO runs.
- HPC diagnose jobs can sweep repair settings in the same way they already
  sweep decoded generation settings.
- The reward behavior itself is unchanged unless the user explicitly enables
  repair.

### Status
Accepted

---

## [2026-04-25] Preserve raw dummy-atom evidence in decoded PPO parse-failure logs

### Background
The decoded-chem PPO reward path already used dummy-aware normalization for
successful capped fragments such as `*CC(=O)O`, but parse-failed fragments were
still difficult to diagnose from logs alone. In particular, once raw parsing
failed, the existing trace fields could lose the evidence that the original
fragment string actually contained `*`, which made it hard to tell whether
failures mostly came from uncapped raw fragments or from the dummy-atom path
itself.

### Decision
Keep the decoded PPO chemistry objective unchanged and apply a minimal logging /
normalization refinement only in `reward_wrapper` and `train_ppo`:

- preserve the raw fragment string as the source of truth for dummy presence and
  dummy count before any RDKit parsing happens;
- continue to parse the raw fragment with `*` intact and never do string-level
  `replace("*", "")` before `MolFromSmiles`;
- keep `core_fragment` as a derived post-parse view used for teacher scoring and
  deletion checks only;
- surface explicit parse metadata in decoded PPO logs, including
  `raw_has_dummy`, `raw_dummy_count`, `parse_stage`,
  `parsed_raw_with_dummy`, `parsed_core`, `dummy_removed_before_parse`, and
  `parse_failed_reason`;
- split parse-failure buckets into
  `parse_failed_raw_with_dummy`, `parse_failed_raw_without_dummy`,
  `parse_failed_after_dummy_removal`, plus the existing obvious closure buckets
  such as `parse_failed_unclosed_ring` and
  `parse_failed_unbalanced_parentheses`;
- add per-batch counters for parse failures with and without raw dummy atoms.

### Alternatives considered
1. Keep the existing reward path unchanged and infer dummy-related failures only
   from ad hoc grep patterns.
2. Strip dummy atoms from strings before parsing so that all failures collapse
   onto core-fragment syntax.
3. Attempt automatic repair of ring digits or parentheses during reward-time
   normalization.

### Consequences
- Diagnose logs can now answer whether parse failures mostly come from raw
  fragments without `*` or from dummy-aware normalization paths.
- Successful capped fragments still use the same raw-then-core workflow, so the
  counterfactual objective and deletion logic do not change.
- The codebase now makes it explicit that dummy removal happens after raw
  parsing, not before it.

### Status
Accepted

---

## [2026-04-25] Harden decoded-chem PPO against overlong invalid fragments and surface failure buckets explicitly

### Background
After the second SFT round, decoded-chem PPO could initialize from
`checkpoint-300`, keep policy/reference aligned, and run a full 200-step
diagnose loop. The dominant failure mode, however, was no longer empty
responses. Instead, many generations failed as invalid or non-substructure
fragments, often because the decoded fragment was too long, truncated, or left
rings / brackets / parentheses unclosed. At the same time, full-parent and
empty-residual failures were still present, but their penalties were not strong
enough to clearly dominate those degenerate behaviors.

### Decision
Keep candidate-pool, selector, and SFT logic unchanged, and apply a minimal
decoded-chem PPO hardening pass only on generation / reward constraints:

- add decoded-generation-specific CLI knobs
  (`--gen-max-new-tokens`, `--gen-temperature`, `--gen-top-p`,
  `--gen-do-sample`) while keeping the legacy PPO generation flags usable;
- when decoded-chem PPO is launched without an explicit generation-length
  override, tighten its default `max_new_tokens` to `48`;
- preprocess decoded fragments before chemistry reward by stripping whitespace,
  keeping only the first line, and rejecting overlong fragments as
  `invalid_generation_too_long`;
- keep parse failures on the normal invalid path, but add an explicit
  `invalid_detail` field for obvious closure issues such as unbalanced
  parentheses, brackets, or ring digits;
- increase the default full-parent / empty-residual penalties to `-6.0` and
  `-4.0`, respectively;
- log `failure_tag`, `invalid_detail`, and generated fragment length alongside
  the existing `CHEM_REWARD_COMPONENTS` fields so bad cases can be grepped
  directly from decoded PPO diagnose runs;
- forward the new decoded-generation controls through the main HPC Slurm
  wrappers using `sbatch --export=ALL,...`.

### Alternatives considered
1. Leave generation settings untouched and only adjust the reward penalties.
2. Solve the issue in the candidate pool or selector instead of in decoded PPO.
3. Add aggressive chemistry-aware truncation that rewrites generated fragments
   rather than rejecting obviously bad strings early.

### Consequences
- Decoded PPO now pushes back earlier on the observed overlong / truncated
  invalid fragments without changing the project objective.
- Logs can distinguish `invalid_generation_too_long`,
  `invalid_or_not_substructure`, `full_parent_fragment`, and `empty_response`
  directly.
- HPC diagnose runs can sweep decoded generation settings through Slurm exports
  instead of editing shell scripts by hand.

### Status
Accepted

---

## [2026-04-25] Let decoded-chem PPO initialize both policy and reference from one explicit SFT LoRA checkpoint

### Background
The decoded chemistry PPO path is meant to start from an SFT policy rather than
from the raw base model. That becomes especially important once we want to run
PPO from a chosen SFT v2 checkpoint such as `checkpoint-300`. If the trainable
policy starts from that checkpoint but the KL reference remains the bare base
model, the KL term is misaligned and can exaggerate drift. At the same time,
collapsed generations such as empty responses, full-parent fragments, and
empty-residual deletions need to be surfaced explicitly in logs rather than
appearing as opaque chemistry failures.

### Decision
Keep the decoded-chem PPO objective unchanged, but make initialization and
anti-collapse diagnostics explicit:

- add `--sft-lora-path` as the preferred CLI name for the SFT initialization
  checkpoint while keeping the old `--sft-adapter-path` path for backward
  compatibility;
- resolve one effective SFT LoRA path and use it for both the trainable PPO
  policy and the frozen KL reference model;
- log the resolved policy/reference initialization path so HPC runs can verify
  that both models start from the same checkpoint;
- treat empty decoded responses as empty fragments instead of letting prompt
  echo accidentally fall back to the parent molecule;
- add explicit `full_parent` and `empty_residual` handling with configurable
  penalties and stable log fields (`empty_response`, `full_parent`,
  `empty_residual`, `oracle_ok`, `cf_drop`, `cf_flip`, `reward_total`).

### Alternatives considered
1. Keep using only `--sft-adapter-path` and rely on users to infer whether the
   reference model matches the policy.
2. Let the KL reference remain the raw base model even when PPO starts from an
   SFT adapter.
3. Leave empty-response and full-parent cases to be inferred indirectly from
   teacher-oracle failure reasons.

### Consequences
- PPO runs can now be launched from an explicit SFT v2 checkpoint such as
  `checkpoint-300` with policy/reference alignment preserved.
- Decoded-chem logs are easier to grep for collapse-related cases without
  changing the underlying deletion-based counterfactual objective.
- The full-training Slurm wrapper can forward the chosen SFT LoRA checkpoint
  and the new explicit penalties through `sbatch --export=ALL,...`.

### Status
Accepted

---

## [2026-04-25] Make SFT fragment-distribution audits chunkable and existence-first for HPC runs

### Background
The new SFT audit scripts are intended to characterize whether weak labels and
SFT generations collapse toward near-parent fragments or tiny trivial pieces.
On larger files, however, a few symmetric molecules caused audit runs to stall
for a long time inside RDKit substructure enumeration, which made whole-file
audits unreliable on both local machines and HPC nodes.

### Decision
Keep the audit objective unchanged, but make the audit path operationally safe:

- add chunk/window controls and progress logging to
  `scripts/analyze_sft_fragment_distribution.py`;
- isolate per-sample audit exceptions so one bad molecule does not abort the
  entire batch unless `--fail-fast` is explicitly requested;
- prefer existence-first substructure checks (`HasSubstructMatch` or capped
  queries limited to `maxMatches=1`) when the audit only needs to know whether
  a match exists;
- add cheap pruning before expensive chemistry work, including parse failures,
  fragment-larger-than-parent checks, and a full-parent shortcut based on
  canonical core equality;
- emit slow-sample records so long-running parent/fragment pairs can be
  inspected and re-run in isolated chunks on HPC.

### Alternatives considered
1. Keep the original all-in-one audit and rely on manual interruption when a
   job stalls.
2. Disable substructure and deletion checks globally, even for manageable
   molecules.
3. Rewrite the chemistry layer around a separate matching backend.

### Consequences
- SFT audits can now be submitted in bounded chunks through Slurm.
- Large runs produce explicit progress, warning, and slow-sample artifacts
  instead of appearing silently hung.
- The chemistry layer still enforces the same counterfactual-fragment
  definition, but audit-time existence checks avoid enumerating every symmetric
  match when only a yes/no answer is needed.

### Status
Accepted

---

## [2026-04-19] Add explicit teacher-semantic scoring on core fragments in the decoded chemistry PPO path

### Background
The decoded chemistry PPO loop now proves that generated text is decoded,
normalized, scored by chemistry utilities, and then used in a PPO update.
However, the logs still showed:

- `[CHEM_REWARD_COMPONENTS_MISSING] missing=teacher_sem`

That made it impossible to tell whether any auxiliary fragment-level semantic
signal was actually being applied after dummy-atom normalization.

### Decision
Add a dedicated `TeacherSemanticScorer` and wire it into the decoded chemistry
reward path only.

Key rules:

- the teacher always receives `core_fragment_smiles`, never the raw capped
  fragment with `*`;
- invalid or non-substructure fragments do not call the teacher and instead log
  a skip reason;
- when the teacher backend is unavailable, the code uses an explicit fallback
  penalty and logs the unavailability rather than pretending a real score
  exists;
- the repository's existing residual-molecule counterfactual term remains in
  place, so the teacher-semantic term is an auxiliary signal rather than a
  replacement for the deletion-based objective.

Because the repository currently ships one concrete classifier artifact at
`outputs/hpc/oracle/aids_rf_model.pkl`, the scorer first supports that
scikit-learn style bundle format (`predict_proba` plus fingerprint metadata).
Torch checkpoints are only accepted when they carry equally explicit
fingerprint configuration.

### Alternatives considered
1. Continue logging `teacher_sem` as missing.
2. Treat the residual-molecule oracle as if it already satisfied the teacher
   role and hide the distinction.
3. Require a new `teacher/teacher.pt` artifact before any teacher-semantic work
   could proceed.

### Consequences
- decoded PPO logs now expose `[TEACHER_SEM_CALLED]`,
  `[TEACHER_SEM_RESULT]`, `[TEACHER_SEM_SKIPPED]`, and
  `[TEACHER_SEM_UNAVAILABLE]`;
- `CHEM_REWARD_COMPONENTS` now shows both `teacher_sem` and the residual
  counterfactual term separately;
- the decoded chemistry smoke-test Slurm script now checks for a teacher file
  before submitting training.

### Status
Accepted

---

## [2026-04-19] Treat dummy-atom attachment points as valid decoded-fragment syntax in PPO chemistry rewards

### Background
The decoded chemistry PPO path now makes reward computation explicit, but the
rewarder was still too harsh on fragment strings containing `*`, for example
`*CC(=O)O`. In this project those stars are not arbitrary garbage characters;
they encode attachment points created by fragment cutting. If the rewarder
treated them as invalid text, PPO would incorrectly learn that many chemically
meaningful capped fragments were malformed.

### Decision
Keep two fragment views inside `src/rewards/reward_wrapper.py`:

- `raw_fragment_smiles`: the exact decoded fragment candidate, which may contain
  dummy atoms such as `*`;
- `core_fragment_smiles`: a dummy-free core used for substructure checks,
  compactness statistics, and any future fragment-level teacher signal.

The rewarder now:

- parses capped fragments with dummy-aware RDKit sanitization;
- removes dummy atoms through molecule editing instead of string replacement;
- checks parent substructure on the core fragment rather than on the raw capped
  string;
- counts fragment size on non-dummy atoms only;
- exposes raw/core parse flags and dummy counts in reward traces and decoded PPO
  logs.

### Alternatives considered
1. Keep treating all `*` tokens as invalid output.
2. Strip `*` with naive string replacement before every reward computation.
3. Move the dummy-aware normalization into TRL adapters instead of the chemistry
   rewarder.

### Consequences
- Decoded PPO logs can now distinguish raw capped fragments from their
  dummy-free core.
- Validity and substructure rewards no longer collapse to zero solely because a
  fragment uses attachment-point notation.
- The chemistry reward path remains honest about what is still missing, such as
  any dedicated teacher-semantic term.

### Status
Accepted

---

## [2026-04-18] Add a decoded-SMILES chemistry reward PPO loop alongside the TRL compatibility baseline

### Background
The repository's TRL experimental PPO smoke test now runs end to end, but the
successful path still relies on a hidden-state reward adapter that only
validates trainer interface compatibility. The logs already make this
limitation explicit:

- `ChemRewardModelWrapper remains the chemistry reward component and is not equivalent to TRL hidden-state reward head`

That means the baseline does not prove that decoded fragment strings are being
scored by the chemistry reward and then used for PPO updates.

### Decision
Keep the TRL experimental path as the engineering baseline, but add a second
training mode in `scripts/train_ppo.py`:

- `--ppo-loop decoded_chem`

This mode performs the reward flow explicitly:

- prompt batch
- `policy_model.generate()`
- decode generated response text
- extract one fragment candidate
- call `ChemRLRewarder.compute_rewards_from_decoded(...)`
- run a local PPO-style update with policy logprobs, reference logprobs,
  token-aligned value predictions, KL-shaped rewards, clipped policy loss, and
  clipped value loss

The CLI also now supports:

- `--require-chemistry-reward-path`
- `--decoded-chem-smoke-test`

and the repository includes a dedicated Slurm smoke-test script:

- `scripts/slurm/debug_decoded_chem_ppo_smoketest.sh`

### Alternatives considered
1. Keep using the TRL hidden-state reward adapter and treat it as good enough.
2. Patch TRL site-packages until they can consume the chemistry wrapper
   directly.
3. Block all further work until a legacy `PPOTrainer.step(...)` API is proven
   available in every environment.

### Consequences
- The repository now has one baseline path for TRL interface compatibility and
  one separate path that makes decoded-SMILES chemistry rewards enter PPO
  updates explicitly.
- Smoke-test logs can now distinguish “trainer compatibility succeeded” from
  “decoded chemistry reward was called and used in an update”.
- The local PPO loop stays inside repository code, so no external environment
  patching is required.

### Status
Accepted

---

## [2026-04-18] Skip trainer-managed completion previews when experimental PPO has no usable eval dataloader

### Background
After the ChemLLM cache fix, the value-model `.score` adapter, and the
reward-model compatibility adapter were all in place, the PPO smoke test
finally entered the trainer loop and emitted PPO metrics. The next failure came
from `trl.experimental.ppo.PPOTrainer.generate_completions()`, which tried to
iterate `self.eval_dataloader` even though the smoke-test path had no usable
evaluation data source. That eventually surfaced as:

- `TypeError: object of type 'NoneType' has no len()`

### Decision
Keep the main PPO training loop intact, but add a local repository-side guard
in `scripts/train_ppo.py` for the trainer-managed completion-preview stage:

- add `--skip-generate-completions` as an explicit CLI escape hatch;
- add `--diagnose-reward-flow` as a smoke-test-friendly debug flag;
- detect unusable completion-preview loaders such as missing
  `eval_dataloader`, missing `dataset`, or `sampler.data_source is None`;
- replace `ppo_trainer.generate_completions()` with a no-op logger only when
  the skip flag is enabled or the evaluation loader is clearly unusable.

The HPC smoke-test script now always passes both diagnostic flags so it can
focus on initialization and the core PPO loop without failing in the preview
generation branch.

### Alternatives considered
1. Patch `trl.experimental` directly in site-packages to special-case missing
   eval loaders.
2. Fabricate a fake evaluation dataset just to satisfy completion previews.
3. Catch and suppress broad exceptions around `ppo_trainer.train()`.

### Consequences
- The smoke test can keep exercising PPO initialization and training steps even
  when the trainer's optional preview-generation path has no evaluation data.
- Main training errors are still surfaced normally because only
  `generate_completions()` is replaced, not the full trainer loop.
- Slurm logs now include explicit `[PPO_GENERATE_COMPLETIONS_SKIPPED]` markers
  so it is obvious when this guard was applied.

### Status
Accepted

---

## [2026-04-18] Separate chemistry reward logic from TRL experimental reward-model interface compatibility

### Background
After the local `.score` adapter fixed the PPO critic-side value-model crash,
the smoke test progressed further and then failed inside
`trl.experimental.utils.get_reward()` with:

- `AttributeError: 'ChemRewardModelWrapper' object has no attribute 'base_model_prefix'`

The repository's `ChemRewardModelWrapper` computes chemistry-aware rewards by
decoding generated text back into parent / fragment pairs and calling
`ChemRLRewarder`. That is not the same interface as the Hugging Face-style
reward model expected by this experimental TRL path, which assumes:

- `base_model_prefix`
- a forwardable LM backbone accessible through that prefix
- `score(hidden_states)`

### Decision
Keep `ChemRewardModelWrapper` as the repository's chemistry reward component,
but stop passing it directly to experimental PPO when the runtime expects a
hidden-state reward model.

Instead, `scripts/train_ppo.py` now adds
`ensure_reward_model_for_experimental_ppo(...)`, which:

- reuses an existing reward-side backbone if one is already exposed;
- otherwise builds a lightweight TRL-compatible reward adapter around a
  fallback LM backbone such as `value_model.pretrained_model`;
- adds `base_model_prefix` and a token-level `score` head for interface
  compatibility;
- logs explicitly that the fallback adapter is only for smoke-test /
  interface-validation purposes and is not equivalent to the repository's
  deletion-based chemistry reward objective.

### Alternatives considered
1. Patch TRL directly in site-packages so it can accept the chemistry wrapper.
2. Pretend the chemistry wrapper is a native hidden-state reward model by
   adding only one missing attribute at a time.
3. Remove the chemistry reward component entirely and silently replace it with
   a generic reward head.

### Consequences
- The smoke test can progress past TRL's stricter `reward_model` interface
  checks without mutating the external conda environment.
- Repository code now makes the mismatch between chemistry rewards and TRL's
  hidden-state reward-model contract explicit instead of hiding it behind
  brittle monkey patches.
- Future work can reconnect true chemistry rewards to trainer-managed PPO more
  cleanly because the current limitation is now documented in code and docs.

### Status
Accepted

---

## [2026-04-18] Attach a local `.score` adapter for TRL value-head critics in experimental PPO

### Background
The ChemLLM PPO smoke test moved past the earlier InternLM2 cache failure, but
then failed inside `trl.experimental.utils.get_reward()` when the trainer tried
to evaluate the critic with:

- `model.score(output.hidden_states[-1])`

Our repository-side `value_model` was still
`AutoModelForCausalLMWithValueHead`, which exposes `v_head` rather than a
top-level `.score` method. Patching TRL in site-packages was explicitly out of
scope for the VS Code plus Git plus HPC workflow.

### Decision
Add a repository-local compatibility helper in `scripts/train_ppo.py`:

- `ensure_score_head_for_experimental_ppo(model, name=...)`

The helper now:

- leaves models that already expose `.score` unchanged;
- searches the top-level object and common wrapper layers such as
  `pretrained_model`, `base_model`, and `model` for a reachable `v_head`;
- attaches `model.score(hidden_states)` dynamically when only `v_head` exists;
- logs which wrapper layer supplied the adapted value head.

The adapter is applied to `value_model` before `PPOTrainer` construction. The
policy and reference models remain untouched so generation behavior stays
isolated from this critic-side interface patch.

### Alternatives considered
1. Patch `trl.experimental` directly inside the conda environment.
2. Replace the value wrapper again with another custom critic type.
3. Fork the PPO rollout path away from the trainer-managed `get_reward()`
   utility.

### Consequences
- Experimental PPO can keep using the existing TRL value-head wrapper while
  satisfying the newer `.score(hidden_states)` critic contract.
- The compatibility layer stays local to repository code and is therefore easy
  to review, sync to HPC, and remove later if upstream APIs converge.
- Smoke-test logs now make it explicit whether the value model has both
  `v_head` and the adapted `.score` interface before training begins.

### Status
Accepted

---

## [2026-04-09] Rebuild repository from scratch with documentation-first workflow

### Background
The previous project evolved through multiple experimental iterations. It likely accumulated script coupling, outdated assumptions, and mixed objectives inherited from earlier versions. To prevent repeated confusion during reconstruction, the repository needs a stable written definition before code implementation begins.

### Decision
Rebuild the repository from an empty root using a documentation-first workflow. The initial authoritative files are:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

These files define the objective, engineering rules, roadmap, and future design log.

### Alternatives considered
1. Start coding immediately and add documentation later.
2. Recover the old repository structure first and refactor afterward.
3. Keep all design notes only in chat history.

### Consequences
- The repository gets a clear source of truth from the start.
- Codex can be guided by repository-local instructions instead of relying on conversational memory.
- Early progress is slower in appearance but more stable in direction.

### Status
Accepted

---

## [2026-04-16] Make ChemLLM cache patch tool classify guarded vs unguarded accesses

### Background
The first repository-side ChemLLM cache patch tool could add the helper and
patch the forward-path cache-length logic, but its multiline matching for
`prepare_inputs_for_generation()` was too brittle. As soon as the target block
contained nested indentation, comments, or slightly different formatting, the
prepare-path patch silently failed. The checker output also counted every
`past_key_values[0][0].shape[2]` occurrence as equally dangerous even after it
had been moved under the new guard.

### Decision
Rework `tools/check_or_patch_chemllm_cache.py` to use indentation-aware,
line-based patching and reporting:

- patch the forward cache-length block and the
  `prepare_inputs_for_generation()` block independently;
- insert `else: past_key_values = None` / `past_length = 0` for the prepare
  path;
- detect already-patched blocks and skip them cleanly;
- classify dangerous accesses as either `guarded` or `unguarded`, and treat
  `unguarded_count=0` as the real success criterion.

### Alternatives considered
1. Keep the old regexes and only tweak them slightly.
2. Require manual patching for the prepare path on HPC.
3. Keep counting all `shape[2]` accesses as equally dangerous.

### Consequences
- The repository-side helper can now patch both critical cache branches before
  HPC smoke testing.
- Patch results are easier to interpret because protected accesses are no
  longer reported as unresolved failures.
- HPC documentation can now point users to `unguarded_count=0` instead of
  incorrectly suggesting that all `shape[2]` accesses must disappear.

### Status
Accepted

---

## [2026-04-16] Add PPO runtime import-path introspection for ChemLLM cache debugging

### Background
The ChemLLM / InternLM2 PPO path hit a cache-related generation crash inside
`modeling_internlm2.py`, but local repository edits alone were not enough to
prove which dynamically cached file the Slurm job was actually importing at
runtime. In a VS Code plus Git plus HPC workflow, the repository copy, the
Hugging Face dynamic module cache, and the job's working directory can diverge.

### Decision
Add lightweight runtime introspection to `scripts/train_ppo.py` so every PPO
run logs:

- the wrapped policy / reference / value model module names;
- the resolved module source files;
- the resolved `prepare_inputs_for_generation` source files;
- key environment variables such as `PYTHONPATH`, `HF_HOME`,
  `TRANSFORMERS_CACHE`, and `HUGGINGFACE_HUB_CACHE`.

Also add a dedicated Slurm smoke-test entrypoint:

- `scripts/slurm/debug_check_chemllm_runtime_path.sh`

that reuses the normal HPC environment bootstrap, prints repository and Python
runtime information, and runs a tiny PPO smoke test through
`scripts/train_rl.py`.

### Alternatives considered
1. Keep reasoning about cache behavior from static local files only.
2. Patch more InternLM2 code blindly without first proving the runtime import
   path.
3. Ask users to manually add debug prints on the HPC side.

### Consequences
- Future Slurm logs can show exactly which `modeling_internlm2.py` was imported
  during PPO generation.
- Runtime path mismatches between repository code and Hugging Face dynamic cache
  are easier to detect before chasing deeper trainer bugs.
- The repository now has a reusable HPC-first smoke test for ChemLLM runtime
  path debugging.

### Status
Accepted

---

## [2026-04-15] Implement first PPO training path with residual-based reward wrapper

### Background
The repository already had a trained SFT adapter, a lightweight AIDS oracle,
and a chemistry layer for capped fragment validation and deletion. What was
still missing was a real PPO training path that could optimize ChemLLM outputs
toward the deletion-based counterfactual objective on HPC hardware.

### Decision
Add a concrete PPO training entrypoint in `scripts/train_ppo.py` together with a
unified reward wrapper in `src/rewards/reward_wrapper.py`.

The reward wrapper uses a three-stage early-stop flow:

- parseability and connectedness checks;
- parent-subgraph validation for capped fragments;
- residual-molecule scoring after deleting the fragment from the parent.

The semantic reward term is computed on the residual graph rather than on the
fragment alone, because the v3 objective is label flipping after deletion.

The PPO script loads:

- ChemLLM-7B-Chat in 4-bit mode;
- the SFT LoRA checkpoint as the initial policy;
- a frozen LoRA-backed reference model for KL control;
- prompts from either the raw HIV CSV or a JSONL prompt file.

The training loop logs reward statistics, structural pass rates, simple
collapse diagnostics, and representative generations into the run directory.

### Alternatives considered
1. Keep RL as a runtime-preparation placeholder only.
2. Score the fragment alone with the oracle instead of the residual molecule.
3. Hardcode one dataset schema and refuse either CSV or JSONL prompts.

### Consequences
- The repository now has a runnable PPO-stage backbone aligned with the
  counterfactual v3 objective.
- KL control remains anchored to the SFT adapter rather than drifting from the
  base model directly.
- PPO runs surface chemistry and collapse signals explicitly instead of hiding
  them inside trainer internals.

### Status
Accepted

---

## [2026-04-15] Switch PPO policy loading back to native causal LM for experimental TRL

### Background
After the first PPO entrypoint landed, newer `trl.experimental.ppo` builds
showed constructor-time incompatibilities with explicit value-head wrappers,
including failures around missing wrapper attributes such as
`base_model_prefix`. This indicated that the trainer now expects native causal
LM policies and prefers to manage policy/value wrapping internally.

### Decision
Update `scripts/train_ppo.py` so that PPO loads native
`AutoModelForCausalLM`-style PEFT models for both the trainable policy and the
frozen reference policy, without constructing
`AutoModelForCausalLMWithValueHead`.

The PPO trainer initialization is now version-adaptive in three ways:

- `PPOConfig` kwargs are filtered against the runtime signature;
- trainer kwargs map across `args/config`, `model/policy`, and
  `ref_model/ref_policy` variants;
- external `value_model` is omitted or forced to `None` so TRL can own value
  wrapping internally.

The script also provides a lightweight reward adapter that exposes
`ChemRLRewarder` as either `reward_model` or `reward_funcs` when those newer
experimental hooks are present.

### Alternatives considered
1. Keep the explicit value-head wrapper and patch missing attributes one by one.
2. Pin the repository to one older TRL version instead of adapting the code.
3. Move the whole PPO path back to a handwritten optimizer loop.

### Consequences
- The PPO script is better aligned with current experimental TRL architecture.
- Fewer wrapper-specific attribute mismatches should appear when the trainer is
  upgraded.
- The script still keeps the repository's deletion-based counterfactual reward
  logic outside trainer internals.

### Status
Accepted

---

## [2026-04-15] Use explicit value and reward models for experimental PPOTrainer

### Background
Follow-up PPO integration work revealed another compatibility shift in newer
`trl.experimental.ppo.PPOTrainer` builds: the trainer now expects a real
transformers-style `value_model` object and also routes internal scoring
through a PyTorch `reward_model(input_ids=..., attention_mask=..., ...)`
interface.

### Decision
Keep the policy and reference networks as native causal LMs, but explicitly add:

- a 4-bit `AutoModelForSequenceClassification` value model with `num_labels=1`;
- a torch `reward_model` wrapper that decodes generated sequences back to text,
  reconstructs the parent / fragment pair, and calls `ChemRLRewarder`.

The trainer initialization in `scripts/train_ppo.py` now wires:

- `model`: native causal LM policy with the SFT LoRA adapter;
- `ref_model`: frozen native causal LM reference policy;
- `value_model`: explicit scalar sequence-classification model;
- `reward_model`: deletion-based chemistry reward wrapper.

### Alternatives considered
1. Continue passing `None` for the value model and rely on implicit trainer behavior.
2. Keep reward scoring entirely outside the trainer and ignore new internal hooks.
3. Reintroduce custom non-transformers wrapper classes around the value head.

### Consequences
- PPO initialization is better aligned with stricter experimental TRL releases.
- Internal trainer scoring can now call a PyTorch-compatible reward interface
  without losing the repository's residual-graph counterfactual objective.
- The value network contract is explicit instead of version-dependent.

### Status
Accepted

---

## [2026-04-15] Fall back to TRL value-head wrapper for InternLM2 PPO value model

### Background
The explicit sequence-classification critic introduced for experimental PPO
alignment turned out to be incompatible with the ChemLLM / InternLM2 base
checkpoint in environments where the InternLM2 config is not registered under
Hugging Face's `AutoModelForSequenceClassification` auto-mapping.

### Decision
Replace the PPO `value_model` in `scripts/train_ppo.py` with
`trl.AutoModelForCausalLMWithValueHead` loaded on top of the same quantized
ChemLLM base weights, and monkey-patch:

- `value_model.base_model_prefix = "pretrained_model"`

so that stricter `trl.experimental.ppo.PPOTrainer` builds can still treat the
wrapper like a native model during critic setup.

Only the wrapper's value head remains trainable; the wrapped causal LM backbone
stays frozen.

### Alternatives considered
1. Keep using `AutoModelForSequenceClassification` and require a newer
   InternLM2 registration patch from transformers.
2. Drop experimental PPOTrainer and fully hand-roll critic/value optimization.
3. Try to emulate a sequence-classification head with another custom wrapper.

### Consequences
- PPO critic initialization no longer depends on InternLM2 being registered in
  the sequence-classification auto-model registry.
- The project keeps the reward path aligned to the deletion-based
  counterfactual objective while staying closer to TRL's expected interfaces.
- The monkey patch is intentionally local to the PPO script and should be
  revisited if upstream TRL or transformers support improves.

### Status
Accepted

---

## [2026-04-15] Let experimental PPOTrainer own rollout and optimization

### Background
Once the experimental PPO trainer, explicit value model, and reward model were
all wired successfully, the remaining failures came from the old handwritten
training loop that still tried to call trainer-side generation and manual PPO
updates directly.

### Decision
Remove the legacy step-by-step PPO loop from `scripts/train_ppo.py` and switch
the entrypoint to the trainer-managed flow:

- initialize policy, reference, value, and reward models;
- construct `PPOTrainer`;
- call `ppo_trainer.train()`;
- save the final checkpoint via the trainer's own save path.

The PPO config assembly now also forwards `max_steps` and generation-related
kwargs when the runtime `PPOConfig` signature supports them.

### Alternatives considered
1. Keep maintaining a hybrid script that mixes experimental trainer internals
   with a manual rollout loop.
2. Revert fully to a classic step-based TRL API.
3. Move rollout back outside the trainer and bypass `reward_model`.

### Consequences
- The PPO entrypoint now matches the ownership model of newer
  `trl.experimental.ppo` releases more closely.
- Fewer incompatibilities should appear around missing `generate()` or `step()`
  methods on experimental trainer objects.
- Reward evaluation stays aligned with the repository's residual-graph
  counterfactual objective, but execution control is delegated to TRL.

### Status
Accepted

---

## [2026-04-15] Run PPO WandB logging in offline mode for HPC nodes

### Background
The PPO training entrypoint is designed for HPC execution, but compute nodes in
the target environment do not have outbound internet access. Direct online
Weights & Biases logging would therefore stall or fail during trainer startup.

### Decision
Configure `scripts/train_ppo.py` to force WandB offline mode via environment
variables:

- `WANDB_MODE=offline`
- `WANDB_SILENT=true`

At the same time, keep the trainer-side reporting target aligned to WandB when
supported by the runtime PPO config, so metrics are still written locally into
the standard `wandb/` directory for later sync.

The PPO config builder now forwards:

- `report_to="wandb"` when supported;
- `log_with="wandb"` for older compatible signatures;
- `run_name="ppo_aids_rl_v1"` as the semantic experiment label.

### Alternatives considered
1. Disable WandB entirely on HPC and rely only on stdout or custom JSON logs.
2. Require a separate shell wrapper to export WandB offline variables.
3. Keep online WandB enabled and tolerate repeated timeout failures.

### Consequences
- PPO runs on air-gapped or internal-only HPC nodes no longer depend on live
  WandB connectivity.
- Local WandB artifacts remain available for later `wandb sync`.
- Experiment naming is more stable across local logs and offline WandB runs.

### Status
Accepted

---

## [2026-04-16] Force PPO datasets to emit tensors before entering experimental trainer

### Background
After the PPO entrypoint successfully crossed model and trainer initialization,
the first runtime failure inside `trl.experimental.ppo.PPOTrainer.train()`
occurred when the trainer tried to move `data["input_ids"]` onto the device.
The immediate cause was that the Hugging Face dataset / collator path was still
yielding Python lists instead of PyTorch tensors.

### Decision
Update `scripts/train_ppo.py` so that the PPO training dataset explicitly calls:

- `dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)`

after tokenization, and replace the custom collator with a tensor-aware version
that:

- pads `input_ids` through `tokenizer.pad(..., return_tensors="pt")`;
- emits `input_ids` and `attention_mask` as tensors;
- preserves text metadata fields such as `query` and `parent_smiles` as Python
  lists for downstream reward reconstruction.

The reward wrapper's decoding path also now detaches tensor inputs to CPU
before `batch_decode`, while keeping the returned reward tensor on the same
device as `input_ids`.

### Alternatives considered
1. Rely only on `set_format("torch")` and keep the old collator unchanged.
2. Drop the custom collator entirely and hope the trainer default handles mixed
   text-and-tensor batches correctly.
3. Convert data inside the trainer loop instead of fixing the dataset contract.

### Consequences
- Experimental PPOTrainer can now consume `input_ids` with a valid `.to(device)`
  path.
- The batch contract is more explicit and robust against future TRL internal
  assumptions.
- Reward evaluation remains device-safe even when the trainer feeds tensor
  batches directly from GPU-backed training steps.

### Status
Accepted

---

## [2026-04-16] Use DataCollatorWithPadding-backed PPO collator for tensor-safe batches

### Background
After forcing the PPO dataset into torch format, the next trainer failure still
occurred at batch materialization time because the custom collator path was not
reliably returning a dictionary of tensors. In practice, the trainer ended up
receiving `None` instead of a batch payload.

### Decision
Replace the fragile handcrafted padding path in `scripts/train_ppo.py` with a
wrapper around Hugging Face's standard `DataCollatorWithPadding`, configured
with `return_tensors="pt"`.

The PPO collator now:

- delegates token padding to `DataCollatorWithPadding`;
- guarantees a returned batch dictionary;
- validates that `input_ids` exists and is a tensor;
- preserves non-model metadata fields such as `query`, `parent_smiles`, and
  `original_label` as Python lists for downstream reward reconstruction.

### Alternatives considered
1. Keep the custom collator and only add a missing `return batch`.
2. Drop metadata preservation and use a raw Hugging Face collator directly.
3. Remove the collator override and rely fully on trainer defaults.

### Consequences
- PPO batches now have a much stronger contract before they enter
  `trl.experimental.ppo.PPOTrainer`.
- Standard padding behavior is delegated to a well-tested transformers utility.
- Reward logic can still reconstruct prompt context without coupling that logic
  to the token padding implementation.

### Status
Accepted

---

## [2026-04-16] Monkey-patch experimental PPO wrapper to expose gradient checkpointing hooks

### Background
After the PPO data path was repaired, the next runtime failure came from
`trl.experimental.ppo` itself: the trainer's internal `PolicyAndValueWrapper`
was missing `gradient_checkpointing_disable` and
`gradient_checkpointing_enable`, even though the wrapped policy model already
implemented them.

### Decision
Patch the trainer-managed wrapper immediately after PPO trainer construction in
`scripts/train_ppo.py`:

- if `ppo_trainer.model.policy_model.gradient_checkpointing_disable` exists,
  bind it onto `ppo_trainer.model.gradient_checkpointing_disable`;
- do the same for `gradient_checkpointing_enable`.

In parallel, the PPO config builder now explicitly forwards
`gradient_checkpointing=False` when the runtime config signature supports that
field, so trainer-side generation does not try to rely on checkpoint toggling
more than necessary.

### Alternatives considered
1. Wait for an upstream TRL patch and leave the local training script broken.
2. Disable wrapper-managed generation entirely and fall back to a handwritten
   PPO rollout loop again.
3. Monkey-patch TRL package files directly in the environment.

### Consequences
- The repository no longer depends on an immediate upstream TRL fix for this
  wrapper attribute bug.
- The workaround stays local to the PPO entrypoint instead of mutating
  site-packages.
- Generation-time checkpoint toggling is less likely to derail short HPC PPO
  smoke tests.

### Status
Accepted

---

## [2026-04-16] Escalate PPO wrapper gradient-checkpointing fix from instance patch to class patch

### Background
The first local workaround for the experimental TRL wrapper bug patched the
trainer instance after construction. That turned out to be insufficient because
the runtime path could still recreate or access a wrapper object that had not
received the instance-local method bindings.

### Decision
Move the gradient-checkpointing workaround into the TRL import phase inside
`scripts/train_ppo.py` by patching the wrapper class itself:

- import `trl.experimental.ppo.ppo_trainer` when available;
- if `PolicyAndValueWrapper` exists, inject no-op
  `gradient_checkpointing_disable` and `gradient_checkpointing_enable` methods
  onto the class when they are missing.

This class-level patch supersedes the earlier instance-level workaround. The
config-side guard `gradient_checkpointing=False` remains in place as a second
line of defense.

### Alternatives considered
1. Keep stacking more instance-local patches after trainer construction.
2. Patch TRL directly inside site-packages on the target machine.
3. Revert to non-experimental PPO code paths entirely.

### Consequences
- Any `PolicyAndValueWrapper` instantiated after the patch inherits the missing
  methods automatically.
- The PPO script becomes less sensitive to internal wrapper recreation inside
  experimental TRL.
- The workaround remains local to the repository instead of mutating external
  package files.

### Status
Accepted

---

## [2026-04-16] Disable InternLM2 KV cache across PPO trainer wrappers before rollout

### Background
After the PPO entrypoint finally entered the trainer-managed batch generation
loop, ChemLLM's InternLM2 generation stack still failed inside
`modeling_internlm2.py` when the runtime attempted to consume incompatible
`past_key_values`. This failure occurred during PPO rollout rather than during
model initialization.

### Decision
Add a trainer-side runtime patch in `scripts/train_ppo.py` that recursively
walks the trainer model and common wrapper attributes such as:

- `policy_model`
- `pretrained_model`
- `model`
- `base_model`

For each discovered layer, the patch:

- forces `config.use_cache = False` when available;
- forces `generation_config.use_cache = False` when available;
- synchronizes `pad_token_id` and `eos_token_id` with the tokenizer.

This patch is applied immediately after PPO trainer construction and before
`ppo_trainer.train()`.

### Alternatives considered
1. Rely only on the earlier model-loading-time `use_cache=False` settings.
2. Try to sanitize or rewrite `past_key_values` formats for InternLM2.
3. Revert back to a completely manual rollout loop outside experimental TRL.

### Consequences
- PPO rollout is less likely to trigger InternLM2 KV-cache compatibility bugs
  through deeply wrapped trainer models.
- Token id settings stay aligned between tokenizer and generation config at the
  exact point where trainer-managed generation begins.
- The workaround remains local to the repository and does not require patching
  upstream model files.

### Status
Accepted

---

## [2026-04-16] Replace static InternLM2 cache patch with generate-method hijacking

### Background
Even after synchronizing `use_cache=False` into wrapped model configs, the PPO
runtime could still reintroduce cache-related generation kwargs dynamically
through experimental TRL batch generation. As a result, InternLM2 continued to
receive incompatible cache inputs during rollout.

### Decision
Change the PPO runtime workaround in `scripts/train_ppo.py` from static
generation-config edits to method hijacking:

- keep tokenizer-aligned `pad_token_id` / `eos_token_id` synchronization;
- wrap `generate()` on the trainer model, policy model, and base model when
  available;
- force `kwargs["use_cache"] = False` on every generation call;
- drop `past_key_values` from generation kwargs before delegating to the
  original method.

### Alternatives considered
1. Keep stacking more static `config.use_cache=False` assignments.
2. Patch InternLM2 model code directly.
3. Revert back to a custom manual rollout loop outside experimental TRL.

### Consequences
- Cache disabling now applies at the exact generation call boundary where TRL
  injects kwargs.
- The workaround is less sensitive to internal trainer overrides of static
  generation config.
- The script keeps a narrow, local compatibility layer without mutating
  external package files.

### Status
Accepted

---

## [2026-04-16] Override InternLM2 prepare_inputs_for_generation at the deepest wrapped class

### Background
The generate-method hijack still proved insufficient once TRL, PEFT, and the
current transformers cache stack interacted through multiple wrapper layers.
InternLM2 continued to fail inside its own `prepare_inputs_for_generation`
implementation because the expected tuple-style cache structure no longer
matched what the runtime was trying to pass.

### Decision
Replace the generate-method interception in `scripts/train_ppo.py` with a
deeper class-level patch that:

- unwraps the trainer model through common wrapper attributes such as
  `policy_model`, `base_model`, and `model`;
- identifies the actual underlying causal LM class;
- overrides `prepare_inputs_for_generation` on that class;
- forces `past_key_values=None` and `use_cache=False` on every call.

Tokenizer-aligned `pad_token_id` / `eos_token_id` synchronization remains in
place before the patch is applied.

### Alternatives considered
1. Keep layering more `generate()` wrappers on top of TRL and PEFT.
2. Patch InternLM2 source files directly in the runtime environment.
3. Abandon trainer-managed generation and return to a manual PPO loop.

### Consequences
- The cache-breaking branch is cut off at the exact InternLM2 method where the
  incompatibility manifests.
- The fix is more resilient to outer wrapper churn because it targets the
  effective model class rather than only the outermost object.
- The workaround stays local to repository code and can be removed later if
  upstream InternLM2 / transformers compatibility improves.

### Status
Accepted

---

## [2026-04-09] Treat counterfactual fragment generation as the sole primary objective

### Background
Earlier project stages and surrounding discussions may involve concept extraction, class-indicative subgraphs, or rationale-like objectives. Those formulations are related but not identical to the present v3 goal.

### Decision
Define the project strictly as counterfactual fragment generation: the generated fragment should be useful because deleting it is likely to flip the label.

### Alternatives considered
1. Optimize for fragment-only label predictiveness.
2. Optimize for class-shared concept extraction.
3. Optimize for rationale sufficiency rather than deletion-based effect.

### Consequences
- Reward design must include deletion-based semantics.
- Evaluation must include flip-related metrics.
- Old code paths aligned to non-counterfactual targets should be treated as legacy behavior.

### Status
Accepted

---

## [2026-04-09] Use modular repository structure rather than monolithic training script

### Background
The project involves chemistry checks, prompt generation, SFT, RL, evaluation, and logging. A single large script would make the system difficult to debug and easy to break.

### Decision
Adopt a modular structure with separate folders for data, models, rewards, training, evaluation, chemistry utilities, and general utilities.

### Alternatives considered
1. Keep everything in one file for convenience.
2. Split only by training stage.
3. Organize by experiments rather than functionality.

### Consequences
- Initial setup requires more files.
- The system becomes easier to test, replace, and extend.
- Reward and chemistry behavior can be validated independently.

### Status
Accepted

---

## [2026-04-09] Prioritize chemistry and reward layers before large-scale training code

### Background
The most brittle parts of this project are likely to be chemistry correctness and reward semantics. If these layers are unstable, training results will be misleading.

### Decision
Implement chemistry utilities and reward computation before building full SFT/RL training pipelines.

### Alternatives considered
1. Start with full training script and fill utility functions later.
2. Start with model wrapper only.
3. Start with evaluation only.

### Consequences
- Early iteration focuses on correctness rather than speed.
- The train loop can remain thinner and easier to debug.
- Reward failures and chemistry mismatches can be unit-tested.

### Status
Accepted

---

## [2026-04-09] Bootstrap the repository with typed module interfaces before any training implementation

### Background
After the documentation-first reset, the repository still had no source tree, no config folders, and no stable interface boundaries. If training code were introduced at that point, chemistry logic, reward semantics, and CLI behavior would likely become coupled again.

### Decision
Create the repository skeleton first and define minimum typed interfaces for:

- `src/data/`
- `src/chem/`
- `src/rewards/`
- `src/models/`
- `src/train/`
- `src/eval/`
- `src/utils/`

Also add thin placeholder CLI files under `scripts/` and minimal smoke tests for prompt and reward contracts.

The bootstrap intentionally implements only safe low-level helpers such as JSONL IO, prompt construction, reward-term aggregation, and text-level collapse diagnostics. RDKit-backed chemistry logic and all training loops remain deferred.

### Alternatives considered
1. Start with one large training script and refactor later.
2. Implement RDKit parsing and training code immediately without freezing interfaces first.
3. Leave the directory layout undocumented and decide file boundaries ad hoc.

### Consequences
- Future chemistry, reward, inference, and training work now has a stable import surface.
- The repository stays aligned to the deletion-based counterfactual objective instead of drifting toward concept extraction.
- Scripts remain thin by design, because domain logic now has clear module ownership.
- The next implementation phase can focus on chemistry correctness rather than file organization.

### Status
Accepted

---

## [2026-04-09] Implement chemistry utilities with RDKit-first behavior and explicit failure types

### Background
The v3 objective depends on structural correctness before any reward or training logic can be trusted. The repository therefore needs a chemistry layer that can safely parse SMILES, check connectedness and parent-substructure relations, and attempt fragment deletion without forcing train code to interpret raw RDKit exceptions.

### Decision
Implement the first chemistry utility layer in:

- `src/chem/smiles_utils.py`
- `src/chem/substructure.py`
- `src/chem/deletion.py`
- `src/chem/validation.py`

The layer is RDKit-first when RDKit is available, but it must also degrade safely when RDKit is missing by returning structured failure types instead of crashing. The shared result dataclasses now carry normalized failure categories, and `validate_fragment_candidate(...)` aggregates these into one interface for downstream reward and evaluation code.

### Alternatives considered
1. Raise raw exceptions from RDKit and let train or eval scripts handle them.
2. Hard-require RDKit at import time and fail the whole repository if it is absent.
3. Delay chemistry implementation until reward or inference code exists.

### Consequences
- Reward and evaluation modules can consume chemistry results without duplicating error handling.
- The repository remains usable in environments where RDKit has not been installed yet.
- Deletion behavior is deterministic at the interface level, with the first matched connected fragment removed and failures surfaced explicitly.

### Status
Accepted

---

## [2026-04-10] Add a config-driven local and HPC runtime layer without hardcoded paths

### Background
The rebuilt repository now has modular source folders and chemistry utilities, but it still needs a stable way to run from a local workstation and from an HPC cluster. Without a shared runtime layer, path handling, log directories, and environment-specific behavior would drift into ad hoc script logic.

### Decision
Add a runtime adaptation layer based on repository-relative config files and thin CLI entrypoints.

This layer includes:

- base, local, hpc, sft, rl, and eval config files under `configs/`
- environment detection and config merging in `src/utils/env.py`
- repository-relative path resolution in `src/utils/paths.py`
- file-backed logging helpers in `src/utils/logging_utils.py`
- run entrypoints in `scripts/run_sft.py`, `scripts/run_rl.py`, `scripts/run_eval.py`, and `scripts/run_infer.py`
- Slurm templates for single-node single-GPU execution under `scripts/slurm/`

Model and tokenizer handling must support local filesystem paths and avoid silent remote downloads by default.

### Alternatives considered
1. Hardcode local and HPC paths directly inside the stage scripts.
2. Depend on an external YAML package before adding any runtime config support.
3. Jump directly to distributed training setup before the single-node path is stable.

### Consequences
- Local development and HPC submission now share one config and path resolution story.
- The codebase remains portable because configs stay relative and resolved at runtime.
- The repository can prepare deterministic run manifests before full SFT, RL, and evaluation logic is implemented.
- Distributed training remains intentionally out of scope for this phase.

### Status
Accepted

---

## [2026-04-10] Implement a minimal single-sample inference loop with heuristic fragment generation

### Background
The repository already has chemistry utilities and runtime adaptation, but it still needs a runnable end-to-end path from one parent SMILES to one fragment candidate. This is necessary to validate the IO contract before wiring any full SFT or RL training logic.

### Decision
Implement a minimal inference closure centered on `scripts/run_infer.py` and `src/eval/inference.py`.

This path:

- accepts one parent SMILES from CLI or config
- produces one heuristic fragment candidate without using a trained policy
- runs chemistry utilities for parseability, connectedness, and parent-substructure checks
- prints a structured JSON result
- stays independent of the full training stack

The heuristic prefers small connected parent substructures when chemistry backends are available, and falls back to the parent SMILES when no smaller valid candidate can be established.

### Alternatives considered
1. Keep `run_infer.py` as config-only runtime preparation.
2. Wait for a trained model before implementing any inference path.
3. Print only free-form text instead of a machine-readable result.

### Consequences
- The project now has a minimal runnable contract for one-sample inference.
- Chemistry utilities can be exercised from a real CLI path without invoking training code.
- The current fragment proposal is intentionally heuristic and not yet counterfactual-optimal, but it keeps the repository moving toward the v3 objective with a reviewable baseline.

### Status
Accepted

---

## [2026-04-14] Add a lightweight RandomForest Oracle and residual-based PPO reward wrapper

### Background
The repository has completed the SFT stage and is moving toward PPO. At that point the project needs a reward path that is both chemically constrained and fast enough to run inside RL rollouts. The existing codebase already has RDKit-backed capped-subgraph validation and deletion, but it still lacked:

- a lightweight local Oracle for fast activity scoring;
- a PPO-facing reward wrapper that combines validity, subgraph, and counterfactual terms;
- a clear decision on whether the semantic reward should score the fragment itself or the residual molecule after deletion.

### Decision
Add:

- `scripts/train_aids_oracle.py` to train a RandomForest classifier on Morgan fingerprints from `data/aids_dataset.csv`;
- `src/rewards/chem_rules.py` as a small structural reward engine for validity and parent-subgraph checks;
- `src/rewards/reward_calculator.py` as the PPO-facing reward wrapper that loads the Oracle bundle and computes `R_valid + R_subgraph + R_counterfactual`.

The counterfactual term is explicitly defined on the residual molecule `x \ g` after deleting the generated fragment, not on the fragment alone. Dummy atoms (`*`) are cleaned through RDKit graph editing before Morgan fingerprint extraction.

### Alternatives considered
1. Score the generated fragment by itself and treat that as the counterfactual term.
2. Put Oracle logic directly inside the PPO training script.
3. Save a custom Python class inside the pickle bundle instead of a plain dictionary.

### Consequences
- PPO reward computation stays aligned with the v3 deletion-based counterfactual objective.
- Oracle scoring remains fast enough for rollout-time use because it relies on RandomForest plus Morgan fingerprints.
- Structural chemistry checks stay modular and reusable instead of being buried in the train loop.
- The saved Oracle bundle is easier to reload across scripts because it stores plain metadata alongside the fitted sklearn model.

### Status
Accepted

---

## [2026-04-13] Add base-model metric tooling and presentation visualization for the SFT stage

### Background
The SFT stage now has stable headline numbers for validity, capping behavior, and token accuracy, but the project still lacked two practical tools: a deterministic way to compute baseline metrics from JSONL inference logs, and a reusable plotting pipeline for lab-meeting summaries.

### Decision
Add a small evaluation utility layer for base-model JSONL logs and extend the SFT visualization module with presentation-ready outputs.

This change introduces:

- `src/eval/base_metrics.py` for RDKit-backed capping/validity statistics
- `src/eval/base_inference.py` for deterministic base-model batch inference over `sft_val.jsonl`
- `scripts/eval_base_metrics.py` as a thin CLI entrypoint for base-model log evaluation
- `scripts/run_infer_base.py` as a thin CLI entrypoint that saves base-model predictions to JSONL
- an upgraded `src/eval/sft_visualization.py` that renders:
  - base-vs-SFT comparison bars
  - a dual-axis simulated training dynamics figure
  - a high-resolution RDKit rendering of a capped fragment
- the updated `scripts/visualize_sft_summary.py` CLI for parameterized figure generation

### Alternatives considered
1. Compute baseline metrics manually in notebooks.
2. Keep visualization code in a single ad hoc script under `scripts/`.
3. Hardcode baseline numbers directly into plotting code without a reusable evaluation helper.

### Consequences
- The repository now has a reproducible path from base-model inference logs to group-meeting figures.
- Evaluation logic remains modular under `src/eval/`, while CLI scripts stay thin and HPC-friendly.
- Presentation plots can be regenerated with different baseline numbers without editing source code.
- The training dynamics figure now uses discrete epoch-level line charts instead of dense smooth curves, which makes the plot easier to explain during presentations.

### Status
Accepted

---

## [2026-04-11] Add balanced capped-fragment SFT data preparation for the HIV dataset

### Background
The rebuilt repository already has RDKit-backed capped-subgraph utilities and thin runtime entrypoints, but it still lacked a concrete path for constructing supervised fine-tuning data from the local HIV CSV. Because the HIV benchmark is heavily label-imbalanced, a naive sample would underexpose positive molecules during SFT.

### Decision
Add a new balanced SFT data-preparation path centered on `scripts/prepare_sft_data.py` and `src/data/sft_preparation.py`.

This path:

- loads `data/raw/AIDS/HIV.csv` with pandas
- filters out invalid parent SMILES using the existing chemistry layer
- keeps all valid positive molecules and fills the remaining target size with sampled negatives
- constructs capped fragment targets by cutting one or two acyclic single bonds with RDKit dummy-atom capping
- validates generated fragments with the shared capped-subgraph checks
- writes minimal `instruction` / `output` JSONL files for ChemLLM SFT

### Alternatives considered
1. Reuse the placeholder `scripts/prepare_data.py` without adding reusable source modules.
2. Sample the HIV dataset according to its natural class distribution.
3. Generate weak targets without validating capped-subgraph correctness against the parent molecule.

### Consequences
- The repository now has an end-to-end path for building balanced SFT supervision data aligned with the capped-fragment objective.
- Positive molecules are intentionally overrepresented relative to the raw dataset so ChemLLM sees enough active-molecule structure during SFT.
- Fragment generation can still fail for some molecules, so the script explicitly reports success rate and observed label ratios after construction.

### Status
Accepted

---

## [2026-04-10] Treat dummy atoms as attachment-point caps in the chemistry layer

### Background
The project now needs chemically valid graph cutting. Plain atom deletion breaks
valence and makes the fragment contract ambiguous whenever the generated
subgraph has open attachment points to the rest of the parent molecule.

### Decision
Adopt dummy atoms (`*`) as the canonical capping representation inside
`src/chem/`.

This means:

- capped fragment SMILES such as `*c1ccccc1` are parsed and sanitized through a
  dummy-aware RDKit path;
- dummy atoms are treated as attachment-point queries when checking whether a
  fragment is a valid parent subgraph;
- capped deletion uses RDKit core replacement semantics so the remainder graph
  is also capped with dummy atoms instead of leaving broken chemistry.

### Alternatives considered
1. Keep using uncapped raw atom deletion and repair broken valence later.
2. Represent cut points with text markers outside SMILES instead of dummy atoms.
3. Treat dummy atoms as ordinary fragment atoms that should also be deleted from
   the parent match.

### Consequences
- The chemistry layer now distinguishes between real fragment atoms and dummy
  attachment points.
- Validation and deletion stay aligned with the v3 counterfactual objective,
  because both fragment extraction and residual construction use the same capped
  semantics.
- Future model outputs can use capped fragment SMILES directly without adding an
  extra post-processing representation.

### Status
Accepted

---

## [2026-04-10] Bootstrap local-only ChemLLM inference with dataset-backed sampling

### Background
The chemistry layer can now validate capped fragment SMILES, but the repository
still needs a real local-model inference path before SFT or RL code is added.
At the same time, the user needs a safe way to confirm that the local AIDS/HIV
CSV file and local ChemLLM checkpoint are usable without any accidental network
requests.

### Decision
Add a local-only ChemLLM inference bootstrap consisting of:

- `scripts/test_assets.py` for local dataset and model asset validation;
- a ChemLLM-specific prompt builder with hard-coded capped-fragment few-shot
  examples;
- a lightweight Hugging Face `ChemLLMGenerator` that always loads with
  `local_files_only=True`;
- dataset-backed `run_infer.py` behavior that samples one real molecule from the
  local AIDS/HIV CSV file when no explicit SMILES is provided.

If the local transformers stack is unavailable at runtime, the CLI is allowed to
fall back to the existing heuristic inference path, but it must surface the
model-side failure explicitly in the structured result.

### Alternatives considered
1. Wait for training code before integrating any real LLM backend.
2. Hardcode one demo SMILES instead of sampling a real local dataset row.
3. Allow `from_pretrained` to use default remote resolution behavior.

### Consequences
- The project now has a true local model inference integration point without
  depending on vLLM or distributed serving.
- Asset validation becomes safer because the repository refuses to reach out to
  Hugging Face when local files are incomplete.
- The inference CLI stays usable in lightweight dev environments because it can
  degrade to heuristic generation while still reporting why the model path did
  not run.

### Status
Accepted

---

## [2026-04-19] Make decoded chemistry PPO use a deletion-based counterfactual teacher oracle

### Background
The decoded chemistry PPO smoke test already proved that a fragment-level
teacher score on `core_fragment_smiles` could be computed, but the semantic
term that actually entered `total` still came from a fixed
`counterfactual_sem=-5.0` missing penalty whenever the deletion-based branch
was unavailable. That meant the PPO loop still was not aligned tightly enough
with the v3 counterfactual objective.

### Decision
Introduce an explicit deletion-based counterfactual teacher scorer that:

- deletes exactly one matched instance of the core fragment from the parent
  molecule;
- scores both the original parent and the residual parent with the teacher
  classifier;
- computes `cf_drop = p_before - p_after` and adds a configurable flip bonus
  when the residual prediction no longer matches the original label;
- uses this deletion-based `counterfactual_sem` as the default semantic term
  that enters the decoded PPO reward and total score.

The earlier fragment-level teacher score is retained only as an auxiliary
diagnostic field (`fragment_teacher_sem`) so we can still inspect whether the
generated fragment itself looks label-associated.

### Alternatives considered
1. Keep using the fragment-level teacher score as the main semantic reward.
2. Continue using the old residual-oracle fallback plus a fixed missing penalty.
3. Collapse fragment-level and counterfactual teacher semantics into one field.

### Consequences
- The decoded PPO path is now explicitly aligned with the repository's
  counterfactual deletion objective instead of a fragment-only semantic proxy.
- Logs can distinguish fragment-level diagnostics from the real
  counterfactual reward that enters PPO.
- Deletion failures, unavailable teacher backends, and disabled counterfactual
  teacher scoring are surfaced explicitly instead of being hidden behind an
  unexplained `-5.0`.

### Status
Accepted
## [2026-07-15] Use exact MolCLR node Wasserstein as the primary learned CCRCov distance

### Background
Graph-level pooled MolCLR distance can hide local correspondence, while the
existing Node-FGW line mixes learned node features with an explicit
shortest-path structure term. The final comparison needs a primary node-feature
transport line and a separately named structure-aware ablation.

### Decision
Use exact uniform-mass `ot.emd2` over MolCLR node cosine costs as
`MolCLR-Node-Wasserstein`. Keep Node-FGW (`lambda=0.5`) as an ablation. Share a
structure-independent v2 node embedding cache, but use independent symmetric
pair-cache namespaces. Calibrate thresholds from Ours only and use the resulting
absolute thresholds unchanged for every final baseline. Candidate selection is
external; the evaluator never changes Top20 order and does not compute
redundancy.

### Consequences
- WNode does not call shortest-path, GW/FGW, Sinkhorn, or networkx GED.
- Partial output now has a fingerprinted completed-pair resume contract.
- GCF-style reporting continues to aggregate match instances before Top-K
  prefix evaluation and can explicitly plot finite strict-recourse conditional
  median cost.
- Baseline training and candidate generation remain unchanged.

### Status
Accepted

---

## [2026-07-16] Make FGW presentation figures read-only and use the reported conditional-cost field

### Background
FGW final evaluation outputs already contain the Figure 3 prefix curve and
the dense Figure 4 threshold curve. Recomputing them merely to create paper
figures risks changing an evaluation artifact and is not permitted on an HPC
login node. Earlier ad hoc plotting also failed to recognize the actual
`conditional_median_cost` column.

### Decision
Use `scripts/plot_fgw_sota_figures.py` as a read-only post-processing tool.
It prioritizes `conditional_median_cost` (with documented compatibility
aliases), draws Figure 3 at the fixed q30 threshold, and emits separate
`K=1..10` and `K=1..20` figures. Figure 4 is read only from a dense `K=20`
curve; its normalized low-cost AUC always integrates over `[0, q30]`.

Submit plotting through `scripts/slurm/plot_fgw_sota_figures_gpu.sh`, which
uses the confirmed `A800` and `gpu:a800:1` resource combination from the
successful CLEAR Slurm template. No account, qos, or constraint is invented
because none is present in the verified project allocation pattern.

### Consequences
- The figure layer cannot alter strict-flip semantics, candidates, distances,
  or evaluator output.
- The displayed cost is explicitly described as the unified evaluator's
  conditional cost, not as original-paper unconditional GCFExplainer cost.
- A presentation audit permits only the narrowly scoped low-cost,
  compact-budget SOTA statement when all three configured checks pass.

### Status
Accepted

---
## Decision: GCF-style reports separate table and prefix thresholds

**Date:** 2026-07-16

GCF-style recourse post-processing now treats the Table 2 threshold and the
Figure 3 prefix threshold as separate report parameters. Both continue to fall
back to `--theta-star` for command compatibility, while `--table2-theta` and
`--figure3-theta` make the intended protocol explicit.

Figure 3 cost statistics use a fixed one-to-one mapping. In particular,
`conditional_median` means `conditional_median_cost`; it is not an alias for a
theta-covered cost. Full-range cost axes are the default, WNode artifacts use a
`wnode` filename slug, and historical unprefixed or `fgw` aliases are written
only when `--write-legacy-aliases` is requested. The compact Table 2 reports
overall conditional median cost, while theta-covered cost remains in a
separately named audit table.
## [2026-07-22] Use one curated SMILES benchmark for Mutagenicity baselines

### Background
Mutagenicity is available both as a raw/TU graph dataset and as raw, removed,
and curated SMILES tables. Allowing each baseline to choose its own source or
label encoding would make cross-method recourse results incomparable.

### Decision
Use the 4,247-row curated SMILES CSV as the primary v1 benchmark source, with
`1=mutagenic`, `0=non_mutagenic`, and main recourse direction `1 -> 0`. Audit
the 4,337-row raw CSV against TU graph order using the uniquely verified
inverse TU mapping. Preserve isomeric chemistry, reject invalid or disconnected
molecules, deduplicate canonical isomeric SMILES, exclude label conflicts, and
build one deterministic label-aware 70/10/10/10 scaffold-group split shared by
all methods.

### Consequences
- Raw files remain immutable provenance inputs.
- No neutralization, tautomer canonicalization, or stereochemistry removal is
  performed in v1.
- Ours and every baseline must use the same processed benchmark and split
  manifest for final Mutagenicity comparisons.
- Smoke outputs are isolated from the canonical full processed directory.

### Status
Accepted

---
## [2026-07-22] Fit the Mutagenicity RF teacher on train only

### Background
The unified Mutagenicity benchmark provides fixed train, validation,
calibration, and test splits. Reusing the older AIDS script's random holdout
would break this dataset contract and risk calibration/test leakage.

### Decision
Train a Morgan fingerprint RandomForest only on the fixed train split, select
its hyperparameters only on validation balanced accuracy, and reserve the
calibration and test splits for probability/threshold calibration and final
teacher-quality reporting respectively. Persist the model in the existing
oracle bundle format so shared teacher consumers can read it without changing
their core logic.

### Consequences
- The teacher data root is
  `outputs/hpc/datasets/final/mutagenicity_v1_processed`, not the parent full
  run directory.
- Every split receives the same probability and classification metric audit.
- Calibration and test metrics cannot influence model selection.

### Status
Accepted

---
## [2026-07-22] Reuse the AIDS SFT v3 target constructor for Mutagenicity

### Background

The processed Mutagenicity benchmark and RF teacher now provide fixed,
teacher-consistent source-label train and validation views. A new ChemLLM SFT
and stable-PPO data path must preserve the existing counterfactual-fragment
objective without inventing molecule-level pseudo-targets or admitting
calibration/test examples.

### Decision

Build Mutagenicity SFT targets by directly reusing
`select_reference_candidate_for_parent()` from the AIDS SFT v3 implementation.
This preserves the existing projection and Murcko candidate sources, core
normalization, dummy-atom audit representation, exact parent matching, size and
non-empty-residual filters, and optional deletion-based RF ranking. Build PPO
prompts one-to-one from every validated teacher-correct source parent and retain
the stable `molecule_id`. Read calibration/test files only as exclusion
manifests and fail on molecule, canonical-SMILES, or scaffold leakage.

### Consequences

- Mutagenicity uses the same weak target semantics as AIDS SFT v3 rather than a
  newly invented fragment label.
- Parents for which that constructor finds no target are explicit SFT misses;
  they are not assigned fallback pseudo-fragments.
- The PPO prompt set can cover all valid source parents independently of SFT
  target coverage.
- Existing AIDS, SFT trainer, stable PPO, selector, WNode, and CCRCov code paths
  remain unchanged.

### Status

Accepted

---
## [2026-07-22] Rebuild Mutagenicity teacher views from processed metadata

### Background

The first teacher-consistent Mutagenicity files were filtered directly from RF
prediction CSVs. Those files contain IDs, SMILES, labels, and predictions but
not the processed benchmark's `semantic_label`, `scaffold_smiles`, or `split`,
so they cannot satisfy the SFT/PPO parent and leakage-audit contract.

### Decision

Treat each fixed processed split as the authoritative metadata source and
strictly one-to-one join its corresponding RF prediction CSV by
`molecule_id`. Reject duplicate or missing IDs, row-set differences, SMILES or
label mismatches, and split mismatches. Preserve all processed columns, append
teacher fields and the fixed `source_label=1,target_label=0` direction, then
derive source-all, source teacher-correct, and target teacher-correct views.
The SFT/PPO smoke and full wrappers rebuild these views before data generation.

### Consequences

- Scaffold and semantic metadata are preserved from the fixed benchmark; they
  are never fabricated or recomputed from prediction-only files.
- No inner join may silently discard a molecule.
- Calibration and test source views remain exclusion manifests and do not
  become SFT/PPO training inputs.
- Existing teacher, ChemLLM, stable PPO, selector, WNode, and baseline logic is
  unchanged.

### Status

Accepted

---
