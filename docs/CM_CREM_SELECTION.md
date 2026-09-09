# CM-CReM selection and record export

2026-09-10: independent CM-CReM-Global-Budgeted-v1 adaptation, not a change to
the repository's deletion-fragment objective or original 16-cell matrix.
Generated molecules are full-graph endpoint prototypes, not reusable deletion
rules. The original frozen GINE and its original temperature remain the oracle;
this code does not call or alter that oracle, MolCLR, OT, generation or training.

## Architecture and integration API

`src/baselines/cm_crem_selection.py` is a small NumPy-only global selector and
frozen-prefix evaluator. `src/baselines/cm_crem_export.py` handles record-only
CSV/LaTeX/PNG/PDF and offline replot. The stage driver authenticates the train-only
pool, oracle/split/distance contract, completed pair shards, independent saved-record
audit and stage ordering. Library validation does not replace those authorities.

```python
freeze = select_calibration(
    calibration_distances, pair_status=calibration_pair_status,
    parent_ids=calibration_ids, candidate_ids=frozen_pool_ids,
    source_mask=calibration_source_mask, theta=theta, cap=cap,
    contract_sha256=contract_sha256, frozen_pool_sha256=pool_receipt_sha256,
)
# Persist freeze.to_dict() and authenticate it before computing test distances.
# There must be only min(20, M) test columns, in this exact order.
selected_ids = freeze.selected_candidate_ids
evaluation = evaluate_frozen_test(
    freeze, test_distances, pair_status=test_pair_status,
    parent_ids=test_ids, candidate_ids=selected_ids,
    source_mask=test_source_mask, contract_sha256=contract_sha256,
)
export_results(evaluation, run_root, dataset="bace", oracle="gine")
```

The evaluator also supplies strict-JSON `to_dict()` and
`PrefixEvaluation.from_dict()` for a saved `test_evaluation.json`; semantic
infinity is encoded as the string `inf`, not JSON NaN/Infinity. Export accepts
either the typed result or its authenticated mapping. A driver's optional outer
`science_hash` must match `contract_sha256` before the internal hash is checked.
The driver must reopen/validate `SelectionFreeze` **before test features are
loaded or test distances computed**, not merely when auditing them afterward.

IDs are unique nonempty canonical **strings**; masks are true Boolean vectors.
The full base cohort remains in both matrices, including before-not-source rows.
The contract SHA must bind the actual dataset, original GINE weights/temperature,
parent split identities, source/destination predicate, WNode numerical backend,
theta and cap. The pool SHA must identify the already authenticated, train-frozen
<=2000-candidate library, frozen before loading calibration. Neither value is
inferred from an arbitrary directory name. `SelectionFreeze.from_dict` verifies
the JSON self hash, complete pool ID order, selected ID order, calibration
IDs/mask/matrix hash and both external identities. A hash is an integrity binding,
not proof that a forged input was scientifically produced: upstream stage receipts
and saved-record audit remain required.

### Pair statuses are mandatory

- Finite, nonnegative distance: `OK` or `FINITE` only.
- Semantic infinity: `INVALID`, `SEMANTIC_INVALID` or `NO_STRICT_FLIP`.
- Every non-source parent row: `NON_SOURCE` or `BEFORE_NOT_SOURCE`, all infinity.
- NaN, negative values, negative infinity, missing pair rows, failed calculations,
  `ERROR`, `PENDING`, `UNCOMPUTED`, and computation `TIMEOUT` are rejected.

An upstream numerical failure cannot become a semantic infinity. The driver must
not convert an infrastructure error into an `INVALID` label. A legitimate empty
generation pool is different: matrix shape `(N, 0)`, statuses `(N, 0)`, positive N,
and empty candidate IDs are valid. No finite source-false entries are silently
masked after computation; inconsistent masks/statuses are rejected.

## Frozen objective and metrics

Each calibration step selects by greatest newly theta-covered **count**, then
greatest decrease in `mean(min(best_distance, cap))` over all base parents, then
lexicographically smallest canonical candidate ID. No tolerance, Ours Reach,
prediction reranking, shard-local top-K merge or approximate shortlist is used.
Zero coverage/cost gains do not stop the selection. It selects exactly min(20,M)
unique prototypes once; no padding/repetition occurs. The selected prefix is the
same for every K and threshold.

Test accepts only those exact <=20 columns and does not call the selector.
All K1..20 metrics use at-most-K: after the pool is exhausted, the reported
effective K and curve plateau. Coverage uses `best_distance <= theta`. The primary
cost is fixed-denominator capped mean, counting unresolved/non-source parents at
cap. Conditional median uses all finite strict-flip parent-best distances,
regardless of theta; no finite recourse gives `N/A`, not zero. A valid empty pool
produces zero coverage, capped mean=cap, conditional median=N/A.

Exact Figure4 curves retain every unique finite **uncapped** distance (including
distances above cap and distinctions below common rounding precision), with right-
continuous cumulative counts over the complete base N. Infinity remains missing
finite coverage mass, never a fabricated jump at cap or finite-only normalization.
An empty curve has the explicit point (0,0) and unresolved=N. Figure plotting is a
step function, without smoothing or invented intermediate results.

## Artifacts and offline entrypoint

`export_results(evaluation, run_root, dataset=..., oracle="gine", fixture=False,
make_figures=True)` refuses an existing `run_root/results` and creates:

```text
results/export_manifest.json
results/source_csv/prefix_metrics.csv
results/source_csv/parent_best_distances.csv
results/source_csv/figure3_coverage_cost_vs_k.csv
results/source_csv/figure4_k10_exact.csv
results/source_csv/figure4_k20_exact.csv
results/source_csv/table2_k10.csv
results/source_csv/table2_k20.csv
results/figures/figure3_coverage_cost_vs_k.{png,pdf}
results/figures/figure4_k10.{png,pdf}
results/figures/figure4_k20.{png,pdf}
results/figures/table2_k10.tex
results/figures/table2_k20.tex
results/figures/replot_inputs.json
candidate_funnel.csv
candidate_provenance.csv
budget_and_timing.json
```

The manifest records source CSV byte hashes, parent-ID identity, frozen selection
and external contract/pool identities. This module deliberately emits only
`EXPORTED_RECORDS`/`REPLOTTED_RECORDS`, never scientific PASS or `final_audit.json`.
Funnel/provenance and timing sidecars project the driver's authenticated receipts;
upstream completion and independent scientific audit remain driver-owned. A
partial CSV export does not mean final experiment completion.
Synthetic testing sets `fixture=True`, labels every CSV and figure, and cannot
be presented as real BACE/Taste results. Unit tests use `make_figures=False`;
integration owns actual rendering and PDF visual QA before delivery.

```bash
python -m src.baselines.cm_crem_export replot --help
python -m src.baselines.cm_crem_export replot \
  --source-csv /absolute/run/results/source_csv \
  --output-dir /absolute/new-replot \
  --dataset bace --oracle gine
```

`replot(...)` verifies the source manifest and all CSVs before plotting, writes
PNG/PDF/LaTeX and absolute `replot_inputs.json`, and does not connect to a server
or invoke an oracle/encoder/OT/selector. The primary stage driver wraps this API
in the user-requested campaign-aware stable shell entrypoint; the library does
not own site dispatch/Slurm policy. Existing baseline curves are not modified or
mixed into this standalone CM-CReM panel; any combined panel must separately
validate identical oracle, parent IDs, theta/cap and review status.

## Focused verification

```bash
python -m pytest -q tests/test_cm_crem_selection.py tests/test_cm_crem_export.py
```

Fixtures check coverage-vs-cost priority, deterministic ID tie breaking, zero-gain
continuation, at-most-20, a scalar greedy reference, column permutation, frozen
test rejection, semantic/error distinctions, source-false denominator retention,
empty pool, conditional N/A, uncapped exact ECDF and immutable record export.
They do not use model/OT outputs and do not establish scientific completion.

Local focused verification on 2026-09-10: **30 passed** in the existing read-only
`smiles_local` environment; module CLI help, compileall and diff-check passed.
Actual plot rendering and PDF visual QA are a separate integration check.

## Receipt-backed diagnostic sidecars (2026-09-10)

Production `export_results` now calls
`export_diagnostics(run_root, evaluation, fixture=False)` automatically, before
creating `results`. The function requires one actual `spec.json` or
`resolved_spec.json` and the completed `audit/provenance_review.json`. It verifies
the science hash, frozen library/selection, and exact producer-bound bytes of all
receipts used. Missing, changed, or unresolved generation/filter evidence blocks
the export; no fake rows or counts are substituted.

The run-root files contain:

- `candidate_funnel.csv`: one row for every original train parent, including
  non-source parents. Recorded native/raw/chemistry/strict-flip counts are kept
  distinct, with per-parent counts retained in the frozen library and selected
  prefix. Unrecorded or inapplicable values use `N/A`, not numeric zero.
- `candidate_provenance.csv`: one row per actual frozen-candidate origin, bound
  to its train parent, retained raw index/ID/SMILES and filter receipt. Selected
  membership and rank are explicit. A genuine empty library writes only the
  complete CSV header, not a fabricated candidate.
- `budget_and_timing.json`: the frozen specification's budgets, original receipt
  timings, observation/missing counts, and exact source hashes. The complete
  pilot closeout receipt preserves any newly recorded phase timings/resource
  admission fields. Optional `pilot/filter_timing.json` is included only when
  present in the provenance-bound inventory. Missing timing stays `null`; sums
  of measured generation-unit times are not presented as parallel campaign wall
  time. No final campaign wall-time measurement is currently inferred.

Exact repeated sidecar exports verify and reuse identical bytes; conflicting
existing files are rejected without overwrite. `export_manifest.json` binds the
three hashes in `diagnostic_files`, leaving the existing seven plot-source CSVs
and their `source_files` inventory unchanged. This keeps the existing record-only
replot and portable reader interfaces compatible. Numeric-only fixtures without
receipts explicitly report `NOT_AVAILABLE_IN_NUMERIC_ONLY_FIXTURE`.

Focused verification covers empty/nonempty receipt projections, all train-parent
rows, origin indexing, missing-vs-zero timings, changed source rejection,
immutable sidecars, fixture isolation, and unchanged seven-file plot inventory.
No actual generation or final scientific output was produced by these tests.

## Existing four-method comparison: verified input shape, remaining adapter

The supplied local `paper13_audit_plot_kit` has original-GINE BACE rows in
`inputs/figure3_source_reconciled.csv`, K10 in `inputs/figure4_exact_ecdf.csv`, and
141-parent K1..20 minima in `audit/parent_prefix_minima.csv`. Exact K20 ECDF can
be reduced from those saved uncapped minima without another oracle/OT call.
The oracle table is `audit/oracle_matrix.csv`, not `inputs/oracle_matrix.csv`.
The excerpts bind the four original methods to GINE weight hash `4edd23cd…`,
the same MolCLR checkpoint and BACE theta/cap.

The current `replot` remains the CM-only panel. A minimal future optional overlay
needs an adapter from the old kit's `method/cost/threshold/best_distance` columns
to the plot inputs, exact parent-ID-set checks at each K, the verified per-method
temperature/numerical receipts, and input byte hashes. The local offline manifest
alone lists only two source-JSON hashes, not a four-method CSV/temperature closure.
Stage the already independently verified source manifests with the small input
package before enabling that adapter. Preserve the historical hard-graph-validity
caveat on GlobalGCE; do not replace these rows with GIN/A+ or the 80-parent route.
No combined plot or availability claim is made in this bounded implementation.

The official database GET403 is an independent generation asset blocker. This
implementation neither replaces that database nor generates substitute results.
