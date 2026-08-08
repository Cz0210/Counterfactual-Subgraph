# BBBP Ablation Experiments

Status: `FRAMEWORK_ONLY_NOT_RUN`

## Candidate source

The frozen variants are `chemllm_ppo`, `chemllm_sft_only`,
`random_connected_size_matched`, and `random_brics_size_matched`.
`random_topk_from_chemllm_pool` is diagnostic only. All variants share parent
cohort, candidates per parent, teacher, selector, WNode, threshold, K, and
test. Random size matching may use train/validation reference statistics only.
Generators are deterministic by seed, bounded by `max_attempts`, preserve the
common candidate schema, record shortfalls, and never silently use test data.

Planned root: `outputs/hpc/eval/ablation/bbbp/candidate_source_v1/`.

## Selector components

One configurable selector implementation supplies nine frozen variants:
`full_selector`, `no_cf_term`, `no_coverage_term`,
`no_structural_redundancy`, `no_coverage_redundancy`, `no_size_penalty`,
`cfdrop_only`, `coverage_only`, and `random_topk`. It controls
`alpha_cf`, `beta_coverage`, both redundancy weights, `eta_size`, mode, and
tie-break seed. Candidate pool, calibration cohort, teacher, WNode, K, and test
remain identical.

Planned root: `outputs/hpc/eval/ablation/bbbp/selector_v1/`.

## Candidate budgets

Budgets are 1, 2, 4, and 8 candidates per parent. A single ordered max-budget
pool is generated; lower budgets are deterministic nested prefixes. Reports
record requested/effective candidates, unique candidates, pair count, and
runtime/memory fields as `null` until measured.

Planned root: `outputs/hpc/eval/ablation/bbbp/candidate_budget_v1/`.

## Seeds and uncertainty

Every plan freezes seeds 0, 1, and 2 for split, generation/random controls,
selector tie-breaking, and aggregation. Bootstrap resampling is parent-level;
candidate-pair rows are rejected. The default future count is 1000 and output
contains mean, standard deviation, median, and 2.5/97.5 percentiles for
CCRCov, cost, CFDrop, FlipRate, ValidRate, StructRed, and CovRed.
Figure 3/4 confidence bands use the same parent universe at every K/threshold
and the `parent_level_curve_confidence_band_v1` schema.

No performance value or runtime placeholder is filled with zero before an
experiment is run.
