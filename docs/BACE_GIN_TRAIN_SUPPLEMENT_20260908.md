# GIN train-only supplement leaf — 2026-09-08

This is the taskbook §4.2 bounded search leaf, not a new controller, PPO update,
calibration selector or scientific result. It is independent of the running
AIDS RF-aligned CPU owner. The original GINE-generated2607 pool and already
completed GIN verification remain immutable.

## Inputs and first gate

`scripts/experiments/run_bace_gin_reach_supplement.py` accepts the unchanged A+
`--spec`, a fresh `--output-root`, and `--action plan|run|status`. `run` includes
the plan gate, so an Slurm afterok continuation can invoke it directly.

It requires the original adopted2607 train terminal to be
`PARENT_EVALUATION_COMPLETE`, with386 parents and2607 candidates. Every unit
must have the same source spec, exact parent/cohort/GIN weight/temperature,
complete candidate/match accounting, and no reused GINE flip masks. The fixed
base386 is unchanged; only members predicted source1 by GIN and not flipped
by any original2607 rule enter search. Distance thresholds are not used to
declare a source parent unreachable.

The original search receipt supplies the complete `search_budget`, including
minimum atoms and maximum deletion fraction. No size limit is silently widened.
Original GINE generation queries are separately disclosed, not called zero.

## Search, cache and recovery

The original connected hard-deletion `search_parent` implementation is reused.
One optional argument, `initial_oracle_cache`, defaults to the previous empty
cache behavior and cannot be supplied with a previous-pass checkpoint.

Same-GIN saved residual probabilities are reusable only when parent, residual,
model, temperature and feature-schema bindings are valid. Conflicting saved
probabilities are not averaged or cherry-picked: that graph is excluded from
the initial cache and any necessary new inference counts as a new query. There
is no MolCLR, OT, standalone-fragment classifier or old GINE cache in this leaf.

Each eligible parent receives at most128 new queries. Remaining uncovered IDs
are sorted and shuffled using a local `random.Random(7)`; at most128 receive one
additional384-query pass. Global RNG state is untouched. Both passes, actual
queries and cache hits are recorded. Repeated extra passes and cumulative
queries above512 fail closed. Each completed pass is immutable and adopted
on resume. An intent without a completed pass is an explicit incomplete-query
accounting blocker, not silently rerun with its budget reset.

`retain_train_pool` preserves all original2607 records in their original order
and excludes their canonical graphs from additions. New train witness rules
are capped using the existing observed train-support policy; total rules≤4096.
Witness support is not called exhaustive class-level/calibration coverage.

## Outputs and downstream contract

`supplement_contract.json` binds the source spec, train units, exact old pool,
GIN adoption, source size/query budget and eligible train IDs.
`extra_parent_order.json` seals the first-pass state and local-seed7 order.
Per-parent pass receipts contain actual action records and complete predictor
cache/frontier for the additional pass.

`candidate_freeze.json` contains:

- `state=TRAIN_ONLY_POOL_FROZEN`, or `NO_SUPPLEMENT_REQUIRED` when no train gap;
- `source_spec_sha256`, `supplement_contract_sha256`;
- `candidate_universe_path`, `candidate_universe_sha256`;
- `source_candidate_count=2607`, actual `candidate_count`, `new_candidate_count`;
- query counts, extra IDs and cache-hit accounting;
- `old2607_content_unchanged=true`, `calibration_loaded=false`, `test_loaded=false`;
- `additional_ppo_updates=0`, `main_matrix_write=false`.

When there is no gap the freeze points to the already sealed2607 file; it does
not reload a model or emit a duplicate pool. A new downstream driver must bind
this exact freeze and new pool before missing-only calibration and a new global
selector freeze. This leaf never reads calibration/test records and does not
consume their publicly reported performance in its branch decision.

The paired CPU submission script is
`scripts/slurm/run_bace_gin_reach_supplement.sh`: submit from the verified
immutable worktree using Slurm `--chdir`,8CPU/32GiB, no GPU. This is the explicit
taskbook CPU exception to the repository's general GPU template. Inference is
the frozen GIN oracle, not the general heuristic inference pipeline; the latter's
fallback flag is not applicable. No job has been started merely by adding this
leaf.

Focused validation: `tests/test_bace_gin_reach_supplement.py` plus the unchanged
`tests.test_bace_reach_v2.SearchTests`. Covers source/match binding, cached GIN
probabilities, strict reach versus threshold, seed7 isolation, bounded passes,
incomplete query intent rejection, unchanged2607 and no-gap no-model behavior.
