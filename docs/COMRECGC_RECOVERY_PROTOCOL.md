# COMRECGC Recovery Protocol

## Recovery origin

- Recovery tag: `pre-comrecgc-recovery-20260806T064254Z`
- Fixed project base: `60e32989e944c15bdc8ac7ecd73c68850f1a2a18`
- Recovery branch: `baseline/comrecgc-recovery-20260806`
- Local recovery worktree: `/private/tmp/counterfactual-subgraph-comrecgc-recovery`
- Fixed upstream: `122f9341a360e9f06bb58a2f5823bb596021f6bf`
- Upstream source policy: fetch into ignored `external/COMRECGC`; do not vendor it.

The original local worktree was deliberately left untouched. At recovery start it
was on `main` at `7d25a193e230226df118c5050ff867b3d5d3d0e4` with the following dirty-path
manifest:

```text
 m baselines/clear_official
 M docs/EXPERIMENT_LOG.md
 M src/baselines/clear_rf_adapter.py
 M tests/test_generate_gcf_style_recourse_report.py
 M tests/test_merge_candidate_pools.py
?? paper/
?? scripts/slurm/audit_node_fgw_threshold_consistency.sh
```

No recovery operation may reset, stash, clean, overwrite, or otherwise modify
those paths.

## Scientific boundaries

The pinned upstream random walk, importance calculation, neighbor ordering,
transition probabilities, reinforcement, DBSCAN parameters, centroid filtering,
and greedy ordering remain unchanged. Project-owned code may add trace-only
instrumentation, stable provenance, dataset adapters, and a preregistered
deterministic chemistry projection.

The primary adapted chemistry policy is one raw candidate to at most one output.
It must not consult RF, strict-flip, or WNode while repairing; must not search
alternative actions, atom labels, charges, or bond orders; and must preserve
official rank slots without compaction or backfill.

An empty common-recourse set, zero valid representatives, zero strict flips, zero
coverage, or an unavailable conditional cost is a valid scientific result after
the engineering and provenance gates pass. Such an outcome must not be rewritten
as an engineering failure or converted to a numeric zero cost.

## Execution boundary

Heavy generation, embedding, clustering, chemistry replay over full artifacts,
and unified evaluation run only through registered Slurm jobs. Recovery state is
stored under `outputs/hpc/automation/comrecgc_recovery/<run_id>/` with append-only
events and evidence paths. Successful stages are reusable and retries are bounded.

## Frozen blocker identities

- AIDS native candidate artifact:
  `outputs/hpc/baselines/comrecgc/native_smoke/aids/comrecgc_native_common_64p_20260806_v6/counterfactuals.pt`
  (`340685` bytes, SHA256
  `096ddd0f4ac31126a0665a11effb7362c2137229ff6b53b50e16c081ef6c274a`).
  Its exact blocker funnel is 31 model counterfactuals, 1,984 distance
  pairs, 28 theta-eligible pairs, zero DBSCAN clusters, and zero selected
  recourses.
- Mutagenicity project artifact:
  `outputs/hpc/baselines/comrecgc/mutagenicity/smoke_comrecgc_smoke_budget_retry_20260806_v4/generation/counterfactuals.pt`
  (`953049` bytes, SHA256
  `060879cbaf69b1e3279301350f587cab809d48991559a80ff5227c46466df8d0`).
  It contains 164 candidates from the fixed 64-parent smoke. The downstream
  official common-recourse artifact contains 70 model counterfactuals, 4,480
  pairs, 90 eligible pairs, and four original official medoids.

The resolver accepts these only when exact statistics, manifest lineage,
fixed upstream commit, and file SHA256 agree. Directory names alone are not an
identity proof.

## Trace and chemistry gates

Action tracing is project-owned runtime instrumentation around the pinned
upstream functions. It records enumerated actions and the selected transition
predecessor without making RNG calls. Consumed neighbor maps are released after
each move and candidate paths are reconstructed from first predecessors, so
full tracing is linear in selected moves rather than quadratic in path length.
Trace-on output must match the frozen trace-off payload in normalized graph
topology, node features, frequency, importance, and candidate order.

Mutagenicity repair starts only after 100% source and no-op round-trip plus
trace parity. New untyped edges use the existing project decoder's SINGLE rule.
Each official action is attempted once; sanitize failure rolls back that action,
and a later action referencing a rolled-back node is skipped. The repair does
not import or call RF, MolCLR, WNode, or strict-flip logic. The original official
medoid and rank slot remain authoritative even when the repaired output is a
no-op or invalid.

## Native full semantics

The pinned upstream defaults are parsed from the fixed checkout and compared
with the project full contract before preregistration. AIDS native full uses all
official `prediction == 0` parents. A completed run with no eligible DBSCAN
cluster is recorded as `EMPTY_COMMON_RECOURSE`, with coverage zero and cost
`null`/N/A. It is not converted into an engineering failure.

## Slot-preserving unified evaluation

The shared `evaluate_ccrcov_with_molclr_node_wasserstein.py` implementation is
the only component allowed to query the frozen RF teacher or calculate WNode.
Only deterministically repaired, RDKit-valid original medoids are passed to it.
The project-owned slot adapter then restores those pair rows to the immutable
official cluster ranks and emits explicit unavailable rows for invalid medoids.

The final prefixes always represent requested `K=1..20` official slots. An
invalid rank contributes no coverage and is never replaced by a later rank.
`requested_k`, `valid_k`, `invalid_slot_backfill=false`, and
`rank_compaction=false` are persisted in Figure 3 and Table 2 artifacts. If no
strict flip exists, coverage is zero and conditional cost remains `null`/blank,
while the engineering run may still pass. Figure 4 uses the frozen
Mutagenicity threshold artifact verbatim and never fits a threshold on test.

## Submission and freeze boundary

All jobs are submitted through `scripts/exp_sbatch.sh`. Mutagenicity full jobs
are not registered until `mut_chemrepair_smoke_gate` has reached
`COMPLETED/0:0`; this prevents known code failures from leaving a chain of
`DependencyNeverSatisfied` jobs. Once the smoke gate passes, generation,
official common recourse, chemistry, unified evaluation, full gate, and freeze
are registered as one dependency chain.

The project-comparable Mutagenicity result is atomically materialized under a
new `mutagenicity_common4_comrecgc_standardized_v*` root only after the full
gate passes. Existing paper roots are never overwritten. AIDS native full is
kept under the native baseline root and is not frozen into a project AIDS paper
root because TU/AIDS and the project's AIDS/HIV parent universe are distinct.
