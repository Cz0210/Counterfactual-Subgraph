# BACE GCFExplainer Summary Failure 2163954

Date: 2026-08-09

## Scope

This postmortem covers completed BACE GCFExplainer VRRW job `2163953`, failed
summary/export job `2163954`, and the exact dependent jobs `2163955` and
`2163956`. It does not cover or modify AIDS, Mutagenicity, BBBP, or BACE Ours.

## Job evidence

- `2163953` completed with exit code `0:0` after 50,000 VRRW steps.
- Its immutable `counterfactuals.pt` contains 45,488 candidate records, 45,488
  graph-map entries, and 50,000 traversed hashes.
- 38,637 saved candidates satisfy the frozen model-counterfactual importance
  predicate (`importance_parts[0] >= 0.5`) and resolve to a saved graph.
- `2163954` was not an OOM or VRRW failure. It failed closed because the export
  retained only 2 of the requested 20 RF-target, chemically valid candidates.

## Root cause

`build_bace_native_summary` copied the upstream summary's default shortlist
rule: stop after collecting as many model-counterfactual graphs as there are
source parents. For BACE this is 360. The `360/360` progress therefore meant
360 native-summary candidates, not 360 random-walk steps and not the complete
VRRW candidate pool.

The exporter did inspect all 360 saved native ranks. Their structured attrition
was:

- retained: 2 (native ranks 92 and 330);
- generated valence sanitization failure: 248;
- generated aromaticity/bond inconsistency: 73;
- generated kekulization failure: 5;
- other generated sanitization failure: 1;
- RF prediction did not match target label 0: 31.

Thus the exporter itself did not stop after rank 20. The shortlist supplied to
it had already discarded 38,277 other saved model-counterfactual states.

## Codec audit

The prepared BACE graph codec was applied to every unedited prepared graph:

- train: 869/869 exact round trips;
- validation: 162/162 exact round trips;
- generation source: 360/360 exact round trips.

Atom identity, formal charge, aromaticity, explicit hydrogen state, chirality,
bond type/stereo, connectivity, and canonical molecular identity all match.
Explicit hydrogen nodes are an intentional part of the frozen nine-channel
BACE representation. The shared chemistry implementation receives the BACE
schema and does not decode candidates with the Mutagenicity atom vocabulary.

The high generated-graph attrition is therefore an observed property of the
untyped official edit space, not a source-graph mapping failure. No generated
candidate is chemically repaired in this fix.

## Frozen-pool sufficiency validation

A CPU-only, validate-only scan of the immutable `2163953` payload established
that a new VRRW run is unnecessary. In its stored candidate order, the audit
inspected 14,661 of 38,637 model-counterfactual records before retaining 20
unique, sanitized teacher-target candidates. Among those inspected records,
209 decoded and sanitized successfully, 195 were unique and teacher-evaluable,
and 20 predicted target label 0. The structured rejection audit records 14
canonical duplicates, 10,870 valence failures, 3,493 aromaticity failures, 79
kekulization failures, seven disconnected/empty graphs, three other sanitize
failures, and 175 candidates that did not satisfy the RF target condition.

This ordering is an audit-only sufficiency check, not the final NeuroSED greedy
rank. The formal retry recomputes the frozen NeuroSED relation and official
greedy sequence over the full saved model-counterfactual pool before applying
the same sequential validity filter.

## Fix

The completed 2163953 artifact remains the sole VRRW input. A new summary mode
uses all saved model-counterfactual states when `native_candidate_limit=0`,
computes the same frozen NeuroSED coverage relation, and applies the same
official greedy maximum-uncovered-parent ordering. The implementation only
fast-forwards the deterministic zero-gain tail: once every remaining coverage
set is empty, upstream necessarily emits the remaining insertion-ordered keys.
Regression tests compare this fast path against a literal implementation of
the upstream loop.

Large expanded summaries store native-rank graph hashes plus the immutable
VRRW path and SHA256 instead of duplicating every graph object. The exporter
rehydrates those exact graphs, scans native ranks until 20 unique RF-target
candidates are retained or the full pool is exhausted, and records every
decode, sanitization, deduplication, and teacher rejection. It never repairs,
copies, backfills, compacts, or WNode-reranks candidates.

If the complete pool still contains fewer than 20 valid target candidates, the
new path fails closed with `INSUFFICIENT_VALID_NATIVE_CANDIDATES` and preserves
the full attrition audit.

## Retry runtime finding

The first two registered retries never launched because the isolated worktree
did not yet contain its ignored `logs/` directory while the Slurm directives
used relative `logs/%j.out` and `logs/%j.err` paths. Creating that directory
before submission allowed the shell and Python process to start normally.

The resulting full summary then exposed a separate identity-contract issue:
multiple native records can share `stable_graph_candidate_id` when they encode
the same structural graph. Those records are valid members of the ordered
native sequence; `native_rank`, not structural candidate ID, is the unique row
key. The loader now retains every rank and leaves duplicate removal to the
existing sequential canonical-SMILES audit. Final exported candidate IDs remain
unique and no rank is reordered, repaired, compacted, or backfilled.

## Safety

- VRRW `2163953` is adopted, not rerun or modified.
- Failed output from `2163954` is preserved.
- Only recursively proven descendants `2163955` and `2163956` were cancelled.
- Retry outputs use fresh versioned summary/export and paper roots.
- Calibration and test parents do not participate in candidate selection.
