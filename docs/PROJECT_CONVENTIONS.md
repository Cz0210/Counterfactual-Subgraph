# Project Conventions

This document records repository-level conventions that should guide future experiments, baseline adaptation, reporting, and paper-facing summaries.

## 2026-06 Evaluation Convention: Native-action CCRCov

- Current main metric: CCRCov, Close Counterfactual Rule Coverage.
- SuppCov is paused for cross-baseline comparison. SuppCov is temporarily reserved for subgraph-specific auxiliary analysis.
- Ours remains deletion-only.
- Baselines may use native action types.
- Addition / replacement edits are allowed for baselines that natively produce them.
- Local CF methods must be summarized into top-K global candidates.
- Each dataset has exactly one frozen evaluation teacher.
- All reported metrics must be recomputed by the unified teacher.
- The main table uses native-action CCRCov.
- The deletion-only table is auxiliary.

## Baseline Fairness Convention

All baseline comparisons must use:

- the same dataset split;
- the same frozen dataset-level teacher;
- the same K values, normally $K \in \{1, 5, 10, 20\}$;
- the same cost threshold, especially CCRCov@0.10 and CCRCov@0.20;
- the same distance family within a table, either normalized GED or teacher embedding distance;
- the same final reporting teacher, even if a baseline uses an internal model to generate candidates.

Original baseline paper metrics may be recorded as historical or diagnostic context, but the final cross-baseline comparison must be recomputed under the project CCRCov protocol.

## Deletion-only Convention

Ours is a deletion-only subgraph method. Its native intervention is hard deletion:

$$G^a = G \setminus s.$$

Deletion-only evaluation is useful for analyzing methods that naturally support deletion or can be meaningfully projected to deletion. It should not replace the native-action main table, because full-graph, addition, replacement, and rule baselines may be unfairly distorted by deletion-only projection.
