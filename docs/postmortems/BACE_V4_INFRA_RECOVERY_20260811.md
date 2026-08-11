# BACE v4 infrastructure recovery, 2026-08-11

## Scope

This recovery is limited to BACE v4/common4. AIDS COMRECGC, Mutagenicity
COMRECGC, pi05, goal-l4, long-norm, LIBERO, and unrelated jobs are protected.

## Failures

- Jobs 2231029, 2231215, and 2231679 ended at launch with zero elapsed time,
  no Python log, and exit `0:53`. Two ran on gpu8020 and one on gpu8015, so the
  evidence does not identify one common failed node.
- Job 2232102 reached Python and failed because official GlobalGCE indexes
  `MIN_FREQ[dataset.dataset_name]`, while the official table has no BACE entry.
- Job 2232106 reached Python and failed because its independent project
  worktree did not contain the relative `external/COMRECGC` checkout.

## Recovery

Retry roots depend on a short launch preflight and use Slurm requeue without a
node exclusion. GlobalGCE minimum frequency is selected on calibration from a
fixed ratio-derived grid, then frozen in a manifest. COMRECGC uses an immutable
local checkout at the fixed upstream commit, verified before generation. Old
outputs remain evidence and retry2 writes only versioned roots.
