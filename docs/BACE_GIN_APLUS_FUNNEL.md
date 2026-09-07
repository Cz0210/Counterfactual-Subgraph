# A+ saved-record funnel export (2026-09-08)

The dataset-specific exporter reads completed parent receipts only. It does not
load a model, generate candidates, select rules, calculate distances, or publish
the main matrix. It preserves the frozen threshold and cost-cap definitions.

Calibration reports all candidates in the bound campaign contract (2,659 for
the current derived campaign). Test reports K=10 and K=20 for each of the three
actual frozen controls. Test is never described as a full-pool evaluation.
An actual campaign-bound selector freeze and complete test terminal are checked
before any test parent is read. Without a test terminal, only calibration is
exported and test remains PENDING.

Each large parent container is read once. The output includes method counters,
parent-level details, and the ordered first-failure decomposition. A missing
counter stays blank/undefined, with its known sum and number of missing records
reported separately; it is not replaced by zero. These are saved-record
diagnostics, not renewed model or chemistry validation.

CLI: `scripts/experiments/export_bace_gin_aplus_funnel.py` requires `--config`,
`--spec`, and a fresh absolute `--output`. Its paired Slurm script is
`scripts/slurm/export_bace_gin_aplus_funnel.sh`: the explicit CPU-only exception
uses 2 CPU / 8 GiB, with no GPU, and is submitted from an immutable worktree.
No artifacts or production execution are implied by the focused fixture tests.
