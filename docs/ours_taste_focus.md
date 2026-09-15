# Ours–Taste focus: executable saved-matrix stage

The supplied `ours_taste_focus_v1.yaml` is the campaign configuration. Its
contract placeholders resolve from the current published Taste Ours manifests,
not from a historical screenshot. `run_ours_taste_focus.py` currently executes
the saved-matrix adoption, full-P0 calibration bounds and P0/B selector stage.
It does not label an absent train matrix or expansion/test stage as READY/PASS.

Actions (all require `--config configs/hpc.yaml`):

- `extract --source PUBLISHED_ROOT --output FRESH_COMPACT_ROOT`: stream original
  pair JSONL once; preserve source/feature/oracle/operation identities, check
  Cartesian completeness, write small NPZ and separate adoption receipts.
- `bounds-select --source COMPACT_ROOT --output FRESH_ANALYSIS_ROOT`: calibration
  MILP at theta/finite reach (120s each), genuine bounded selector; old selected
  test replay is separate and never supplied to the optimizer.
- `status --source COMPACT_ROOT --output ANALYSIS_ROOT`: actual terminal records.

The paired CPU Slurm wrapper requires `OURS_TASTE_CODE`, `OURS_TASTE_INPUT`,
`OURS_TASTE_OUTPUT`. It requests no GPU and hides CUDA. Missing artifacts are
PENDING. Source infinity means no finite recourse under that saved contract,
not missing work. Source predictions retain all base-cohort denominator rows.

Calibration B is a provisional development freeze, not the final recommendation
among A/B/C/D. Train development requires scaffold/margin-stratified source IDs;
generation may not begin by interpreting unknown matrix rows as uncovered.
