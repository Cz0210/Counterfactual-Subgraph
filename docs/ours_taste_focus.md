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

`run_ours_taste_train_matrix.py --config configs/hpc.yaml --spec ABSOLUTE_SPEC`
is an AutoDL-only finite GPU stage. It uses the existing UUID GPUFileLock in the
science process itself, checks real headroom/storage/GPU occupation, loads the
original frozen GINE and MolCLR and evaluates all P0 rules on the 256 frozen
train-development IDs. Completed parent units are written to one gzip segment
and a compact matrix; NaNs remain for unfinished rows. SIGTERM/SIGINT request a
stop after the current parent, preserving its durable result. This initial
entrypoint refuses an existing output root (no hidden restart/recomputation).
The paired Slurm file is **CPU CLI validation only**, never GPU science on HPC.

The same AutoDL entrypoint now accepts a sealed `stage=SEARCH_CALIBRATE_TEST`
spec bound to `train_matrix_root` (a completed P0 train stage), `selection_B_path`
(the already computed P0/B freeze), and optional `seed_node_database` (a closed,
completed prior node cache, adopted through SQLite backup). It performs one
bounded train search round, then the entire delta-evaluation/freeze/test chain.
It does not call a language model or PPO training. New C/D exist only when a
genuinely new rule pool exists. The caller must wait for the prior owner to exit
and acquire the original UUID lease; no simultaneous restore/science copies.
