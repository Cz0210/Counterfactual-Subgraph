# BACE GIN method-specific fixed-pool comparison

This independent experiment never replaces or publishes the GINE main matrix.
It is post hoc: previously observed backbone results motivated the choice of
GIN. Four methods preserve their own historical generation oracle and native
operation; no retraining, temperature fitting or candidate generation occurs.
The original Ours66, native GCF21958, ComRecGC44 and GlobalGCE80 inventories are
frozen before any new test access. GlobalGCE's original fixed hard outputs do
not currently form connected complete molecules and are BLOCKED_MATERIALIZATION,
not a measured zero. Other methods proceed independently.

`scripts/experiments/run_bace_gin_fixed_pool.py --help` is the actual isolated
mode-safe CLI. An immutable JSON spec supplies existing artifact identities;
the YAML under configs/experiments documents policy, not pretend deployment.
Stages are plan → verify-calibration (stable parent ranges) → freeze (one full
global method selector) → evaluate-test → aggregate → export. Resume reads the
same spec and skips sealed complete parent units. Test raw-cost adoption is
created only after the method's new calibration order has been frozen.

The base denominator remains all66 calibration /141 test parents. Predicted0
base parents remain failures in the primary denominator. The same sequence is
also reported on the auxiliary GIN-native subset. K1..20 is AT_MOST_K; unavailable
rules are never padded. Fixed capped mean remains the primary cost. Conditional
median, finite availability and exact ECDF are additional diagnostics.

The CPU Slurm wrappers are a task-specific exception to the repository's GPU
training default. Submit from a pinned immutable worktree with real absolute
spec path, ≤2 heavy jobs, and measured memory admission. GCF's full selector has
quadratic candidate matrices: no hidden cap is added to avoid that cost. It
requires separate measured admission before selection; 64GiB is an upper
resource allowance, not proof the stage fits.

`replot_bace_gin.py` consumes only exact exported CSVs. Missing methods are
explicit PARTIAL/UNDER_REPAIR, not fake zero curves. A new run is not scientific
PASS merely because all parent jobs or a renderer exited successfully. The
current export state explicitly awaits independent result audit.
