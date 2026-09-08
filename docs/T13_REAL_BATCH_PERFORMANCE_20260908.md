# T13 real-batch performance canary (future process only)

The existing official generator + differentiable frozen GINE bridge implements
Taste GlobalGCE replacement. The only optimized science module reused here is
the previously reviewed `_hard_graph` / `_one_graph` materialization from
`d757b030`; the running c0eb892d worker is not modified.

`scripts/benchmarks/benchmark_t13_real_batch.py` accepts `inspect`, `status`,
`run`, and a narrow `verify-checkpoint` action. All calls require an existing
`--config` and `--set inference.fallback_to_heuristic=false` with `python -I -B`.
The runner consumes a sealed plan JSON; plan fields are validated by
`src/baselines/t13_real_batch_performance.py`. It does not claim a formal quota,
open test/calibration datasets, run mining, train a new oracle, or publish a cell.

The runner uses the exact source cohort and compact index/mask identity. The
current production artifacts only retain their identity, not the compact index
arrays, so reconstruction of that same index is required to extract real
batches. If reconstruction exceeds the 30 minute total bound, the result is
INCOMPLETE, not synthetic PASS. It never creates a full eager augmented dataset.
Real train batches are the first two (500 each) from the pinned source training
loader. Validation is the first full real native validation batch. Three arms
use these same distinct batches: old bridge, optimized bridge, optimized bridge
with a fresh model/optimizer reload after its first bounded update. These
diagnostic two updates are explicitly not two full official five-batch epochs.

The comparison records generator outputs, component losses, gradients,
parameters, optimizer, scheduler, RNG, and the original official validation
function's output. Reload uses a CPU-mapped atomic diagnostic container to
preserve Adam counter placement, and a second isolated process checks its
semantic digest. A successful short comparison remains scoped to those batches;
it does not prove the full 100-epoch trajectory or authorize checkpoint promotion.

## Owner and resource interface

`run` requires an actually inherited `--held-gpu-fd` and `--owner-evidence`.
No direct standalone launch is safe. The existing owner must refresh evidence
at least every 60 seconds (age <=120s), bind plan SHA, owner/child PID + start
ticks, existing GPU2 lock path, and full UUID. Child verifies the inherited inode
and independent nonblocking contention. The child rereads cgroup/statvfs and
preserves other tasks' headroom plus its remaining 64GiB budget.

Required evidence fields are `observed_at_epoch_seconds`, `resource_admission`,
`owner_pid`, `child_pid`, `owner_start_ticks`, `child_start_ticks`, `gpu_index`,
`target_gpu_uuid`, `gpu_lock_path`, `borrow_enabled`, `plan_sha256`, and
`other_task_headroom_required_bytes`. These fields must be mapped from actual
existing owner/provider observations, never hand-filled PASS receipts.

Current deployment gap: the general existing GPU2 owner has not yet been
bound to this canary child/evidence schema. This runner intentionally does not
create another waiting controller. The paired Slurm script is inspect-only;
the actual Taste GPU path stays on AutoDL after P0 resource admission.

Current live handover gap: `run_native_branch` supplies no epoch pause callback.
Do not SIGTERM a healthy target0 to adopt this optimization. Keep the current
run intact, and only use a later genuinely safe original stage boundary if the
existing execution chain exposes one. No same-run handover has been performed.
