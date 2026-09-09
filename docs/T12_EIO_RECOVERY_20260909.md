# T12 committed500 EIO recovery child

Status: implemented locally; no remote deployment, no science launch. Persistent
file fsync is not admitted on the 2026-09-09 recovery probe, so this interface
must not be used to claim that T12 is resumed or that parity passed.

The existing diagnostic implementation restores state, history chain, first-seen
embedding bytes, and external transition records. Its SQLite index is explicitly
non-authoritative. Preserve all old database/sidecar originals as forensic data;
rebuild only a fresh local lookup index from authenticated scientific segments.

## Narrow changes

- `T12CompactHistoryJournal(durable_recovery_index=True)` selects DELETE/FULL and
  checks actual PRAGMA readback. Historical default behavior is unchanged; this
  does not edit the old process/worktree or old SQLite files.
- `run_t12_generation_segment` forwards that option only for explicit-index-root
  resume calls. No model, sampling, RNG, first-seen bytes or record codec changes.
- Existing `run_t12_reference_500_v1.py recovery-segment` calls the new child
  adapter. It does not invoke the fresh reference owner or require a fictional
  natural510 precondition before recovering501–510.
- No parent/owner is created here. Real parent PID/start ticks, inherited locked
  FD, independent lock competitor, full GPU UUID, and a newly executed resource
  provider are required. Child checks ≤120-second resource age and task binding.
- The source500 checkpoint must have a fresh exact relocation receipt and a
  two-round storage acceptance covering file-fsync, rename/reopen and consistent
  snapshot. Local safety reserve is at least2GiB and joint capacity must pass.
- Successful tail output is `RECOVERY_TAIL_COMMITTED_NOT_FULL_PARITY`. It is not
  20k production, raw evidence completeness or scientific parity. No matrix write.

## Not yet closed

1. Real persistent file-fsync/snapshot health; presently blocked. A df or read
   success does not satisfy the gate.
2. The root recovery task preserved the10.4MB checkpoint500 copy and checked it
   against the fault-preceding manifest. CPU-only deserialization in the existing
   AutoDL environment then passed the original internal schema/state/RNG digest
   checks: cursor500, traversed500, retained candidate count496, active transition
   count459. All six external segment committed lengths match the file inventory.
   This did not read/replay those external records or open the old SQLite index:
   `joint_restore_verified=false` still applies. Preserve all six segments and
   validate their authenticated prefixes and reconstructed counters before science.
3. The new storage-only source delta needs a reviewed cross-commit receipt for
   its actual immutable deployed commit. Existing strict four-file source binding
   is retained unchanged; this module does not automatically waive new changes.
4. Canonical owner registry CAS and actual FD/provider task binding must be made
   at activation. No imaginary PID or lease is serialized at preparation time.
5. Full local input/index/journal/checkpoint peaks must fit jointly. Six scientific
   segment files total roughly2.63GB, not merely the first242MB history segment.
   Failed disposable indexes are evidence, not authoritative resume inputs.
6. Remaining raw/parity/prospective diagnostics and eventual fresh-zero activation
   remain separate admitted stages under the already authorized budget. This child
   does not fabricate old raw evidence or reset any diagnostic budget.

## CLI

`run_t12_reference_500_v1.py --config ABSOLUTE_CONFIG --set
inference.fallback_to_heuristic=false recovery-segment --task-spec SEALED_SPEC`
is an inherited-owner-only CLI shape, not a deployed copy-paste command. The
existing owner must call it with `pass_fds=(held_gpu_fd,)`, the full UUID as
`CUDA_VISIBLE_DEVICES`, and `T12_OWNER_HELD_GPU_FD` naming that actual FD.

The paired Slurm wrapper documents this constraint; standalone sbatch cannot
manufacture or inherit an AutoDL canonical GPU owner lease and fails closed.
No science command is advertised as runnable before real storage/source/runtime
bindings are present.

## Focused CPU checks

`/Users/cz0210/miniconda3/envs/smiles_local/bin/python -B -m pytest -q
tests/test_t12_eio_recovery.py tests/baselines/test_tastemolnet_gcf_production_state.py`

Tests cover unchanged old bytes, exact reconstructed observations, DELETE/FULL,
corruption rejection, missing storage proofs,500-only budget, duplicate tail
rejection, real subprocess FD inheritance and independent flock exclusion.
The macOS test stubs only Linux `/proc` start-ticks reading, not FD transfer,
actual parentPID, flock exclusion, or subprocess resource-provider execution.
This is not a real AutoDL GPU or full scientific restore test.

## Deployment state

The two locally affected scientific-source files change only optional storage
policy/plumbing, but their new bytes still require an explicit reviewed source
delta and immutable driver pin. The existing four-file binding is not redefined.
No operator may substitute the CPU test receipt for the actual storage, source,
joint-state, canonical-owner/CAS or resource-provider receipts. No new science
steps have been run by this code change.
