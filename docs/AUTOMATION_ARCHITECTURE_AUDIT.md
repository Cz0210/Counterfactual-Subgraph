# Automation Architecture Audit

## Existing Components Reused

- `scripts/exp_sbatch.py` and `scripts/exp_sbatch.sh` remain the only supported
  path for real Slurm submission. They already record job IDs, submission
  arguments, Git state, selected environment metadata, log hints, and expected
  output roots in `outputs/hpc/experiment_registry/jobs.jsonl` and
  `docs/EXPERIMENT_LOG.md`.
- `scripts/sync_experiment_status.py` remains the on-demand Slurm status
  snapshot tool. It queries `sacct` first and falls back to `squeue`.
- The existing experiment registry schema is append-only submission metadata;
  automation state will reference its job IDs instead of replacing it.
- CLEAR Mutagenicity Phase A already supplies deterministic strict train/val
  loading, train-only chemistry vocabularies, the
  `clear_mutagenicity_atom_sidecar_v2` codec, required-category probe sampling,
  and exact round-trip fields suitable for machine-readable gates.

## Missing Capabilities

The repository did not have a task schema, dependency DAG validation, durable
run/stage state, approval events, exact Git staging controls, safe SSH command
construction, remote fast-forward deployment gates, Slurm dependency planning,
JSON scientific gates, resumable orchestration, or bounded final/blocked
reports. Existing tracking records what was submitted, but does not determine
whether a scientific audit passed.

## `experimentctl` and `exp_sbatch`

`experimentctl` owns planning, permissions, state transitions, Git/SSH
preflight, dependency construction, gate evaluation, approvals, resume, and
reports. It does not call `sbatch` directly. Every real submission is delegated
to `scripts/exp_sbatch.sh`, preserving the current registry and experiment log.
`COMPLETED/0:0` is treated as a scheduler prerequisite; scientific success
still requires a passing audit JSON and required artifacts.

## Long-Running Jobs

A local laptop process should not poll a multi-hour Slurm job: network
disconnects, sleep, SSH authentication expiry, and editor restarts would turn
the local process into a single point of failure. Local `status` and `resume`
are therefore on-demand operations. Compute, audit, and finalization stages
that may outlive the local session must progress through Slurm dependencies on
HPC.

## HPC Dependency Boundaries

Compute jobs are followed by `afterany` audit jobs so failures still produce a
diagnostic gate. Downstream compute uses `afterok` on the audit job, making the
audit exit code the machine gate. Final reporting uses `afterany` to summarize
the whole chain. The dependency graph is submitted through `exp_sbatch`; it is
not advanced by a long-running local loop.

## Mandatory Human Approval

Calibration entry, selector freeze, first test evaluation, full GPU
submission, finalized-artifact overwrite, and any change to thresholds,
metrics, cohort/split, teacher/oracle, or source/target labels stop in
`WAITING_APPROVAL` unless the matching permission is explicitly enabled.
Deletion or movement of existing results is never automatic. The current CLEAR
Phase A v2 specification permits validation and planning only.
