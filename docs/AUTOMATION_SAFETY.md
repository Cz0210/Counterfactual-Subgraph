# Automation Safety

## Default Deny

Task permissions default to no remote write, no Slurm, no GPU smoke, no full
run, no calibration, no test, no finalization, and no overwrite. The schema
and semantic validator reject forbidden split paths and unsafe remote roots
before execution.

## Credentials And Proxies

The controller never records passwords or tokens and uses non-interactive SSH.
It does not edit SSH configuration, `.bashrc`, sockets, or forwarding rules.
Proxy variables are inherited without modification. Reports expose only
presence booleans, not proxy values.

`deploy --preflight-only` is explicitly read-only. Its remote script may run
`hostname`, `pwd`, directory tests, Git status/branch/HEAD reads, Python and
command availability checks, conda activation, finalized-marker tests, and
proxy-presence checks. It cannot run fetch, pull, merge, push, sbatch, file
creation, deletion, or movement. A commit mismatch is reported as
`NEEDS_DEPLOY`; it never triggers synchronization.

## Worktree Safety

Only paths in the task allowlist may be staged. Other local changes, dirty
submodules, experiment logs, and generated outputs are not cleaned, stashed,
reset, or overwritten. A staged path outside the allowlist blocks the run.
Finalized output roots are immutable unless overwrite is explicitly enabled
and separately approved.

## Scientific Boundaries

Automation may run existing commands, verify declared inputs and artifacts,
and apply preregistered gates. It may not silently change a cohort, split,
source/target label, teacher, threshold, metric, candidate order, selector, or
evaluation definition. Those are scientific changes and require code review,
a new task specification, and an approval event.

An engineering failure includes a missing file, nonzero command exit, SSH
failure, dirty remote file, invalid state, or malformed audit JSON. A
scientific failure includes a valid run whose audit reports a failed chemistry,
coverage, provenance, or protocol condition. Both block downstream work, but
only the latter may justify changing the experimental protocol.

## Slurm And Reports

Slurm `COMPLETED` is necessary but insufficient. The audit JSON and required
artifacts determine scientific success. Audit jobs run after any compute
outcome and communicate the gate through their exit code. Downstream compute
depends on a successful audit.

Codex may build and locally test this control plane and may perform an
explicitly allowed read-only preflight. It must not submit full, calibration,
or test work, delete results, overwrite finalized artifacts, or push changes
without the task and user granting that action.
