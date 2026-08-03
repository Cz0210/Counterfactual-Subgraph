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

The proxy gate is readiness metadata, not proxy configuration. An equal-commit
read-only preflight may pass with warnings when no proxy is present. A commit
mismatch with no required Git-network proxy stops at `NEEDS_PROXY_SETUP`; the
controller does not create or alter SSH forwarding, port 39393, `.bashrc`, or
SSH configuration.

`deploy --preflight-only` is explicitly read-only. Its remote script may run
`hostname`, `pwd`, directory tests, Git status/branch/HEAD reads, Python and
command availability checks, conda activation, finalized-marker tests, and
proxy-presence checks. It cannot run fetch, pull, merge, push, sbatch, file
creation, deletion, or movement. A commit mismatch is reported as
`NEEDS_DEPLOY`; it never triggers synchronization.

`adopt-existing` has a narrower, manifest-oriented read-only contract. It may
read JSON/JSONL files, file sizes, SHA256 digests, and Git HEAD through SSH.
It cannot create compatibility aliases, current-format marker files, or any
remote report. All evidence and reports are written only below local
`ops/reports`. A configured alias is a verification lookup, never a filesystem
operation. Remote bytecode generation is disabled so verification does not
create `__pycache__`. The command is rejected unless every remote-write, Slurm, GPU,
full, calibration, test, finalization, and overwrite permission is false.

Legacy adoption does not rewrite provenance. The current local and remote
commits must match each other, while the legacy generation commit must match
both the completion marker and manifest. Adopted stages are marked as not
executed under the current commit, and execution stops before the next
approval stage.

## Worktree Safety

Only paths in the task allowlist may be staged. Other local changes, dirty
submodules, experiment logs, and generated outputs are not cleaned, stashed,
reset, or overwritten. A staged path outside the allowlist blocks the run.
Finalized output roots are immutable unless overwrite is explicitly enabled
and separately approved.

Remote patched repositories are not generally allowlisted. Each permitted
nested repository declares an exact top-level tracked-dirty allowlist entry,
exact modified paths, and required markers.
Unexpected modified paths, any forbidden staged or untracked path, or a missing
marker blocks preflight while preserving the nested working tree exactly as it
was found.

Remote tracked-dirty exemptions are exact paths, not patterns or directory
prefixes. They default to empty and cannot include the automation control paths
`scripts/ops`, `tests/ops`, `ops/specs`, or `ops/schemas`. An exemption changes
a successful read-only preflight into a warning; it never grants a remote
write. The same protection applies to exact root-level untracked-file
allowlists. Untracked directories are never allowlisted by prefix; the remote
read-only scan enumerates individual files before policy evaluation.

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
