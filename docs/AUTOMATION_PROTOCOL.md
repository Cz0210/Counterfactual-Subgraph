# Automation Protocol

## 1. Architecture

`scripts/ops/experimentctl.py` is a control plane. It validates a task,
records state, runs bounded local gates, prepares safe Git/SSH commands, and
builds Slurm dependency submissions. It does not replace
`scripts/exp_sbatch.sh`: every real Slurm submission still passes through that
existing logging entrypoint.

Long jobs progress inside Slurm:

```text
compute --afterany--> audit --afterok--> downstream compute
                         \
                          --afterany--> final report
```

The audit process converts artifact JSON into an exit code. A local process is
therefore not required to poll for hours.

## 2. Task Spec

Task YAML is checked against `ops/schemas/task_spec.schema.json` and additional
semantic rules. Commands are argv arrays. Paths are resolved before use.
Dependencies must form a DAG. The spec declares Git scope, remote identity,
permissions, stages, gates, expected artifacts, retry limits, and stop points.

The default security posture forbids remote writes, Slurm submission, full
runs, calibration, test, finalization, overwrite, and proxy mutation. Enabling
one permission does not bypass an approval boundary.

## 3. Local Gate

`run-local` runs only `local_command` stages. Each invocation records argv,
cwd, return code, bounded log paths, artifacts, provenance, and a gate JSON.
Successful stages are skipped by `resume`. State is atomically written while
`events.jsonl` and `commands.jsonl` are append-only with `flush` and `fsync`.

The safe self-test is:

```bash
python scripts/ops/experimentctl.py validate-spec ops/specs/example_smoke.yaml
python scripts/ops/experimentctl.py plan ops/specs/example_smoke.yaml
python scripts/ops/experimentctl.py run-local ops/specs/example_smoke.yaml
```

## 4. Git Sync

Only dirty files matching `git.allowed_paths` may be staged. The implementation
uses `git add -- <exact paths>`, compares the staged set with the expected set,
and runs `git diff --cached --check`. Unrelated dirty files are recorded and
left untouched.

Before push, the branch, commit, and origin ancestry are checked. Push uses:

```text
git push origin HEAD:<branch>
```

No automation path uses `git add -A`, `git add .`, stash, clean, reset, or
automatic conflict resolution.

## 5. SSH

SSH supports a normal connection or an existing ControlMaster socket. It uses
port 10022 and `BatchMode=yes`; it never prompts for, stores, or synthesizes a
password. The caller's proxy environment is inherited unchanged. Audit output
records only whether proxy variables exist, never their values.

The read-only preflight checks hostname, repository path, branch, commit,
Python, `sbatch`, `sacct`, and conda activation. Bash initialization runs with
`nounset` temporarily disabled:

```bash
set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
```

Deploy permits only status, fetch, ancestry verification, and `pull --ff-only`.
It never cleans or resets the remote tree.

`deploy` has three distinct modes:

- `--dry-run` creates a persisted run and report but executes neither SSH nor
  remote Git. Its terminal state is `DRY_RUN_COMPLETED`.
- `--preflight-only` executes one read-only SSH command. It records hostname,
  repository path, branch, commit, dirty summary, Python, conda, `sbatch`,
  `sacct`, finalized markers, and proxy-variable presence. It never fetches or
  pulls and does not require remote-write permission.
- deploy without either flag retains the separately permission-gated
  fast-forward synchronization path.

The remote argv contains one `bash -lc` layer. Bash `nounset` remains disabled
while sourcing `.bashrc` and activating conda; it is not re-enabled by the
preflight script. Proxy values are never emitted.

`remote_dirty_policy.allowed_tracked_paths` is an optional, default-empty list
of exact repository-relative POSIX paths. It never applies glob or directory
prefix matching, and it cannot include `scripts/ops`, `tests/ops`,
`ops/specs`, or `ops/schemas`. Allowlisted tracked dirt produces
`REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS`; observed, allowed, and disallowed paths
are persisted in state, report, and read-only preflight evidence.

Root-level untracked files are collected independently with
`git ls-files --others --exclude-standard` between explicit preflight markers;
the controller never infers a file from Git's folded `?? directory/` status.
`remote_dirty_policy.allowed_untracked_paths` is likewise optional,
default-empty, and exact-file-only. It rejects globs, absolute/traversing paths,
directory entries, and protected automation files. Tracked-only,
untracked-only, and combined policy failures use distinct stop reasons.

`remote_dirty_policy` also separates declared patched nested repositories. A
nested repository is accepted only when its parent path is exactly allowlisted,
its unstaged paths are a subset of `allowed_modified_paths`, it has no forbidden
staged or untracked paths, and every required marker is present. Preflight
collects this evidence with nested `status --porcelain=v1`, `diff --name-only`,
and `diff --cached --name-only`; it never cleans, restores, or reapplies a
patch. This is a verified expected-patch state, not an unconditional submodule
exemption.

`proxy_policy` records whether any inherited proxy variable is present without
recording a value. Missing proxy readiness does not prevent a read-only check.
When commits are equal it yields
`REMOTE_PREFLIGHT_PASSED_WITH_WARNINGS`; when Git synchronization is needed it
yields `NEEDS_PROXY_SETUP` and fetch/pull remain prohibited.

A commit mismatch after a successful read-only check produces `NEEDS_DEPLOY`
with `next_action=deploy` when Git-network proxy readiness is available; it
produces `NEEDS_PROXY_SETUP` otherwise. Equal commits normally produce
`REMOTE_PREFLIGHT_PASSED` with
`next_action=remote_write_approval_required`. A missing tool, finalized output,
unexpected remote dirty file, wrong branch, or SSH error produces a bounded
blocked report.

Remote execution and policy evaluation are recorded independently. A zero SSH
return code produces `command_status=PASSED`; a failed dirty or provenance gate
produces `gate_status=BLOCKED` and `REMOTE_PREFLIGHT_BLOCKED`, not a misleading
command failure.

## 6. Slurm Dependencies

`experimentctl submit` builds calls to `scripts/exp_sbatch.sh`. It parses
`[EXP_SUBMIT_OK]` and a numeric `job_id`. Compute failures still trigger audit
through `afterany`; downstream compute uses `afterok` on a successful audit.
Final report jobs use `afterany`.

Use `status` on demand. It may consult the existing experiment registry,
`sync_experiment_status.py`, `sacct`, or `squeue`; it is not a resident poller.

## 7. Gate JSON

The normalized gate schema is `ops/schemas/gate_result.schema.json`. A pass
requires:

- `audit_passed=true`;
- `run_complete=true`;
- an empty `failed_hard_checks`;
- all required artifacts present and nonempty;
- exact required fields and absent forbidden values;
- requested SHA256 matches;
- Slurm `ExitCode=0:0`.

Float comparisons use a declared absolute tolerance. Markers are diagnostic;
they never replace the JSON and artifact checks.

## 8. Approval

Calibration entry, selector freeze, first test evaluation, full GPU jobs,
final-artifact overwrite, and changes to thresholds, metrics, cohorts,
teachers, or labels require an approval event. Approve with:

```bash
python scripts/ops/experimentctl.py approve \
  --run-dir ops/reports/<task>/<run> \
  --stage <stage_id> \
  --reason "reviewed scientific and resource boundary"
```

The event stores UTC time, username, hostname, stage, reason, and the reviewed
Git commit. Submission is blocked if `HEAD` changes after approval.

## 9. Resume

`resume` loads `state.json`, validates the append-only event sequence, and
loads the execution snapshot. It performs at most one action: submit an
approved job or issue one bounded status refresh. It never loops. A stage
already recorded as `PASSED` is skipped, and an already persisted Slurm job ID
is never submitted again.

## 9.1 Adopt Existing Artifacts

`adopt-existing` records a completed legacy stage without claiming that the
current commit reran it. The mode is enabled explicitly by a task-level
`adopt_existing` block. Its capabilities are operation-scoped: the adoption
path is always read-only and cannot inherit remote-write, Slurm, or GPU-smoke
authority that the same spec grants to later stages. Global permissions remain
mandatory for `submit`, `resume`, and `execute-stage`.

`adopt_existing.stages` must form one contiguous dependency segment. Their
only outside prerequisites may be local command gates, and the first stage
after the segment must be a directly dependent approval stage. Adoption marks
only that configured segment and stops at this approval boundary; it does not
reuse `execution.auto_until` or `execution.stop_before` to advance farther.

The verifier first runs the task's bounded local regression gate. It then
uses one non-interactive SSH command to read the completion marker, manifest,
artifact sizes and SHA256 values, scientific summary fields, JSONL row counts,
the remote Git commit, and the finalized marker. Every manifest entry is
checked; omitted or mismatched artifacts block adoption. Artifact aliases map
the current expected path to a legacy path for verification only. They never
create a directory, symlink, copied file, or replacement marker on the HPC.

Successful legacy stages are recorded as `ADOPTED_EXISTING` with
`command_status=NOT_EXECUTED`, `gate_status=PASSED`, and provenance
`source=legacy_manifest_sha256`. The current local/remote commit and the
legacy generation commit remain distinct evidence fields. The run stops at
`STOPPED_BEFORE_APPROVAL`; it does not approve or submit the next stage.

Review the fully side-effect-free command before real verification:

```bash
python scripts/ops/experimentctl.py adopt-existing \
  ops/specs/clear_mutagenicity_phase_a_v2.yaml \
  --dry-run
```

The dry-run executes neither local tests nor SSH. It persists a local plan,
state, command record, and report, and exposes the exact read-only remote
script for review and `bash -n` validation.

## 10. Reports

Terminal runs produce short Markdown and JSON reports. A blocked report names
the failed stage, error class, return code, artifact paths, retry count,
recommended action, semantic-risk flag, and at most 80 trailing stderr lines.
Detailed output remains in the run directory.

## 11. Daily Use

Start with validation and a side-effect-free plan:

```bash
python scripts/ops/experimentctl.py validate-spec ops/specs/example_smoke.yaml
python scripts/ops/experimentctl.py plan ops/specs/example_smoke.yaml
python scripts/ops/experimentctl.py run-local ops/specs/example_smoke.yaml --dry-run
python scripts/ops/experimentctl.py deploy ops/specs/example_smoke.yaml --dry-run
```

After reviewing the dry-run report, a read-only preflight may be run with
`deploy <spec> --preflight-only`. Real deploy and submission require matching
permissions and explicit approval. They are deliberately separate commands.

## 12. CLEAR Phase A And Phase B Smoke

`ops/specs/clear_mutagenicity_phase_a_v2.yaml` records the four strict
train/validation inputs, expected counts, 64-row codec probe, sidecar v2, and
round-trip gates. Its `adopt_existing` block describes how the already-frozen
Phase A was verified without creating compatibility artifacts.

Phase B keeps `phase_b_gpu_smoke` as an explicit approval boundary. After that
approval, `phase_b_gpu_smoke_submit` may submit exactly one 64-parent GPU job
through `scripts/exp_sbatch.sh`. The output root includes the automation run
ID and must not exist before submission. `status --refresh` performs one
read-only `sacct`/`squeue` query. Pending and running jobs remain resumable;
failed Slurm terminal states block immediately. Only `COMPLETED/0:0` invokes
the read-only `phase_b_gpu_smoke_gate`, which validates provenance, counts,
JSON finiteness, logs, and completion markers. A pass stops at
`phase_c_full_run` with `STOPPED_BEFORE_APPROVAL`; full remains disabled.
