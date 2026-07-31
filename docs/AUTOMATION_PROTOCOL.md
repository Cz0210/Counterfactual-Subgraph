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

A commit mismatch after a successful read-only check produces `NEEDS_DEPLOY`
with `next_action=deploy`. Equal commits produce
`REMOTE_PREFLIGHT_PASSED` with
`next_action=remote_write_approval_required`. A missing tool, finalized output,
unexpected remote dirty file, wrong branch, or SSH error produces a bounded
blocked report.

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

The event stores UTC time, username, hostname, stage, and reason.

## 9. Resume

`resume` loads `state.json`, validates the append-only event sequence, and
loads `spec.snapshot.yaml`. A stage already recorded as `PASSED` is skipped.
Retries increment the stage attempt without deleting earlier stdout, stderr,
commands, or events.

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

## 12. CLEAR Phase A

`ops/specs/clear_mutagenicity_phase_a_v2.yaml` records the four strict
train/validation inputs, expected counts, 64-row codec probe, sidecar v2, and
round-trip gates. It stops before `phase_b_gpu_smoke`. Its remote-write and
Slurm permissions are false, so the current artifact is suitable only for
validation and planning until a reviewed task spec enables the next action.
