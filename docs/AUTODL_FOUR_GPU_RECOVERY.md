# AutoDL four-GPU recovery controller

This controller is an AutoDL control plane, not a scientific implementation.
It schedules foreground scientific argv declared in one frozen JSON/YAML
manifest and delegates every detached worker to `scripts/autodl/exp_run.py`.
The canonical `exp_run` registry, UUID GPU locks, stage state, manifests, gates,
tmux/nohup fallback, and result-contract checks therefore remain authoritative.

It never calls `sbatch`, never kills an existing process, never edits `paper/`,
and always reports TasteMolNet as
`TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW`.

## Persistent layout

Set both roots to persistent AutoDL storage. The controller rejects a relative
control root, a control root outside the selected data root, and a control root
inside the code worktree.

```text
$AUTODL_DATA_ROOT/counterfactual-subgraph-runtime/
├── control/four_gpu_recovery/<controller_id>/
│   ├── controller_manifest.json
│   ├── controller_state.json
│   ├── heartbeat.json
│   ├── controller.log
│   ├── registry/events.jsonl
│   └── tasks/<task_id>/{manifest,state,gate}.json
├── locks/gpu-<physical-uuid>.lock
├── logs/four_gpu_recovery/<controller_id>/
├── outputs/autodl/experiment_registry/
│   ├── runs.jsonl
│   └── status_updates.jsonl
└── docs/AUTODL_FOUR_GPU_EXPERIMENT_LOG.md
```

The per-task manifest is immutable, state and gate documents are atomically
replaced, and the controller registry is append-only. `exp_run` also records
each child in its existing canonical control registry. The two user-facing
JSONL mirrors and Markdown log above are append-only and carry flat
`input_root`, `output_root`, `checkpoint_hash`, `config_hash`, `retry_count`,
and `dependency_ids` fields in addition to the detailed nested provenance.
Checkpoint identity is copied from the frozen launch spec/environment; the
controller never re-hashes a large checkpoint on each status tick.

## Manifest contract

Start from
`configs/autodl/four_gpu_recovery.template.json`, copy it to persistent storage,
and fill only the integration placeholders after Commit A/B/D entrypoints are
available. The committed template is intentionally BLOCKED and cannot launch
placeholder science. After every placeholder and exact launch contract is
filled, clear that task's `blocked_reason` to `null`; leaving it set is an
intentional terminal block.

Every runnable task must declare:

- an argv-array `command` (never a shell string);
- `depends_on`, `dataset`, `stage`, `resource`, and `data_splits`;
- an absolute persistent `input_manifest` after token expansion;
- a fresh output under `{artifact_root}`, required files, and a PASS log marker;
- any prescribed audit file outside the task output via
  `required_absolute_output_files`, still restricted to persistent outputs;
- explicit config files and non-secret environment values when needed.

If a MUT/AIDS task was already launched through `exp_run`, set its exact
`adopt_existing_run_id`, GPU index/full UUID, project root, full Git SHA,
`max_gpus`, and `heavy` values before the controller's first start. Adoption verifies
the existing launch spec's dataset/stage/argv/interpreter/environment, input
path and SHA, fresh output path, required files, marker, and CPU/GPU binding.
A matching RUNNING worker is monitored, a PASS is re-gated, and a
FAILED/BLOCKED run is terminal. A mismatch fails closed; the task is never put
back in the launch queue, so adoption cannot create a second writer. The
adopted immutable Commit-A worktree may differ from the controller worktree;
`{project_root}` is expanded against `adopt_project_root` for that task and is
compared exactly with the existing launch spec.
For a GPU-reserved adoption, also freeze `adopt_gpu_index` and
`adopt_gpu_uuid`; both must match the launch spec exactly. This applies even if
the adopted scientific wrapper clears `CUDA_VISIBLE_DEVICES` and performs a
CPU-only recovery: its live `exp_run` UUID/project-slot reservation still
removes that card from controller capacity (for example, GPU0/1 leave only
GPU2/3 eligible).

Supported path tokens are `{project_root}`, `{data_root}`, `{runtime_root}`,
`{artifact_root}`, `{control_root}`, `{python}`, `{task_id}`, `{stage}`,
`{instance_id}`, `{shard_id}`, `{shard_manifest}`, `{attempt}`, and
`{batch_size}`. Python commands must use `{python}` so non-interactive SSH never
falls back to the base environment.
For each dependency `<task_id>`, the controller also exposes
`{dep_<normalized_task_id>_output}`. Consumers of an OOM-retryable task must use
that token instead of an `attempt-0` path, because the only passing immutable
bundle may be `attempt-1`. Sharded dependency tokens resolve to their frozen
aggregate task root.
The launch shell and every newly created child spec also freeze
`PYTHONDONTWRITEBYTECODE=1`, preventing imports from mutating an immutable
execution clone with `__pycache__` files.

MUT/AIDS Commit A commands and BACE Commit B commands are injected through
this manifest. The controller does not import their scientific modules. A
wrapper is safe only if it remains a foreground payload; do not configure a
wrapper that starts a second detached `exp_run` process.

The BACE control dependencies are:

```text
B5 PASS (external)
  -> policy provenance audit
  -> clean initializer
  -> adapter canary (diagnostic only; cannot release B7)
  -> B6_PPO_SMOKE_V2 (fresh formal run; does not overwrite legacy B6)
  -> B7_PPO_FULL
  -> B8_POOL_BASE (four fixed train-parent shards) ┐
  -> B9_POOL_HIGHTEMP (four fixed train-parent shards) ┘
  -> B10_POOL_MERGED
  -> B11_CROSS_PARENT_VERIFIED (four calibration-parent shards)
  -> B12_SELECTOR (freezes selector)
  -> B13_FINAL_EVAL (four fixed one-shot read-only test shards)
  -> B14_FROZEN (manifest-only; no raw calibration/test load)
```

Both B8 and B9 depend directly on B7; B10 is their join. MUT and AIDS remain independent roots, so any ready lane starts without
waiting for the other lines. A FAILED, BLOCKED, or SKIPPED dependency blocks
its descendants. Independent ready work continues.

Four Commit-D prep slots (calibration GNN-before, MolCLR embeddings, frozen
shard manifests, and output preflight) depend on B6 and may run beside B7.
They are not B7 dependencies and cannot inspect held-out test bytes or an
unfinished PPO candidate pool.

The provenance-audit task binds its externally prescribed timestamped CSV via
`BACE_POLICY_AUDIT_CSV=$RUNTIME/outputs/autodl/audits/...csv` and lists that
same path as required absolute evidence. The initializer consumes the small
`BACE_POLICY_AUDIT_SELECTION` artifact, avoiding repeated hashing of the large
base model. B6/B7 bounded OOM retries use Commit B's exact
`BACE_PPO_BATCH_SIZE` variable.

## Resource and recovery policy

- The controller has an explicit four-GPU ceiling; the existing frozen-GNN
  runners keep their conservative two-GPU default.
- A GPU is eligible only if all samples across at least 60 seconds show no
  compute process, enough free memory, and utilization below the threshold.
- `exp_run` takes the project slot and physical UUID lock. The controller audits
  lock metadata against worker PID generation and GPU process rows, but never
  deletes a lock or signals a process.
- CPU load, available RAM, and persistent-disk free space gate every launch.
- A detected CUDA OOM may receive exactly one manifest-declared lower-batch
  retry. The retry must use an `{attempt}`-qualified fresh output. A second OOM
  fails closed. Semantic/provenance/test-leakage failures never retry.
- Sharded tasks consume one frozen parent-ID manifest. The controller creates
  deterministic, disjoint, exhaustive shard manifests once and refuses changed
  bytes on resume. Shards may not read held-out test except for the gated B13
  case below.
- Only B13 may declare test access, and only through a transitive passing B12
  task with `freezes_selector=true`, `selector_parameters_frozen=true`, and
  `read_only_test=true`. Test-looking paths are scanned in argv, config files,
  environment, input manifests, and parent-shard manifests. Only a conforming
  B13 may materialize test-parent shards.

## AutoDL commands

Prepare a persistent manifest and validate it before launch:

```bash
export AUTODL_DATA_ROOT=/autodl-fs/data
export AUTODL_CONTROL_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
export PYTHONDONTWRITEBYTECODE=1
export FOUR_GPU_RECOVERY_MANIFEST="$AUTODL_CONTROL_ROOT/four_gpu_recovery_manifest.json"

cp configs/autodl/four_gpu_recovery.template.json "$FOUR_GPU_RECOVERY_MANIFEST"
# Fill Commit A/B/D argv and immutable input/output contracts, then:
"$AUTODL_PYTHON" scripts/autodl/run_four_gpu_recovery_controller.py \
  --project-root "$PWD" \
  --data-root "$AUTODL_DATA_ROOT" \
  --control-root "$AUTODL_CONTROL_ROOT" \
  --python "$AUTODL_PYTHON" \
  validate --manifest "$FOUR_GPU_RECOVERY_MANIFEST"
```

Launch once. If tmux is unavailable, the shell automatically uses nohup and
writes the controller log under the persistent control directory.

```bash
bash scripts/autodl/launch_four_gpu_recovery.sh "$FOUR_GPU_RECOVERY_MANIFEST"
```

Read or watch status without changing scheduler state:

```bash
"$AUTODL_PYTHON" scripts/autodl/status_four_gpu_recovery.py \
  --project-root "$PWD" \
  --data-root "$AUTODL_DATA_ROOT" \
  --control-root "$AUTODL_CONTROL_ROOT" \
  --manifest "$FOUR_GPU_RECOVERY_MANIFEST"

"$AUTODL_PYTHON" scripts/autodl/status_four_gpu_recovery.py \
  --project-root "$PWD" \
  --data-root "$AUTODL_DATA_ROOT" \
  --control-root "$AUTODL_CONTROL_ROOT" \
  --manifest "$FOUR_GPU_RECOVERY_MANIFEST" \
  --watch 60
```

The JSON view preserves full output paths and reports the next priority-ready
task, GPU used/free/total memory, worker/child PID, tmux session, elapsed time,
and last heartbeat. The table truncates only long output paths for readability.

The paired Slurm wrapper for the controller performs validation only. It does
not launch AutoDL work from HPC.
