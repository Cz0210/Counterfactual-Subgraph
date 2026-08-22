# AutoDL AIDS ComRecGC CPU/high-memory repair v3

This is a bounded fresh-root retry for the AIDS ComRecGC standardized cell.
It does not use HPC, modify `paper/`, rerun generation, alter thresholds, or
weaken the independent Mutagenicity trace-parity gate.

## Why repair-v2 failed

The repair-v2 AIDS scientific child exited by `SIGKILL` while executing
`run_common_recourse.py`. On the same AutoDL container, cgroup-v1 recorded:

```text
memory.limit_in_bytes     = 515396075520 (480 GiB)
memory.max_usage_in_bytes = 515396108288
memory.failcnt            = 1400
memory.oom_control        = oom_kill 1
```

At that time the AIDS and Mutagenicity full common-recourse computations ran
concurrently. This is a host-memory scheduling failure, not a scientific
failure. The failed attempt and all old roots remain immutable.

The repair-v2 controller's periodically reconciled task instance may omit its
denormalized `exit_code`. The authoritative terminal representation is the
matching immutable `experiment_registry/run_state/<run_id>/state.json`, which
records `state=FAILED` and `exit_code=1` for the wrapper. Repair v3 accepts a
controller-instance exit code only when it is absent or exactly `1`, and still
requires all of the following together: the exact FAILED controller gate and
run ID, exp-run exit `1`, the attempt's `CalledProcessError` naming child
`SIGKILL: 9`, and the cgroup limit/peak/fail/OOM counters. No arbitrary
execution failure satisfies this gate.

Mutagenicity failed for a separate scientific reason. Its 100-rule
common-recourse stage completed, but its chemistry preregistration requires a
real trace-on/trace-off parity artifact. The frozen lineage-v3 root contains
complete streamed trace-integrity evidence, not `trace_parity.json` with
`trace_parity_passed=true`. A parity result cannot be derived from the traced
payload alone without becoming a tautological self-comparison. Repair v3
therefore contains no Mutagenicity scientific task.

The read-only inventory was taken from:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z
```

That root has one traced `counterfactuals.pt`, the authoritative graph store,
`trace/trace_summary.json`, `_TRACE_COMPLETE.json`, the lineage index, and all
selected trace chunks. It has neither a second trace-disabled
`counterfactuals.pt` nor `trace_parity.json`. Stripping trace fields from the
same payload would not establish that trace instrumentation left generation
unchanged.

## Fixed resource contract

Controller ID:

```text
four_methods_four_datasets_aids_comrecgc_repair_v3
```

The exact task graph has three CPU tasks:

1. revalidate repair-v2's AIDS generation source gate;
2. revalidate repair-v2's AIDS threshold source gate; and
3. run one fresh AIDS standardized continuation.

The scientific task has:

```text
resource=cpu
gpu_required=false
DEVICE=cpu
CUDA_VISIBLE_DEVICES=""
runtime.max_cpu_tasks=1
```

Before the output root is created, the wrapper:

- holds the persistent exclusive lock
  `$RUNTIME/locks/comrecgc_common_recourse_highmem.lock` for its full lifetime;
- pins the lock implementation to AutoDL's `/usr/bin/flock`;
- verifies cgroup-v1 limit and usage counters;
- requires at least the configured headroom (the template fixes 440 GiB);
- rejects an already-running legacy `run_common_recourse.py`; and
- confirms the scientific output root is fresh.

This prevents another cooperating full common-recourse task from sharing host
memory and fails closed if a legacy non-locking process is already active.

## Build and launch on AutoDL only

Copy
`configs/autodl/aids_comrecgc_repair_v3.template.json` to a fresh persistent
build directory. Replace only the immutable execution worktree placeholder.
The two repair-v2 AIDS source-gate PASS roots are pinned to their verified
attempt-0 paths. Do not point at the old failed scientific output as a source
gate.

```bash
set -euo pipefail

PY=/root/miniconda3/envs/smiles_pip118/bin/python3.10
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
CONTROL=$RUNTIME/control
PROJECT=/root/autodl-tmp/worktrees/run-aids-comrecgc-repair-v3-<commit>
SPEC=$CONTROL/four_methods_four_datasets_continuation/build-aids-repair-v3/spec.json
MANIFEST=$CONTROL/four_methods_four_datasets_continuation/manifests/four_methods_four_datasets_aids_comrecgc_repair_v3.json

cd "$PROJECT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

"$PY" scripts/autodl/build_aids_comrecgc_repair_v3_manifest.py \
  --config configs/hpc.yaml validate --spec "$SPEC"

"$PY" scripts/autodl/build_aids_comrecgc_repair_v3_manifest.py \
  --config configs/hpc.yaml build --spec "$SPEC" --output "$MANIFEST"

"$PY" scripts/autodl/run_four_by_four_controller.py \
  --project-root "$PROJECT" \
  --data-root /autodl-fs/data \
  --control-root "$CONTROL" \
  --python "$PY" \
  validate --manifest "$MANIFEST"

AUTODL_DATA_ROOT=/autodl-fs/data \
AUTODL_RUNTIME_ROOT="$RUNTIME" \
AUTODL_CONTROL_ROOT="$CONTROL" \
AUTODL_PYTHON="$PY" \
AUTODL_MAX_GPUS=4 \
GLOBAL_MAX_CONCURRENT_GPU_JOBS=4 \
GPU_IDLE_STABLE_SECONDS=60 \
SCHEDULER_POLL_SECONDS=60 \
RUN_TASTEMOLNET=0 \
scripts/autodl/launch_four_by_four.sh "$MANIFEST"
```

Do not submit the paired Slurm files. They exist only for repository CLI
parity. The controller and output root must be new; repair-v2 remains
unchanged.

## Mutagenicity release condition

A safe no-generation-rerun release requires an independently frozen,
trace-disabled full-budget `counterfactuals.pt` produced with the exact same
dataset, classifier, seed, 50,000-step configuration, upstream/project commit,
and candidate budget. It can then be compared against lineage-v3 using the
existing `assert_trace_parity` contract in a fresh CPU preflight root. No such
reference or existing parity artifact is present in the audited AutoDL payload.
Without it, the only alternatives are a formal generation rerun or a new
explicit scientific decision changing the Mutagenicity evidence contract.
