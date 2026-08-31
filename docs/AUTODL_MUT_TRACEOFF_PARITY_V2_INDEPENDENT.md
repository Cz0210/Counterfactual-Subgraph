# Mutagenicity independent trace-off parity v2

## Scope

This is the minimal successor to the deployed AIDS-dependent Mut trace-off
parity v1 controller.  It removes only the artificial dependency on the AIDS
paper cell.  It does not weaken, replace, or bypass the independent Mut
trace-on/trace-off parity requirement.

The v2 controller uses a new identity and a fresh root:

```text
controller_id=four_methods_four_datasets_mut_traceoff_parity_v2_independent
fresh_output_root=.../repairs/four_methods_four_datasets_mut_traceoff_parity_v2_independent
```

The old v1 manifest, controller state, and output root remain immutable.  A v2
spec containing `aids_dependency` is rejected rather than ignored.

## Frozen science

The seven-task route keeps all scientific inputs from v1:

1. revalidate the frozen lineage-v3 trace-on generation;
2. revalidate the frozen Mut threshold source;
3. compare the exact 500-step legacy `7f7ed51` prefix with the checkpointed
   `66487c0` prefix on an exclusive GPU;
4. execute the fresh seed-0, trace-disabled 50k reference from `66487c0` with
   completed-step checkpoints and an independent mirror;
5. run the real normalized trace parity assertion against the lineage-v3
   source;
6. adopt the completed repair-v2 common recourse read-only only after parity;
7. run the CPU-only chemistry and held-out standardization continuation.

The generator still uses 1,448 parents, batch size 128, candidate capacity
100,000, five heads, sample size 10,000, teleport 0.1, theta 0.1, and seed 0.
Generation loads neither calibration nor test.  A copied payload, self
comparison, synthesized receipt, or trace-stripped payload cannot pass.

## Spec preparation

Copy
`configs/autodl/mut_traceoff_parity_v2_independent.template.json` to the fresh
control spec path and replace only the three immutable-worktree placeholders:

```text
__IMMUTABLE_EXECUTION_WORKTREE__
__IMMUTABLE_7F7ED51_SCIENCE_WORKTREE__
__IMMUTABLE_66487C0_CHECKPOINT_WORKTREE__
```

The controller worktree must be clean and fixed at the reviewed deployment
commit.  The scientific worktrees must resolve exactly to:

```text
7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4
66487c062c86d53ef2f762ce04d0fb965af5af08
```

Do not copy the old v1 manifest and delete a task by hand.  The builder must
create the new manifest so the fresh output paths and scientific-command hash
are recomputed together.

## Resource preflight

Both the 500-step equivalence task and the 50k reference require one exclusive
GPU.  They must not share a GPU with another process.  The template preserves
the v1 value:

```text
min_cgroup_free_bytes=472446402560  # 440 GiB
```

Before launching, verify that
`memory.limit_in_bytes - memory.usage_in_bytes` is at least that value, the
shared high-memory lock has no owner, no common-recourse process is active,
and one physical GPU is genuinely idle with its UUID lock available.  Do not
launch early and rely on retry: insufficient cgroup headroom is a fail-closed
execution error in the stage wrapper.

## Build and launch

On AutoDL, after the preflight passes:

```bash
set -euo pipefail

PY=/root/miniconda3/envs/smiles_pip118/bin/python3.10
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
CONTROL=$RUNTIME/control
PROJECT=/root/autodl-tmp/worktrees/run-mut-traceoff-independent-<commit>
SPEC=$CONTROL/four_methods_four_datasets_continuation/specs/four_methods_four_datasets_mut_traceoff_parity_v2_independent.json
MANIFEST=$CONTROL/four_methods_four_datasets_continuation/manifests/four_methods_four_datasets_mut_traceoff_parity_v2_independent.json

cd "$PROJECT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

"$PY" scripts/autodl/manage_mut_traceoff_parity_v1.py \
  --config configs/hpc.yaml validate --spec "$SPEC"

"$PY" scripts/autodl/manage_mut_traceoff_parity_v1.py \
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

This repository change does not execute that launch command.
