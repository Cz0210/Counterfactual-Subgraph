# AutoDL AIDS/Mutagenicity ComRecGC repair v2

This is a minimal fresh continuation for the two failed AIDS/Mutagenicity
ComRecGC standardized cells. It does not use HPC, modify `paper/`, regenerate
random walks, rerun thresholds, or schedule BACE, GCFExplainer, TasteMolNet,
matrix-audit, figure, table, or final-export work.

## Frozen scope

The controller ID is fixed to:

```text
four_methods_four_datasets_am_repair_v2
```

Its task graph has exactly six tasks:

1. exact runtime re-verification of repair-v1's Mutagenicity recovered-
   generation adoption;
2. exact runtime re-verification of repair-v1's Mutagenicity frozen threshold;
3. exact runtime re-verification of repair-v1's AIDS recovered-generation
   adoption;
4. exact runtime re-verification of repair-v1's AIDS frozen threshold;
5. one fresh Mutagenicity ComRecGC standardized held-out continuation; and
6. one fresh AIDS ComRecGC standardized held-out continuation.

Each source must be the single passing attempt recorded by
`four_methods_four_datasets_repair_v1`, with task state and gate `PASS`, exact
absolute output identity, required physical files, and no writable procfs file
descriptor below it. The generation-adoption JSON must retain the six-member
historical recovery closure and the cross-manifest claimed-SHA agreement. The
large `counterfactuals.pt` is not rehashed by this builder; the scientific
continuation keeps its existing exactly-once payload hash gate. Thresholds are
adopted only when the 601-point WNode/strict-flip contract, theta `0.05`, cap
`0.0535`, dataset identity, and no-test-selection audit all agree.

All scientific dataset, RF, MolCLR, distance-checkpoint, and upstream checkout
paths are copied from the immutable repair-v1 standardization task definitions.
The new spec therefore cannot silently substitute a second set of inputs.

## Required code ancestry

The immutable execution worktree must contain both this AM repair-v2 builder
commit and the reviewed `verify_comrecgc_checkout` Git-safety fix:

```text
d8b113281d24e9340bfe2379e7451ffa8adff70a
```

The spec must contain that exact full SHA. The builder checks it is an ancestor
of execution `HEAD`, and every runtime source-gate task repeats the same check.
This prevents a differently owned migrated ComRecGC checkout from reaching the
GPU path through the previously unsafe Git call.

## Concurrency and isolation

The controller retains a four-GPU ceiling and `max_cpu_tasks=2`. It uses the
same project-global UUID locks under
`$RUNTIME/locks/gpu-<UUID>.lock` as every existing controller. It has no
`continuation` object and inherits no old predecessor guard. All outputs are
fresh below:

```text
$RUNTIME/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/four_methods_four_datasets_am_repair_v2
```

Do not build or launch if that output root, the exact manifest path, or the
controller root already exists.

## Configure, validate, build, and launch

Copy
`configs/autodl/four_by_four_am_repair_v2.template.json` to a fresh persistent
build directory. Replace the five placeholders with the immutable execution
worktree and the four exact repair-v1 PASS attempt roots. Do not point at task
container directories or failed attempts.

Run on AutoDL only:

```bash
set -euo pipefail

PY=/root/miniconda3/envs/smiles_pip118/bin/python
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
CONTROL=$RUNTIME/control
PROJECT=/root/autodl-tmp/worktrees/run-four-by-four-am-repair-v2-<merged-commit>
SPEC=$CONTROL/four_methods_four_datasets_continuation/build-am-repair-v2/repair-spec.json
MANIFEST=$CONTROL/four_methods_four_datasets_continuation/manifests/four_methods_four_datasets_am_repair_v2.json

cd "$PROJECT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

"$PY" scripts/autodl/build_four_by_four_am_repair_manifest.py \
  --config configs/hpc.yaml \
  validate \
  --spec "$SPEC"

"$PY" scripts/autodl/build_four_by_four_am_repair_manifest.py \
  --config configs/hpc.yaml \
  build \
  --spec "$SPEC" \
  --output "$MANIFEST"

"$PY" scripts/autodl/run_four_by_four_controller.py \
  --project-root "$PROJECT" \
  --data-root /autodl-fs/data \
  --control-root "$CONTROL" \
  --python "$PY" \
  validate \
  --manifest "$MANIFEST"

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

Status:

```bash
PYTHONPATH=$PROJECT "$PY" scripts/autodl/status_four_by_four.py \
  --project-root "$PROJECT" \
  --data-root /autodl-fs/data \
  --control-root "$CONTROL" \
  --controller-id four_methods_four_datasets_am_repair_v2 \
  --format table
```

Only restart after confirming the recorded controller PID is absent and its
heartbeat is stale. The persistent controller resumes PASS tasks from its own
state; do not delete or rewrite attempts. The paired Slurm wrapper exists only
for CLI parity and must not be submitted for this AutoDL-only repair.
