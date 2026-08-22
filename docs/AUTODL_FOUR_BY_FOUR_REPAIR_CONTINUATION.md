# AutoDL four-by-four bounded repair continuation

This continuation repairs only four failed closures from
`four_methods_four_datasets_continuation_v1`. It does not rerun a passing
B0--B14 scientific stage, does not include TasteMolNet, does not build the final
matrix/figures, and does not write under `paper/`.

## Scope

The frozen task graph contains exactly these routes:

1. BACE ComRecGC: the existing generic native-GNN fragment from train
   generation through calibration selector, post-freeze test, final freeze,
   and artifact-only standardization.
2. Mutagenicity GCFExplainer: exact adoption of the passing v1 candidate
   freeze, followed by fresh calibration, held-out evaluation, and
   artifact-only standardization.
3. Mutagenicity and AIDS ComRecGC: fresh threshold-contract verification and
   fresh standardized continuation from the immutable recovered generation
   roots. Random-walk generation is not repeated.
4. BACE Ours: artifact-only standardization of the exact passing v1 B14 root.

Every adopted source is checked twice: once before the manifest is written and
again as a controller task. A controller source must have task state `PASS`,
gate `PASS`, one unambiguous passing attempt equal to the configured absolute
root, required physical files, and no live writable file descriptor below that
root. Historical recovered COMRECGC roots have no bare `PASS` file. They are
accepted only when the five existing recovery manifests close consistently,
the physical `counterfactuals.pt` stat agrees with their byte claim, all
recorded payload SHA-256 claims agree, and the same writer audit passes. The
builder neither requires the mutable experiment registry nor rehashes the
large tensor; the scientific continuation retains its existing exactly-once
payload SHA-256 gate.

## Concurrency contract

The controller ID is fixed to:

```text
four_methods_four_datasets_repair_v1
```

It uses the same runtime layout as v1, including the project-global
`<runtime_root>/locks/gpu-<UUID>.lock` files. It can therefore run while v1 is
still alive: a card held by v1 is unavailable to the repair controller, and a
repair task can use only another card that passes the stable-idle and UUID-lock
checks. The repair manifest fixes `runtime.max_cpu_tasks=2` because both
controllers share the host.

The repair manifest has no `continuation` object. It does not acquire or retain
the old BACE recovery/v2 predecessor guard; coordination with v1 is only via
the shared GPU UUID locks and read-only source audits.

## Required execution fixes

Build from one immutable execution commit containing all of:

- the AutoDL-pinned Python branch in
  `scripts/autodl/run_mut_gcf_legacy_evaluation.sh`;
- the COMRECGC Git safe-directory/preflight fix used by native BACE
  generation; and
- the BACE Ours standardizer fix for the exact historical B13/B14 artifact
  layout.

Put the full commit IDs in `required_execution_commits`. The builder checks
that each is an ancestor of the execution worktree HEAD. It does not silently
weaken this into a filename or branch-name check.

## Configure the exact spec

Copy
`configs/autodl/four_by_four_repair_v1.template.json` to a fresh persistent
control build directory and replace every `__CONFIGURE_...` value. In
particular, use the actual passing attempt roots from v1 for `bace_b14` and
`mut_gcf_freeze`; do not point at their task container directories. Keep the
two failed v1 attempts and all historical PASS roots untouched.

The new `fresh_output_root` must not exist and must be below
`$RUNTIME/outputs/autodl`. A recommended root is:

```text
$RUNTIME/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/four_methods_four_datasets_repair_v1
```

## Exact validate, build, and launch commands

Run on AutoDL only, from the immutable merged execution worktree:

```bash
set -euo pipefail

PY=/root/miniconda3/envs/smiles_pip118/bin/python
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
CONTROL=$RUNTIME/control
PROJECT=/root/autodl-tmp/worktrees/run-four-by-four-repair-<merged-commit>
SPEC=$CONTROL/four_methods_four_datasets_continuation/build-repair-v1/repair-spec.json
MANIFEST=$CONTROL/four_methods_four_datasets_continuation/manifests/four_methods_four_datasets_repair_v1.json

cd "$PROJECT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

"$PY" scripts/autodl/build_four_by_four_repair_manifest.py \
  --config configs/hpc.yaml \
  validate \
  --spec "$SPEC"

"$PY" scripts/autodl/build_four_by_four_repair_manifest.py \
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

Do not launch if the repair PID/heartbeat already exists. The launcher uses
tmux when available and otherwise nohup. It writes under:

```text
$CONTROL/four_methods_four_datasets_continuation/four_methods_four_datasets_repair_v1
```

## Status and safe restart

```bash
cd "$PROJECT"
PYTHONPATH=$PWD "$PY" scripts/autodl/status_four_by_four.py \
  --project-root "$PROJECT" \
  --data-root /autodl-fs/data \
  --control-root "$CONTROL" \
  --controller-id four_methods_four_datasets_repair_v1 \
  --format table
```

Only after the recorded PID is absent and the heartbeat is stale may the same
launch command be used to resume. The persistent controller reloads its task
states and passing attempt outputs; it does not rerun a PASS task. Never create
a second manifest at the same path or delete a failed attempt to make a retry
look fresh.

The paired `scripts/slurm/build_four_by_four_repair_manifest.sh` exists solely
for repository CLI parity. This campaign must not submit it and must not use
HPC.
