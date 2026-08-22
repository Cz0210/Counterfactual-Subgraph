# AutoDL BACE B11--B14 continuation

This procedure creates a new controller. It does not restart or modify the old
v2 controller. Run it only after the status board shows B10 `PASS`, every old
non-Taste task terminal, no active old instance, and the old controller lock
released.

## 1. Build the manifest

Use an immutable execution worktree containing the continuation implementation:

```bash
set -euo pipefail

PROJECT=/root/autodl-tmp/worktrees/run-bace-continuation-<commit>
PY=/root/miniconda3/envs/smiles_pip118/bin/python
DATA=/autodl-fs/data
CONTROL=/autodl-fs/data/counterfactual-subgraph-runtime/control
SOURCE_MANIFEST="$CONTROL/four_gpu_recovery/manifests/autodl-four-gpu-recovery-20260822T044445Z-v2-0ad1494.json"
REPAIR_RUN=bace-molclr-parent-repair-v2-d26fe27-20260822T052100Z
CID="autodl-bace-b11-b14-continuation-$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/run_four_gpu_recovery_controller.py" \
  --project-root "$PROJECT" \
  --data-root "$DATA" \
  --control-root "$CONTROL" \
  --python "$PY" \
  build-bace-continuation \
  --source-manifest "$SOURCE_MANIFEST" \
  --molclr-repair-run-id "$REPAIR_RUN" \
  --controller-id "$CID"

MANIFEST="$CONTROL/four_gpu_recovery/manifests/$CID.json"
```

The builder fails closed unless all 15 adopted runs are exact PASS evidence:
B6, B7, three original passing prep runs, corrected MolCLR prep, eight B8/B9
shards, and B10. It also requires a fresh controller ID, output root, and WNode
cache. The eight historical pool shards become eight non-sharded adopted tasks;
no source task state is rewritten.

## 2. Validate, launch, and inspect

```bash
PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/run_four_gpu_recovery_controller.py" \
  --project-root "$PROJECT" \
  --data-root "$DATA" \
  --control-root "$CONTROL" \
  --python "$PY" \
  validate --manifest "$MANIFEST"

AUTODL_DATA_ROOT="$DATA" \
AUTODL_CONTROL_ROOT="$CONTROL" \
AUTODL_PYTHON="$PY" \
AUTODL_MAX_GPUS=4 \
GLOBAL_MAX_CONCURRENT_GPU_JOBS=4 \
RUN_TASTEMOLNET=0 \
PYTHONDONTWRITEBYTECODE=1 \
"$PROJECT/scripts/autodl/launch_four_gpu_recovery.sh" "$MANIFEST"

PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/status_four_gpu_recovery.py" \
  --project-root "$PROJECT" \
  --data-root "$DATA" \
  --control-root "$CONTROL" \
  --controller-id "$CID" \
  --watch 60
```

During its complete lifetime the continuation holds the old v2 controller
lock, so an attempted v2 restart fails instead of creating duplicate writers.
The generated manifest sets `runtime.keep_alive_when_blocked=true`; after all
non-Taste tasks are terminal it continues only heartbeat/status polling and
does not launch artificial GPU or CPU work.

Do not launch a second continuation if the new controller heartbeat or lock is
live. TasteMolNet remains `BLOCKED_LICENSE_REVIEW`, and `paper/` remains frozen.
