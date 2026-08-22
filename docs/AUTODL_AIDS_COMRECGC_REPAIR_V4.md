# AutoDL AIDS ComRecGC repair-v4 runbook

This route replaces only the failed AIDS ComRecGC standardization attempt with
an exact, disk-backed common-recourse engine.  It is CPU-only and AutoDL-only.
Repair-v2 and repair-v3 outputs are immutable inputs/evidence; BACE,
TasteMolNet, `paper/`, and HPC execution are out of scope.

## Release gates

Do not launch the full controller until all of the following are true:

1. the immutable execution checkout contains
   `d5c1d67339df4b9642beaf2b10908ed92bac30de` and the builder commit;
2. the focused test suite and shell/compile checks pass in that checkout;
3. a fresh diagnostic-only AutoDL legacy-versus-external smoke proves identical
   pair order, elementwise sklearn labels, official coverage result, selected
   rows, and selected-row hash;
4. the builder revalidates the exact repair-v2 source gates and the reviewed
   repair-v3 cgroup OOM evidence;
5. the destination manifest, controller directory, and repair-v4 output root
   do not exist.

The smoke uses a fresh derived diagnostic payload/root and is marked
`eligible_for_main_results=false`.  It cannot be adopted as the full cell.

## Build the frozen manifest

Copy the template to a fresh persistent configuration file and replace only
`__IMMUTABLE_EXECUTION_WORKTREE__` with the absolute immutable checkout path
and `__FRESH_EQUIVALENCE_GATE_JSON__` with the exact successful real-data
diagnostic gate.  Do not edit the scientific paths or resource values.  The
builder rehashes the gate, its source identity, and its evidence artifacts;
the path alone is insufficient.

```bash
export AUTODL_DATA_ROOT=/autodl-fs/data
export AUTODL_RUNTIME_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime
export AUTODL_CONTROL_ROOT="$AUTODL_RUNTIME_ROOT/control"
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
export PYTHONPATH="$PWD"

SPEC="$AUTODL_CONTROL_ROOT/four_methods_four_datasets_continuation/specs/four_methods_four_datasets_aids_comrecgc_repair_v4.json"
MANIFEST="$AUTODL_CONTROL_ROOT/four_methods_four_datasets_continuation/manifests/four_methods_four_datasets_aids_comrecgc_repair_v4.json"

"$AUTODL_PYTHON" scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py \
  --config configs/hpc.yaml validate --spec "$SPEC"
"$AUTODL_PYTHON" scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py \
  --config configs/hpc.yaml build --spec "$SPEC" --output "$MANIFEST"
```

The controller has exactly these tasks:

- `am_v4_source_aids_comrec_generation`;
- `am_v4_source_aids_comrec_threshold`;
- `aids_comrecgc_standardized_external_memory`.

The scientific task command is exactly:

```text
bash {project_root}/scripts/autodl/run_aids_comrecgc_repair_v4_supervisor.sh
```

Its frozen environment uses `GPU_REQUIRED=0`, `DEVICE=cpu`, 128 GiB minimum
cgroup headroom, a 96 GiB hard external RSS budget, query block size eight,
sklearn 1.7.2, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and exactly one
same-root process-loss retry.

## Launch and inspect

```bash
RUN_TASTEMOLNET=0 \
AUTODL_DATA_ROOT="$AUTODL_DATA_ROOT" \
AUTODL_RUNTIME_ROOT="$AUTODL_RUNTIME_ROOT" \
AUTODL_CONTROL_ROOT="$AUTODL_CONTROL_ROOT" \
AUTODL_PYTHON="$AUTODL_PYTHON" \
scripts/autodl/launch_four_by_four.sh "$MANIFEST"

"$AUTODL_PYTHON" scripts/autodl/status_four_by_four.py \
  --project-root "$PWD" \
  --data-root "$AUTODL_DATA_ROOT" \
  --control-root "$AUTODL_CONTROL_ROOT" \
  --controller-id four_methods_four_datasets_aids_comrecgc_repair_v4 \
  --format json
```

The launcher uses tmux when available and nohup otherwise.  Reissuing the same
launch command after an outer-controller crash reuses the frozen manifest and
reconciles a live exp-run.  It must not create another scientific attempt while
the original supervisor identity is alive.

## Failure handling

- SIGKILL/SIGTERM during common recourse: the supervisor validates the exact
  resume contract and may retry once in the same root.
- RSS, sklearn, input/hash, lineage, leakage, or other semantic failure: fail
  closed without retry.
- Second process loss: terminal failure with all partial/checkpoint evidence
  retained.
- Insufficient memory or disk: remain blocked; do not delete old results.
- Never use SIGKILL to manage this route and never edit a failed attempt.

The expected full terminal root is:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/four_methods_four_datasets_aids_comrecgc_repair_v4/cells/aids/comrecgc/standardized/attempt-0
```
