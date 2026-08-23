# AutoDL three-dataset release supervisor

## Scope

`scripts/autodl/run_three_dataset_release_supervisor.py` is a CPU-only,
read-only scientific release sidecar. It does not join or modify any active
controller, acquire a GPU lock, write a standardized cell root, or write
`paper/`. Its private control root contains only controller identity, PID,
heartbeat, state, transaction, and log records.

The sidecar waits for the exact twelve standardized AIDS, Mutagenicity, and
BACE cells. Before all twelve pass, it reports `WAITING_DEPENDENCY` and creates
neither a matrix registry nor any table/figure output. After all twelve pass,
it atomically publishes one canonical sixteen-row registry: twelve audited
paper cells plus four `TasteMolNet` rows with canonical status
`BLOCKED_LICENSE` and reason `BLOCKED_LICENSE_REVIEW`. It then calls the
existing staging-only exporter and
publishes byte-identical trees at:

- `$MATRIX_ROOT/three_datasets_complete_v1`
- `$RUNTIME/outputs/autodl/paper_staging/three_datasets_complete_v1`

The complete four-dataset paper release remains frozen at 12/16.

## Cell catalog

The tracked
`configs/autodl/three_dataset_release_cells_v1.template.json` records the
audited path decision. A path does not become an active cell binding until its
external owner manifest is also frozen by SHA and proves the expected task ID
and exact output path/template.

`--cell-root` names the external task's exact `expected_output` root. For the
AIDS/Mutagenicity common-recourse controllers this is a
`nested_standardized` container, so the scientific matrix root is its
`standardized/` child and the controller-level `PASS` remains at the parent.
BACE standardizers and the six adopted v4 roots use the `direct` layout. The
layout is immutable in the catalog/spec and both directories are inode/hash
bound at release time.

| Dataset/method group | Catalog state | Required authority |
| --- | --- | --- |
| AIDS + Mutagenicity: Ours, GCFExplainer, GlobalGCE | Fixed | Exact user-approved v4 adoption manifest and six adopted standardized roots |
| BACE Ours | Fixed | `bace_ours_standardized` owner task |
| BACE GCFExplainer | Fixed | `bace_gcfexplainer_standardized` owner task |
| BACE ComRecGC | Fixed | `bace_comrecgc_standardized` owner task |
| AIDS ComRecGC | Placeholder until route choice is immutable | Exact selected standardization task, manifest, and fresh root |
| Mutagenicity ComRecGC | Placeholder until the AIDS dependency selects the route | Exact selected standardization task, manifest, and fresh root |
| BACE GlobalGCE | Placeholder until CPU standardization is scheduled | `bace_globalgce_standardized` owner task and its fresh standardized output |

The GlobalGCE v5 `final/attempt-0` candidate is a native scientific-final
root, not a standardized paper cell. It is only an upstream dependency of the
existing CPU entrypoint
`scripts/autodl/standardize_bace_frozen_cell.py`. The supervisor must never
bind or count that raw final root directly.

For every active non-v4 cell, the frozen release spec contains:

1. the exact standardized root;
2. the owner manifest path, size, and SHA-256;
3. the owner task ID;
4. the JSON path and exact/template output binding from that task;
5. at release time, the SHA-256 and inode/stat identity of `PASS`,
   `_FINALIZED.json`, and every numeric/provenance closure file.

Changing an owner manifest after spec construction blocks the controller even
while other cells are still missing.

## Build a release spec

The spec builder is non-scientific. It writes one fresh immutable spec and one
fresh build audit. If any route is unresolved, it prints
`[THREE_DATASET_RELEASE_SPEC_BLOCKED_PLACEHOLDERS]`; it does not invent a root
from a candidate hint.

Use one `--cell-root`, `--cell-owner-manifest`, and `--cell-owner-task` triple
for each placeholder after the corresponding external controller manifest is
immutable. For example:

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
MATRIX_ROOT=$RUNTIME/outputs/autodl/paper_matrix/four_methods_four_datasets_v1
RELEASE_ID=three_dataset_release_v1

$PY scripts/autodl/build_three_dataset_release_supervisor_spec.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --catalog "$PWD/configs/autodl/three_dataset_release_cells_v1.template.json" \
  --controller-id "$RELEASE_ID" \
  --project-root "$PWD" \
  --runtime-root "$RUNTIME" \
  --python "$PY" \
  --state-root "$RUNTIME/control/$RELEASE_ID" \
  --registry-root "$MATRIX_ROOT/registries/three_datasets_complete_v1" \
  --output-root "$MATRIX_ROOT/three_datasets_complete_v1" \
  --paper-staging-root "$RUNTIME/outputs/autodl/paper_staging/three_datasets_complete_v1" \
  --expectations-json /ABS/PINNED_EXPECTATIONS.json \
  --taste-license-gate-json /ABS/TASTE_LICENSE_GATE.json \
  --cell-root 'AIDS/ComRecGC=/ABS/SELECTED_AIDS_STANDARDIZED/attempt-0' \
  --cell-owner-manifest 'AIDS/ComRecGC=/ABS/AIDS_OWNER_MANIFEST.json' \
  --cell-owner-task 'AIDS/ComRecGC=aids_comrecgc_standardized_external_memory' \
  --cell-root 'Mutagenicity/ComRecGC=/ABS/SELECTED_MUT_STANDARDIZED/attempt-0' \
  --cell-owner-manifest 'Mutagenicity/ComRecGC=/ABS/MUT_OWNER_MANIFEST.json' \
  --cell-owner-task 'Mutagenicity/ComRecGC=mut_standardize_from_parity_common' \
  --cell-root 'BACE/GlobalGCE=/ABS/FRESH_BACE_STANDARDIZATION/bace/globalgce/standardized/attempt-0' \
  --cell-owner-manifest 'BACE/GlobalGCE=/ABS/BACE_STANDARDIZATION_MANIFEST.json' \
  --cell-owner-task 'BACE/GlobalGCE=bace_globalgce_standardized' \
  --spec-output "$RUNTIME/control/specs/$RELEASE_ID.json" \
  --build-audit "$RUNTIME/outputs/autodl/audits/$RELEASE_ID-build.json" \
  --require-runnable
```

`--require-runnable` is the deployment gate. The output root is required to be
exactly `$MATRIX_ROOT/three_datasets_complete_v1`; every mutable release path
must remain under `$RUNTIME`, and all three publish destinations must be
absent at build time.

## Run, restart, and inspect

Start only after the immutable execution commit and spec SHA have been
reviewed:

```bash
nohup "$PY" scripts/autodl/run_three_dataset_release_supervisor.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  run --spec "$RUNTIME/control/specs/$RELEASE_ID.json" \
  >"$RUNTIME/control/$RELEASE_ID/supervisor.log" 2>&1 &
```

Status is read-only:

```bash
"$PY" scripts/autodl/run_three_dataset_release_supervisor.py \
  --config configs/hpc.yaml \
  status --state-root "$RUNTIME/control/$RELEASE_ID"
```

The heartbeat interval is fixed at 60 seconds. PID identity binds Linux boot
ID, procfs start ticks, and command-line SHA rather than trusting a reusable
numeric PID. An exclusive `flock` rejects a second supervisor. `SIGTERM` or
`SIGINT` records a graceful `STOPPED` state; restart uses the same spec and
state root and increments `restart_count`.

If the parent process is lost during export, the transaction records the
exporter PID identity. A still-live owned exporter is observed but not
duplicated. Complete temporary trees are verified and promoted; incomplete
trees remain preserved and at most one fresh retry is allowed. Directory
publication uses no-replace semantics, and a half-promoted runtime/staging
pair is reconciled only when both trees are byte-identical.

## Release gates

The supervisor opens no numeric file until all twelve roots contain a
non-empty standardized closure including exact `PASS\n` and `_FINALIZED.json`.
It then reruns the existing registry and presentation audits without changing
scientific values. In particular:

- the six v4 cells are validated and rendered from adopted files; they are not
  scientifically recomputed;
- BACE remains GINE/RF-free under the existing registry contracts;
- raw GlobalGCE final artifacts cannot substitute for standardized closure;
- TasteMolNet smoke output cannot enter the matrix;
- no missing curve point is interpolated and no missing cell is filled with
  zero;
- no file under `paper/` is read as a release destination or written.

The static Slurm wrappers exist only for repository CLI parity. They exit 78
before execution because this active sidecar is AutoDL CPU-only and must not
consume an A800 allocation.
