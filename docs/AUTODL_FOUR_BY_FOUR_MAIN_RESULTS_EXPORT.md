# AutoDL four-by-four main-results exporter

The final exporter is presentation-only. It reads the frozen
`matrix_status.json` registry and the 16 standardized cell roots recorded by
that registry. It never recomputes a prediction, distance, strict flip,
threshold, selector, candidate order, or scientific metric.

## Release gate

Final outputs are created only when all of the following are true:

- `audit_complete=true`, `matrix_total_cells=16`,
  `matrix_complete_cells=16`, and `all_cells_complete=true`;
- the exact cells are four datasets × `Ours`, `GCFExplainer`, `GlobalGCE`, and
  `ComRecGC` in that order; `CLEAR` is rejected rather than aliased;
- every status is `FROZEN_PASS` or `ADOPTABLE_PASS`;
- every standardized root contains the complete CSV/JSON contract and its
  final artifact audit closes the source-file hashes;
- dataset, held-out split, oracle, MolCLR encoder, distance line, strict-flip
  mode, and frozen threshold identity agree across all four methods within a
  dataset;
- selection/threshold manifests prove calibration-only fitting and prove that
  held-out test was opened only after selector freeze;
- BACE and TasteMolNet prove `rf_oracle_used=false`; AIDS and Mutagenicity
  retain RF;
- TasteMolNet destination labels remain in `{0,2}` and destination-distribution
  fields are copied into the combined output.

When any cell or closure is unavailable, the output root contains only
`partial_staging_audit.json` and `BLOCKED_INCOMPLETE_MATRIX`. No numerical CSV,
PNG, PDF, TeX, or plausible zero-filled row is emitted. With
`--require-complete`, that state exits nonzero.

## Direct AutoDL command

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
PROJECT=/root/autodl-tmp/worktrees/run-four-by-four-<commit>
MATRIX=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/matrix_status.json
OUTPUT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/final/<fresh-run>

PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/export_four_by_four_main_results.py" export \
  --matrix-status "$MATRIX" \
  --output-root "$OUTPUT" \
  --project-root "$PROJECT" \
  --require-complete
```

The exporter rejects any output path below `paper/`. Per-dataset results are
written under `<dataset>/combined/`; four-dataset panel files live at the fresh
output root. Figure 3 uses K=1..20, Figure 4 plots the raw threshold rows in
their frozen order with no interpolation/spline/smoothing, and Table 2 uses
K=10. The serif fonts, line widths, sparse markers, and established method
colors follow the audited AIDS/Mutagenicity renderer. The active method order
is `Ours`, `GCFExplainer`, `GlobalGCE`, `ComRecGC`.

## Persistent-controller task fragment

Prepare a dependency contract whose `cells` object contains exactly 16 unique
terminal PASS task IDs. The matrix audit must be a separate seventeenth task;
its output provides the final post-cell `matrix_status.json`.

```json
{
  "matrix_task_id": "final_matrix_audit",
  "matrix_status": "{dep_final_matrix_audit_output}/matrix_status.json",
  "cells": {
    "AIDS/Ours": "aids_ours_final",
    "AIDS/GCFExplainer": "aids_gcfexplainer_final",
    "AIDS/GlobalGCE": "aids_globalgce_final",
    "AIDS/ComRecGC": "aids_comrecgc_final",
    "Mutagenicity/Ours": "mut_ours_final",
    "Mutagenicity/GCFExplainer": "mut_gcfexplainer_final",
    "Mutagenicity/GlobalGCE": "mut_globalgce_final",
    "Mutagenicity/ComRecGC": "mut_comrecgc_final",
    "BACE/Ours": "bace_ours_b14",
    "BACE/GCFExplainer": "bace_gcfexplainer_final",
    "BACE/GlobalGCE": "bace_globalgce_final",
    "BACE/ComRecGC": "bace_comrecgc_final",
    "TasteMolNet/Ours": "taste_ours_final",
    "TasteMolNet/GCFExplainer": "taste_gcfexplainer_final",
    "TasteMolNet/GlobalGCE": "taste_globalgce_final",
    "TasteMolNet/ComRecGC": "taste_comrecgc_final"
  }
}
```

```bash
PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/export_four_by_four_main_results.py" task-fragment \
  --controller-id four_methods_four_datasets_continuation_v1 \
  --dependency-contract /persistent/control/final-export-dependencies.json \
  --output-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/final/controller \
  --fragment-output /persistent/control/fragments/final-export.json
```

The emitted generic task has:

```text
task id:           four_by_four_main_results_export
stage:             FOUR_BY_FOUR_MAIN_RESULTS_EXPORT
resource:          cpu
dependencies:      16 distinct cell terminal tasks + final matrix audit
required marker:   [FOUR_BY_FOUR_MAIN_RESULTS_PASS]
```

Consequently, a blocked Taste cell or the currently blocked BACE GlobalGCE
native route keeps the exporter non-READY without a dummy workload. The paired
Slurm file is static CLI parity only; this continuation never submits it and
never connects to HPC.
