# AutoDL four-method × four-dataset matrix audit

`scripts/autodl/audit_four_methods_four_datasets.py` builds the read-only
registry used before any continuation controller schedules missing paper
experiments. It scans existing output trees, but it never runs an oracle,
recomputes a distance, selects a candidate, opens a test split, or renders a
paper number.

## Fixed matrix

The registry always emits exactly these sixteen cells:

```text
AIDS, Mutagenicity, BACE, TasteMolNet
×
Ours, GCFExplainer, GlobalGCE, ComRecGC
```

`CLEAR` is not an alias for `ComRecGC`. Historical CLEAR files remain visible
in `artifact_inventory.csv` and `stale_artifacts.csv`, but cannot fill any of
the sixteen cells.

## Status contract

The only accepted states are:

```text
FROZEN_PASS
ADOPTABLE_PASS
RUNNING
READY
MISSING
STALE_ORACLE
STALE_DATASET
STALE_SPLIT
STALE_METRIC
INCOMPLETE
BLOCKED_LICENSE
BLOCKED_CODE
FAILED
```

The audit fails closed. A similar directory name, a PNG, a combined rendering,
or a Figure/Table CSV alone cannot produce a passing cell. A passing cell must
have self-identifying JSON evidence, raw method evidence, a passing final
artifact audit, the complete Figure 3/Figure 4/Table 2 CSV contract, frozen
dataset/test/oracle/MolCLR/threshold hashes, strict-flip semantics, and explicit
test-selection exclusion. Cross-method hash disagreement downgrades all
otherwise passing cells for that dataset.

The auditor also understands the continuation layout without using its path
name as scientific evidence:

```text
<fresh-cell-root>/
├── final_gate.json
├── run_manifest.json
├── _RUN_COMPLETE.json
├── PASS
└── standardized/
    ├── freeze_manifest.json
    ├── _FINALIZED.json
    └── <standardized files>
```

All top-level gates must agree, `PASS` must contain the literal `PASS`, the
recorded standardized root and manifest hashes must match, and every required
small file must match the frozen byte count and SHA. Large frozen payloads stay
under the bounded-hash policy. Neither the directory name nor a marker by
itself promotes a cell.

`generation_adoption_candidate=true` means only that raw evidence appears
available for deterministic unified re-evaluation. It does not mean the
generation has already been adopted, and it never promotes the cell to a paper
PASS. A paper PASS additionally needs an explicit raw-completeness/adoption
gate; merely finding `pair_details.csv` is insufficient.

TasteMolNet defaults to `BLOCKED_LICENSE`. It can leave that state only when an
explicit license gate contains all of:

```json
{
  "status": "PASS",
  "passed": true,
  "license_basis": "an explicit reuse basis for the exact data"
}
```

A downloadable file or public paper is not, by itself, a license basis.

## Inputs

Multiple scan roots may be supplied. They are read only:

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
PROJECT=/root/autodl-tmp/worktrees/<immutable-execution-worktree>
RUNTIME=/autodl-fs/data/counterfactual-subgraph-runtime
RUN_ID=four_methods_four_datasets_audit_$(date -u +%Y%m%dT%H%M%SZ)
OUT=$RUNTIME/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/$RUN_ID

PYTHONPATH="$PROJECT" "$PY" \
  "$PROJECT/scripts/autodl/audit_four_methods_four_datasets.py" \
  --runtime-root "$RUNTIME" \
  --scan-root "$RUNTIME/outputs/hpc" \
  --scan-root "$RUNTIME/outputs/autodl" \
  --scan-root "$RUNTIME/outputs/final" \
  --output-root "$OUT" \
  --expectations-json /path/to/frozen_expectations.json \
  --explicit-cells-json /path/to/candidate_roots.json
```

The output root must be fresh and empty. Re-running into an existing non-empty
root fails instead of overwriting audit evidence.

`frozen_expectations.json` may bind independent dataset identities:

```json
{
  "datasets": {
    "AIDS": {
      "dataset_hash": "<sha256>",
      "split_hash": "<sha256>",
      "oracle_backend": "rf",
      "classifier_family": "random_forest",
      "oracle_checkpoint": "/persistent/path/aids_rf_model.pkl",
      "oracle_hash": "<sha256>",
      "molclr_checkpoint_hash": "<sha256>",
      "threshold_config_hash": "<sha256>",
      "thresholds": [0.0, 0.01, 0.02],
      "theta_star": 0.01,
      "cost_cap": 0.02,
      "threshold_source": "frozen calibration manifest",
      "threshold_source_split": "calibration",
      "test_used_for_selection": false
    }
  }
}
```

An explicit candidate map is optional. It cannot override contradictory
artifact metadata:

```json
{
  "cells": {
    "AIDS/Ours": {
      "standardized_output_root": "/persistent/path/aids/ours"
    },
    "Mutagenicity/ComRecGC": [
      "/persistent/path/mut/comrecgc_candidate_1",
      "/persistent/path/mut/comrecgc_candidate_2"
    ]
  }
}
```

If more than one passing candidate for the same cell has different hashes, the
cell becomes `INCOMPLETE` with
`AMBIGUOUS_MULTIPLE_PASSING_ARTIFACT_ROOTS`; the audit never chooses by mtime or
directory name.

## Outputs

The command writes exactly the registry/audit layer:

```text
matrix_status.csv
matrix_status.json
combined_audit.json
oracle_registry.json
evaluation_contract.json
artifact_inventory.csv
stale_artifacts.csv
adoption_report.md
threshold_contracts/
├── aids.json
├── mutagenicity.json
├── bace.json
└── tastemolnet.json
```

`evaluation_contract.json` is the unified evaluation/export skeleton. It fixes
WNode, strict flip, K=1..20, Table 2 K=10, native action semantics, required
metrics, Taste destination fields, and the final standardized filenames. It
does not synthesize a missing evaluator or result.

Each per-dataset threshold artifact is directly consumable by
`run_slot_unified_eval.py --thresholds-json` only when its `status` is `PASS`;
then it has top-level `thresholds`, `theta_star`, and `cost_cap` plus an explicit
calibration source and source hash, or an existing frozen protocol with the
same source identity and test-selection exclusion. If these values were not
supplied by frozen expectations, the file says `MISSING_NOT_INFERRED` and
deliberately omits the numeric fields. A test-derived or incompletely
attributed contract becomes `INVALID_FAIL_CLOSED`. The registry never mines
Figure 4/test curves for these values.

`combined_audit.json` closes the byte size and SHA-256 of every registry
artifact other than itself, including the terminal `matrix_status.json`
payload. It also records the exact status histogram, matrix count, read-only
source policy, and absence of scientific recomputation or numeric imputation.
All siblings and their containing directories are fsynced before
`matrix_status.json` is atomically published last.

Inventory hashing is deliberately bounded by `--max-hash-bytes` (64 MiB per
file by default). Larger files are listed as `SKIPPED_SIZE_LIMIT`; model and raw
payload SHA values are read from their frozen manifests rather than repeatedly
rehashing multi-gigabyte artifacts.

Use `--require-complete` only for the final 16/16 publication gate. Without it,
an incomplete audit still returns success because recording missing and blocked
cells is itself a valid audit result.

## Strict BACE GCFExplainer 7/16 -> 8/16 append

A fresh audit containing only BACE GCFExplainer is not a successor matrix.
After its shared-threshold standardized root passes, use the dataset-specific
append entrypoint to reopen the frozen seven-cell authority and prove an exact
one-cell transition:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT" "$PY" -I -B \
  "$PROJECT/scripts/autodl/append_bace_gcf_matrix_authority.py" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --prior-authority-root "$MATRIX/audits/bace_ours_frozen_7of16_20260829T062100Z" \
  --bace-gcf-standardized-root "$BACE_GCF_STANDARDIZED" \
  --superseded-audit-root "$INCOMPLETE_GCF_ONLY_AUDIT" \
  --output-root "$MATRIX/audits/bace_gcf_strict_append_8of16_$TS"
```

The destination must not exist and the execution checkout must be clean and
committed. The append command scans no ambient output tree. It re-audits only
the seven passing predecessor roots and the new cell, rejects any change to a
non-target row, and requires the new BACE cell to share the frozen Ours
dataset/split/GINE/MolCLR/threshold identities. `append_authority.json` and
`superseded_snapshots.json` are included in `combined_audit.json`; the terminal
`matrix_status.json` remains the last publication. Superseded roots are never
modified. A historical top-level `matrix_status.json` that predates
`combined_audit.json` is recorded explicitly as
`LEGACY_MATRIX_STATUS_ONLY_COMBINED_AUDIT_ABSENT`; its exact physical file
identity and SHA are preserved, and the append does not pretend that it had a
combined closure.

## Controller hand-off contract

This is a finite foreground CPU command; it does not daemonize. A controller
task can require the eight top-level files and four threshold-contract files
listed above and use:

```text
marker_path = <fresh-output-root>/matrix_status.json
marker_field = audit_complete
marker_value = true
```

That marker means the read-only audit completed, not that the paper matrix is
complete. `matrix_status.json` is atomically published after every sibling
audit and threshold artifact, so controllers cannot observe the marker early.
A final Figure/Table exporter must instead require:

```text
marker_path = <fresh-output-root>/matrix_status.json
marker_field = all_cells_complete
marker_value = true
matrix_complete_cells = 16
```

The same fail-closed check is available from the foreground CLI through
`--require-complete`, which returns exit code 3 after writing truthful partial
artifacts when any cell is not passing. A TasteMolNet license block remains a
`BLOCKED_LICENSE` row with unavailable identities/results; it is never rendered
as numeric zero.

A paired Slurm wrapper is retained only as static CLI parity required by the
repository policy. The active execution is CPU-only and AutoDL-local; this
campaign does not submit the wrapper or access HPC.
