# MolCLR-Node-FGW Threshold Consistency Audit

## Purpose

MolCLR-Node-FGW runs may use the same quantile labels while evaluating at
different absolute distance radii. For example, two methods can both report
the grid `0.05,0.10,0.20,0.30,0.50,0.70,0.90` with
`threshold_source=auto_quantile`, but each run derives its thresholds from its
own parent-candidate distance distribution. Equal quantiles therefore do not
imply equal thresholds.

The audit tool reads existing results only. It does not recompute FGW, modify
run configuration, select thresholds, or change cache contents.

## Fair Comparison Contract

A direct threshold-level comparison requires all of the following:

- the same Node-FGW distance definition;
- the same `fgw_lambda`, structure mode, feature cost, and atom penalty;
- the same frozen teacher, parent dataset, CF mode, candidate count, and
  evaluation protocol;
- the same number of evaluated parents;
- the same absolute FGW thresholds within the configured tolerances.

The quantile grid is audited separately. A pair can have
`same_quantile_grid=true` and `same_absolute_thresholds=false`; such a pair is
not directly comparable at those threshold labels.

Auto-quantile thresholds are useful for debugging, within-method curves, and
understanding a method's distance distribution. Final fair main tables should
use one explicit absolute FGW threshold list shared by every method.

## Usage

Audit every discovered Node-FGW run:

```bash
python scripts/audit_node_fgw_threshold_consistency.py \
  --eval-root outputs/hpc/eval \
  --output-dir outputs/hpc/eval/audits/node_fgw_threshold_consistency
```

Audit full runs only:

```bash
python scripts/audit_node_fgw_threshold_consistency.py \
  --eval-root outputs/hpc/eval \
  --output-dir outputs/hpc/eval/audits/node_fgw_threshold_consistency \
  --full-only
```

Compare every run against an explicitly selected reference:

```bash
python scripts/audit_node_fgw_threshold_consistency.py \
  --eval-root outputs/hpc/eval \
  --output-dir outputs/hpc/eval/audits/node_fgw_threshold_consistency \
  --reference-run-id '<relative-run-dir>::<method>'
```

The tool never selects Ours automatically. An explicit reference avoids
introducing method preference into the audit.

On HPC, submit:

```bash
sbatch scripts/slurm/audit_node_fgw_threshold_consistency.sh
```

Optional Slurm environment variables include `FULL_ONLY`, `STRICT`,
`INCLUDE_REGEX`, `EXCLUDE_REGEX`, `METHOD_REGEX`, `ATOL`, `RTOL`, and
`REFERENCE_RUN_ID`.

## Outputs

The output directory contains:

- `node_fgw_run_inventory.csv`: one row per `run_dir + method`;
- `node_fgw_threshold_long.csv`: one row per method and threshold;
- `node_fgw_threshold_pairwise.csv`: all run-pair consistency checks;
- `node_fgw_threshold_vs_reference.csv`: emitted when a reference is given;
- `node_fgw_threshold_audit.json`: machine-readable counts and warnings;
- `node_fgw_threshold_audit_report.txt`: human-readable audit report.

`direct_threshold_comparison_ok=true` means that FGW configuration,
evaluation protocol, parent count, and absolute thresholds are all aligned.
It does not mean that the methods share a selector or candidate-generation
algorithm; `selection_method` is retained as provenance.

## Interpreting Auto-quantile Warnings

When full runs share quantile labels but differ in absolute thresholds, the
report states:

> These runs use the same quantile labels but different absolute FGW radii;
> their coverage values should not be treated as measurements at the same
> threshold.

The recommended next step is to choose a reference explicitly, extract an
agreed absolute threshold list, and rerun every final-table method using that
fixed list.
