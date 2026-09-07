# BACE A+ offline reporting (2026-09-08)

This leaf reporting change does not alter any active science, selector, oracle,
temperature, candidate pool, raw distance, main matrix or registry.

`src/experiments/bace_gin_reporting.py` reduces saved parent minima once for all
K=1..20. Figure3 contains coverage (%) and the unchanged fixed-capped mean cost.
Primary Figure4 is exact ECDF at K10; a separate auxiliary panel uses K20.
Table2 at K10 shares the same reducer. Missing minima remain infinite internally;
their capped contribution is the original cap, not zero. Undefined conditional
median is N/A. Missing methods remain PENDING or BLOCKED without a plotted line.

The input Figure3/Figure4/Table2 must agree with saved parent records, including
fixed parent IDs, classifier predictions, theta/cap and nested prefixes. Any
numeric conflict rejects the export. Display consistency is explicitly not a new
independent scientific acceptance. The output version label distinguishes V1,
A+, and other future versions; old V1 figures and data are not overwritten.

CLI (explicit source and fresh output directory):

```sh
python -I -B scripts/experiments/replot_bace_gin.py --source-csv /absolute/source_csv --output /absolute/fresh-figures --version-label BACE-GIN-fixed-pool-v1
```

The paired Slurm wrapper remains CPU-only; this is display work with no model
loading or inference, so GPU and inference config flags are intentionally absent.
