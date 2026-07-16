# FGW SOTA Figure Generation

`scripts/plot_fgw_sota_figures.py` is a read-only presentation layer for
existing MolCLR-Node-FGW evaluation outputs. It does not load a teacher,
recompute embeddings or FGW distances, alter candidate rankings, or rewrite
the underlying evaluator artifacts.

## Inputs

The Figure 3 report directory is searched in this order:

1. `fgw_q30_k10_main_figure3_fgw_coverage_cost_vs_k.csv`
2. `figure3_fgw_coverage_cost_vs_k.csv`

The required Figure 3 fields are `method`, `k`, `theta`, `coverage`, and one
conditional-cost field. `conditional_median_cost` is preferred, followed by
the legacy-compatible names accepted by the script. The field is presented as
the unified evaluator's conditional cost, not as the original GCFExplainer
paper-style unconditional recourse cost.

Figure 4 must be a dense `K=20` threshold curve. Its low-cost plot displays
`[0.015, q30]`, while normalized AUC is always computed over `[0, q30]`.

## HPC execution

Use the compute-node wrapper rather than running matplotlib on an HPC login
node:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/plot_fgw_sota_figures_gpu.sh
```

The wrapper uses the verified `A800` / `gpu:a800:1` allocation pattern from
`scripts/baselines/clear/slurm_clear.sbatch`. It validates source schemas and
all expected output files before declaring success.

## Outputs

The default output directory is
`outputs/hpc/eval/paper/molclr_node_fgw_sota_figures/`. It contains Figure 3
for `K=1..10` and `K=1..20`, low-cost and full-range Figure 4, a compact Table
2, source-data extracts, low-cost AUC, and `sota_presentation_audit.txt`.

The audit permits only the scoped phrase "low-cost and compact-budget SOTA"
when Ours is best at `K=10, q30` for coverage and conditional cost and has the
best normalized AUC on `[0, q30]`. It does not establish all-K or
all-threshold SOTA.
