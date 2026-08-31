# TasteMolNet calibration-only threshold authorities

This command materializes two missing, non-paper authorities in one short
GPU run:

- `t7_neurosed_threshold_authority.json`: official normalized NeuroSED q30,
  with residual/generated graph as query and original parent as target;
- `tastemolnet.json`: one q05/q10/q20/q30/q50/q70/q90 MolCLR WNode grid shared
  by all four Taste methods, theta=q30 and cost cap=q90.

The selector replays only the frozen T4 calibration cohort.  It must reproduce
64 selected parents, 733 valid connected deletions, and 38 strict flips before
measuring distances.  It does not open train, validation, or test payloads and
does not publish a main-table cell.

## AutoDL command

After pulling the implementation commit into a clean checkout, choose a fresh
output path and run:

```bash
export TASTE_T3_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed
export TASTE_T4_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/t4-oracle-smoke-03dd2e64-ab20-4df7-b514-fdafaeed0e52
export TASTE_GRAPH_CACHE_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/cache/tastemolnet/16af8ead8a17b6bd3941d9eb5879c5be75c14114/molecular_graph_v1
export TASTE_MANAGED_NEUROSED_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/gcfexplainer/neurosed_fixed_budget/seed7/managed_adoption/v2/final/neurosed-fixed-budget-managed-ef46bb01-6e80-43c5-bebd-ac65250f2bc6
export MOLCLR_ROOT=/root/autodl-tmp/counterfactual-subgraph/pretrained_models/MolCLR
export MOLCLR_CHECKPOINT=/root/autodl-tmp/counterfactual-subgraph/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth
export TASTE_THRESHOLD_OUTPUT_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/threshold_authorities/selector-$(date -u +%Y%m%dT%H%M%SZ)
sbatch scripts/slurm/select_tastemolnet_threshold_authorities.sh
```

Expected runtime is roughly 5--15 minutes on one A800.  The output root must
be fresh.  Success requires `PASS` to contain
`[TASTE_THRESHOLD_AUTHORITIES_PASS]` and the selection receipt to report
`test_payload_loaded=false` and `paper_cell_published=false`.

For T7, read `neurosed_distance_threshold` from
`t7_neurosed_threshold_authority.json` and retain that file and its SHA with
the typed release.  For WNode consumers, point the existing threshold-contract
argument directly at `tastemolnet.json`; its format is accepted by both the
strict shared loader and the T11/T13 loaders.
