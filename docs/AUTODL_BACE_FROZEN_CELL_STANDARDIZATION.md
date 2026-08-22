# AutoDL BACE frozen-cell standardization

The BACE scientific terminal roots are intentionally not paper-matrix cells.
`bace_b14_frozen`, `bace_gcfexplainer_final_freeze`, and
`bace_comrecgc_final_freeze` prove the scientific freeze, but they do not carry
the complete common Figure 3, Figure 4, Table 2, registry, and hash-closure
schema.

`scripts/autodl/build_bace_cell_standardization_tasks.py` adds three fresh,
CPU-only, manifest-only continuation tasks:

| Matrix cell | Scientific dependency | Standardized terminal task |
|---|---|---|
| BACE / Ours | `bace_b14_frozen` | `bace_ours_standardized` |
| BACE / GCFExplainer | `bace_gcfexplainer_final_freeze` | `bace_gcfexplainer_standardized` |
| BACE / ComRecGC | `bace_comrecgc_final_freeze` | `bace_comrecgc_standardized` |

GlobalGCE is not synthesized by this layer. Its reviewed `BLOCKED_CODE`
terminal remains unchanged.

## Safety boundary

The standardizer follows only identities already frozen by the scientific
terminal:

```text
FINAL_PASS.json
  -> frozen selection manifest
  -> held-out test/merge manifest
  -> four verification-shard manifests
  -> frozen pair matrix and final prefix metrics
```

Every followed artifact path, size, and SHA256 is checked. The test split
path/hash recorded by the verification shards must equal the frozen GINE
bundle's split manifest, but the raw held-out CSV is never opened or statted.
The selected rule order, thresholds, classifier prediction, WNode distance,
and scientific selector are never recomputed. The exporter only replays the
already-frozen deterministic prefix aggregation and checks it against the
terminal metrics.

The gate additionally requires:

- BACE GINE, two classes, source label 1;
- `oracle_backend=gnn` and `rf_oracle_used=false` throughout the chain;
- `CF_MODE=strict_flip` and binary strict flips `1 -> 0`;
- calibration selector frozen before test;
- no selector or threshold refit on test;
- identical GINE, MolCLR, dataset/split, candidate-order, and threshold
  identities through the chain.

Metrics absent from the frozen terminal evidence (`StructRed`, `CovRed`,
`ValidRate`, and `AvgSize`) are emitted as `N/A` with an explicit reason. They
are never filled with zero or inferred values.

## Controller fragment

```bash
python scripts/autodl/build_bace_cell_standardization_tasks.py \
  --controller-id four_methods_four_datasets_continuation_v1 \
  --output-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/runs/four_methods_four_datasets_continuation_v1/cells \
  --gnn-checkpoint /autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689 \
  --fragment-output /absolute/fresh/bace_cell_standardization.tasks.json
```

Optional `--expected-*-hash` arguments bind a preregistered dataset, test
split, MolCLR, or threshold-grid identity. A mismatch fails closed.

The final matrix audit must map BACE Ours/GCFExplainer/ComRecGC to the three
standardized terminal task IDs above. `build_four_by_four_final_tasks.py`
rejects direct mappings to raw B14/baseline terminal tasks.

## Standardized output

Each task writes a fresh direct standardized root containing:

```text
figure3_coverage_vs_k.csv
figure4_coverage_vs_threshold.csv
table2_<method>_k10.csv
prefix_metrics.csv
prefix_metrics.json
parent_best_distances.csv
destination_distribution.csv
summary.json
run_manifest.json
oracle_manifest.json
evaluation_manifest.json
artifact_manifest.json
freeze_manifest.json
_FINALIZED.json
final_artifact_audit.json
PASS
```

`PASS` is published last inside an atomically renamed output directory.

The paired files under `scripts/slurm/` are static CLI-parity wrappers required
by repository policy. The AutoDL continuation does not submit or execute them.
