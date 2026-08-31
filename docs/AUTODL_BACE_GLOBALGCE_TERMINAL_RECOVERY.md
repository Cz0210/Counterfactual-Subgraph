# AutoDL BACE GlobalGCE terminal recovery

This is a train-only, read-only recovery for the completed 100-epoch run whose
old materializer rejected all 80 native rules with
`rhs_edge_attr contains values outside [0,1]`. It does not restart GlobalGCE.
It decodes the pinned official affine bond-class scores by categorical
`argmax`, converts them to one-hot typed bonds, and then runs the unchanged hard
native-rule validator.

The source controller is fixed to:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/bace_globalgce_k20/run-97df90c4-5e1b-4f14-93d3-ebbf45ab7811
```

Set the remaining paths from that run's frozen `run_manifest.json` and the
existing BACE route configuration, then recover into a fresh root:

```bash
export FAILED_CONTROLLER_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/bace_globalgce_k20/run-97df90c4-5e1b-4f14-93d3-ebbf45ab7811
export SOURCE_ROUND_ROOT=$FAILED_CONTROLLER_ROOT/rounds/round-1-seed-7
export OUTPUT_DIR=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/bace_globalgce_terminal_recovery/attempt-$(date -u +%Y%m%dT%H%M%SZ)

python scripts/autodl/recover_bace_globalgce_terminal.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  recover \
  --failed-controller-root "$FAILED_CONTROLLER_ROOT" \
  --source-round-root "$SOURCE_ROUND_ROOT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --native-train-csv "$NATIVE_TRAIN_CSV" \
  --official-root "$OFFICIAL_ROOT" \
  --gnn-checkpoint "$GNN_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"
```

Adopt the output only when the process prints
`[BACE_GLOBALGCE_AFFINE_EDGE_TERMINAL_RECOVERY_PASS]`, the fresh root contains
`PASS`, and `recovery_receipt.json` reports at least ten semantic-unique valid
rules. `10 <= R < 20` is valid: all `R` rules are used and prefixes above `R`
plateau without copying rules. If `R < 10`, this route remains a scientific
failure and must not proceed to calibration or test.

`build-fragment` creates a production-controller fragment in which the recovery
keeps the existing `bace_globalgce_train_candidates` task ID. Therefore the
unchanged four calibration shards, calibration-only selector, four held-out
test shards, final freeze and CPU standardization retain their original
dependency and split-isolation contracts.
