# AutoDL TasteMolNet T13 native GlobalGCE full route

`scripts/run_tastemolnet_globalgce_full.py` is the executable, dataset-specific
successor to the bounded T8 smoke. It is not a release gate or a disabled
skeleton. Once the T8 PASS receipt and all explicit inputs are present, it runs
the real official GlobalGCE science and seals a complete main-table cell.

`T8_PASS_ROOT` must name the published managed-execution-v2 final directory,
not a legacy marker or standalone JSON receipt. T13 retains and revalidates
the managed terminal, reads the nested typed T8 verification, and requires its
`oracle_checkpoint_hash` to equal the T13 frozen GINE `model.pt` hash.

## Scientific contract

- dataset: TasteMolNet;
- classifier: the same frozen, validation-temperature-calibrated three-class
  GINE used by T3/T4/T8;
- source class: Sweet (`1`);
- native branches: Sweet-to-Bitter (`1 -> 0`) and Sweet-to-Tasteless (`1 -> 2`);
- generation/training data: train only;
- action: official attachment-aware LHS-to-RHS GlobalGCE transformation rule;
- rule identity: canonical native tensor-content hash, deduplicated across the
  two target branches;
- calibration: deterministic greedy marginal close-coverage selector;
- test: opened only after the selected rule order is fsynced and checkpointed;
- distance: exact MolCLR-Node-Wasserstein;
- counterfactual: untargeted strict flip, `pred_before == 1` and
  `pred_after != 1`;
- `K_MAX=20`, `MIN_RULES=10`, Table 2 at `K=10`;
- no RF oracle and no test-driven selection or threshold fitting.

The threshold grid is an explicit dataset-level frozen input. The runner does
not derive a threshold from held-out test data and does not invent a
method-specific grid. Its `threshold_config_hash` must equal the registry's
canonical hash of the numeric threshold list.

`MIN_RULES=10` applies after canonical cross-branch deduplication. An
individual target branch may contribute fewer than ten rules; T13 still fails
unless the merged unique set contains at least ten. Rules are never copied to
meet the gate.

## Real checkpoint and resume

Each target branch calls the existing
`OfficialGlobalGCEMutagenicityGenerator` with `resume=True`. The official
training path persists model, optimizer, scheduler, RNG, epoch checkpoint, and
training heartbeat state. T13 additionally persists `checkpoint.json` at the
dataset-specific stage boundaries.

Calibration and held-out test evaluation use one durable JSONL chunk plus one
manifest per parent. Each manifest binds the complete chunk hash and the full
split/parent/ordered-rule/GINE/MolCLR/threshold/distance/counterfactual
identity. A restart verifies and adopts complete chunks, then continues at the
first absent parent. It never appends blindly to a partial aggregate.

Each completed target branch carries an exact recursive file inventory that
includes its model, rules, training summary, epoch checkpoint, heartbeat, and
native rule catalogs. Resume and terminal verification re-hash that inventory
and re-parse the typed training identity before adoption.

Use `--resume` only with the same output root and exact input/config identity.
A different split, GINE, MolCLR checkpoint, threshold contract, T8 receipt, or
science setting is rejected.

## Two-process terminal publication

The science invocation ends at `SEALED` and does not write `PASS`. A separate
`--verify-only` process reopens the complete raw-file inventory, re-hashes both
branch trees and every parent chunk, replays canonical merge/dedup and the
calibration-only selector, checks calibration/test isolation, and replays
every standardized metric from the held-out pair matrix. It then runs the
ordinary 4-by-4 registry audit and only then writes `PASS` plus
`final_artifact_audit.json`. Publishing the updated 4-by-4 matrix remains an
explicit external registry/publisher action after this cell PASS.

The paired Slurm wrapper performs both invocations in that order:

```bash
export T8_PASS_ROOT=/absolute/t8/pass/root
export TASTEMOLNET_GNN_CHECKPOINT=/absolute/t3/checkpoint
export TASTEMOLNET_TRAIN_CSV=/absolute/train.csv
export TASTEMOLNET_CALIBRATION_CSV=/absolute/calibration.csv
export TASTEMOLNET_TEST_CSV=/absolute/test.csv
export GLOBALGCE_OFFICIAL_ROOT=/absolute/GlobalGCE
export MOLCLR_ROOT=/absolute/MolCLR
export MOLCLR_CHECKPOINT=/absolute/molclr/checkpoint.pth
export TASTEMOLNET_THRESHOLD_CONTRACT=/absolute/tastemolnet.json
export T13_OUTPUT_DIR=/absolute/fresh/t13/output

sbatch scripts/slurm/run_tastemolnet_globalgce_full.sh
```

Optional environment settings are `WNODE_CACHE_DB`,
`NODE_EMBEDDING_CACHE_DIR`, and `T13_EPOCHS` (default `100`, minimum `25`). If
`checkpoint.json` already exists, the wrapper supplies `--resume`
automatically.

## Standardized output

The terminal root includes:

- `figure3_coverage_vs_k.csv`;
- `figure4_coverage_vs_threshold.csv`;
- `table2_globalgce_k10.csv`;
- `prefix_metrics.csv` and `prefix_metrics.json`;
- `parent_best_distances.csv`;
- `destination_distribution.csv`;
- `summary.json`, `oracle_manifest.json`, `evaluation_manifest.json`;
- `freeze_manifest.json`, `run_manifest.json`,
  `final_artifact_audit.json`, and `PASS`;
- train-only branch artifacts, frozen selection evidence, and held-out pair
  matrices under `raw/`.

## Focused verification

```bash
conda run -n smiles_local python -m pytest -q \
  tests/baselines/test_tastemolnet_globalgce_full.py
python -m compileall src scripts
git diff --check
bash -n scripts/slurm/run_tastemolnet_globalgce_full.sh
```

The tests cover a real managed-v2 terminal with nested typed T8 verification,
the frozen threshold identity, exact prepared-split role, test-open
authorization, calibration-only ordering, merged-total `MIN_RULES`, K=10-to-20
plateau without rule copying, complete branch/checkpoint resume, strongly
bound parent-chunk resume, independent merge/selector/metric replay, registry
acceptance, and Slurm science/verifier ordering.
