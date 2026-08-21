# BACE and TasteMolNet frozen-GNN route on AutoDL

Date: 2026-08-22

## Audit conclusions

- The frozen BACE split is reusable: 1,513 unique valid molecules with
  scaffold-disjoint train/validation/calibration/test partitions of
  959/187/129/238. Validation and test remain unchanged.
- No existing BACE checkpoint qualifies as a task-specific frozen GNN.
- The historical BACE Morgan-RF teacher and every artifact selected or verified
  with it are `RF_CONTAMINATED`; they remain available for diagnosis but cannot
  enter this route.
- ChemLLM weights are proposal assets and MolCLR is a distance encoder. Both are
  independent of the new classifier checkpoint.
- AutoDL has a compatible Python 3.10 / Torch 2.7.1+cu118 / PyG 2.8 / RDKit
  environment. The implementation detects this environment and does not
  reinstall or upgrade it blindly.

## Classifier route

The default classifier is a five-layer GINE with shared atom/bond schemas,
mean pooling, residual connections, batch normalization, a two-layer readout,
class-weighted cross entropy, AdamW, and seed 7. The same registry exposes GIN,
GCN, and GATv2 for later sensitivity studies without duplicating data or
training pipelines.

BACE has two output logits; TasteMolNet has three. Checkpoint selection and
temperature fitting use validation only. Calibration data is reserved for
thresholds, selector fitting, and ordered-rule freezing. Held-out test data is
evaluated only after those choices are frozen.

## Counterfactual semantics

For source class `y`:

```text
strict_flip = pred_before == y and pred_after != y
cf_drop = p_before[y] - p_after[y]
margin_drop = margin_before - margin_after
```

Every record stores full probability vectors, source and destination labels,
the classifier checkpoint identity, and a separate MolCLR checkpoint identity.

## BACE state machine

The AutoDL route uses the following gated stages:

```text
B0_AUDIT -> B1_DATA_READY -> B2_GNN_SMOKE -> B3_GNN_FULL
-> B4_GNN_CALIBRATED -> B5_ORACLE_SMOKE -> B6_PPO_SMOKE
-> B7_PPO_FULL -> B8_POOL_BASE -> B9_POOL_HIGHTEMP
-> B10_POOL_MERGED -> B11_CROSS_PARENT_VERIFIED -> B12_SELECTOR
-> B13_FINAL_EVAL -> B14_FROZEN
```

Each stage writes state, gate, and manifest evidence. RF provenance or a failed
upstream gate prevents downstream launch. The new runner is independent of the
MUT/AIDS recovery controller and never invokes Slurm on AutoDL.

## TasteMolNet execution boundary

`RUN_TASTEMOLNET=0` is the default. This task may prepare provenance, clean
data, scaffold splits, graph caches, configs, CPU tests, and a tiny forward
smoke. It must not launch full GNN training, PPO, candidate generation,
verification, selector fitting, or baselines until explicitly enabled.
