# BACE / TasteMolNet Frozen-GNN AutoDL handoff

Date: 2026-08-22

Scope: AutoDL only. No HPC GPU or CPU was used for this route. AIDS and
Mutagenicity remained read-only and were not stopped, restarted, or modified.

Current boundary: BACE B0--B5 and the bounded TasteMolNet CPU foundation smoke
are complete. B6 ran once as a calibrated-GNN scoring preflight and correctly
ended `BLOCKED`: no GNN PPO adapter or provenance-clean BACE policy
initialization exists yet. B7--B14 remain `NOT_STARTED`. TasteMolNet heavy work
remains blocked by license review and `RUN_TASTEMOLNET=0`.

## 1. Git branch and commit

- Branch: `feat/bace-tastemolnet-gnn-autodl`.
- B6 route and currently deployed commit:
  `8b17fb1096666852b0680f899073dd82f207cce1`.
- Current subject: `feat: gate frozen GNN BACE route`.
- B0--B5 scientific artifact commit:
  `8b9628c6fa4125f05cd84f05212aa2b76b34b8a3`.
- Earlier logical commits:
  - `3bc8bb4` — TasteMolNet data and multiclass semantics;
  - `b3dcc30` — generic frozen molecular-GNN oracle;
  - `80c068b` — AutoDL frozen-GNN scheduler;
  - `861ba55` — BACE/TasteMolNet Frozen-GNN route documentation.

The commit gate was checked once at deployment. Do not repeatedly poll or
re-verify SHA while experiments are running.

## 2. Pre-existing dirty files

The primary local worktree contained a pre-existing untracked `paper/` tree.
It belongs to the user and was not bulk-added, deleted, reset, or moved. Within
that untracked tree, this task made five paper-facing edits:

- `paper/sections/experiments.tex`;
- `paper/scripts/build_main_result_figures.py`;
- `paper/data/main_results/table2_fixed_budget.csv`;
- `paper/sections/generated/table1_dataset_statistics.tex`;
- `paper/scripts/build_dataset_statistics.py`.

Because the whole `paper/` tree was already untracked, Git status alone cannot
separate its older contents from these five task edits. Preserve it as one
pre-existing user tree and review/stage the five files separately if the paper
is later brought under version control.

The B6--B14 fail-closed route was committed as `8b17fb1`, pushed, deployed to
AutoDL, and exercised through B6 only. The feature worktree was clean after
that commit; no pending route code remains outside Git.

## 3. Files added or modified by this route

The route through `861ba55` added or updated these coherent groups:

- active registry/configuration: `README.md`,
  `configs/datasets/{bace_gnn,tastemolnet}.yaml`,
  `configs/gnn/{gine,gin,gcn,gatv2}.yaml`, and
  `configs/autodl/{bace_gine,tastemolnet_gine}.yaml`;
- data and multiclass semantics: `src/data/dataset_registry.py`,
  `src/data/molecular_graph_{featurizer,dataset}.py`,
  `src/eval/counterfactual_semantics.py`, and
  `scripts/prepare_tastemolnet.py`;
- generic GNN/oracle: `src/models/{molecular_gnn,gnn_backbone_registry}.py`,
  `src/oracles/`, `scripts/train_molecular_gnn.py`,
  `scripts/evaluate_molecular_gnn.py`, and
  `scripts/calibrate_gnn_classifier.py`;
- AutoDL control plane: `src/utils/autodl_runtime.py`, `scripts/autodl/`, and
  their thin paired Slurm wrappers;
- provenance guard, focused tests, and route documentation.

Release `8b9628c` additionally hardened the persistent control root, graph
cache, exact BACE B4/B5 stages, checkpoint invariants, launch contracts, and
their paired wrappers/tests. Its exact 32-file release delta is recoverable
with:

```bash
git diff --name-status 861ba55..8b9628c
```

Verification for the deployed release:

- local focused release gate: `84 passed`;
- AutoDL Linux focused release gate: `84 passed`;
- B6 route focused gate at `8b17fb1`: local `26 passed`, AutoDL Linux
  `26 passed`;
- one earlier repository-wide diagnostic found `18 failed, 1557 passed,
  10 skipped, 1 error`; those failures were pre-existing data, checkout, or
  platform-collection issues and were not repeatedly rerun.

## 4. BBBP replacement scope

The active four-dataset matrix is AIDS, Mutagenicity, BACE, and TasteMolNet.
BBBP was removed only from the active dataset registry, active configs, new
automation, paper-facing active dataset table, and future formal experiment
matrix. AIDS and Mutagenicity were unchanged and remained read-only.

At migration base there was no tracked active runtime BBBP registry; tracked
BBBP mentions were historical. The one-time migration audit passed at:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_tastemolnet_gnn_migration_20260822_0119_v2`

The earlier sibling directory ending in `_0119` is an incomplete diagnostic
attempt caused by `rg` being absent in the AutoDL image. No scientific stage
used that incomplete directory.

## 5. BBBP historical preservation

No BBBP data, checkpoint, log, output, paper draft, or Git record was deleted
or rewritten. Historical tracked material remains in Git history;
pre-existing runtime artifacts remain at their original paths. The exact
repository references observed during migration are frozen in:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_tastemolnet_gnn_migration_20260822_0119_v2/bbbp_reference_inventory.csv`

There is no invented consolidated BBBP directory. The inventory above is the
source of truth for historical locations.

## 6. TasteMolNet source, hash, and license

- Public source repository:
  `https://github.com/MujeebOnawole/Taste_Prediction_RGCN`.
- Fixed upstream commit:
  `16af8ead8a17b6bd3941d9eb5879c5be75c14114`.
- Selected file: `processed_data/taste_scaffold_split.csv`.
- One-time fixed-source CSV SHA-256:
  `b7308b3277fd07ed6af4b861c0d2ce2d843f92cc81a9e5e4efd65cf4040a291b`.
- AutoDL upstream copy:
  `/autodl-fs/data/counterfactual-subgraph-runtime/data/tastemolnet/upstream/16af8ead8a17b6bd3941d9eb5879c5be75c14114/taste_scaffold_split.csv`.
- Prepared foundation:
  `/autodl-fs/data/counterfactual-subgraph-runtime/data/tastemolnet/prepared/16af8ead8a17b6bd3941d9eb5879c5be75c14114`.
- Data marker: `LICENSE_REVIEW_REQUIRED`.
- Route status: `BLOCKED_LICENSE_REVIEW`.

The upstream repository and CSV do not provide a standalone explicit data
license. Source and derived rows remain untracked. Preparation, graph cache,
tests, and a bounded CPU smoke were allowed; full TasteMolNet training and
publication of derived data remain fail-closed.

## 7. TasteMolNet cleaning and split statistics

- Input rows: `14,158`.
- Input labels: Bitter `3,165`, Sweet `6,085`, Tasteless `4,908`.
- Upstream groups: train `11,330`, validation `1,415`, test `1,413`; these are
  provenance groups, not the project four-way split.
- Retained conflict-free supported molecules: `13,421`.
- Excluded during canonicalization/support/conflict filtering: `737`.
- Project scaffold-disjoint split (seed 7): train `9,437`, validation `1,328`,
  calibration `1,328`, test `1,328`.
- All three labels occur in every project split.
- Cross-split scaffold-overlap gate: `PASS`.
- Molecular graph cache: `13,421` graphs at
  `/autodl-fs/data/counterfactual-subgraph-runtime/cache/tastemolnet/16af8ead8a17b6bd3941d9eb5879c5be75c14114/molecular_graph_v1`.

Exact per-file hashes and detailed exclusion/component counts remain in the
prepared manifest/provenance; they are not replaced by prose here.

## 8. BACE split and source label

- Source split root:
  `/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/data/processed/BACE`.
- Frozen split sizes: train `959`, validation `187`, calibration `129`, test
  `238`; total `1,513` unique valid molecules.
- Scaffold overlap between every split pair: zero.
- Source label: `1` (`Active`); strict destination is untargeted class `0`.
- Molecular graph cache: `1,513` graphs at
  `/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/scaffold_v1/molecular_graph_v1`.
- Validation and test remained unchanged. Validation selected the classifier
  and fitted temperature; calibration is reserved for downstream rules; test
  was not loaded or evaluated by B2--B5.

## 9. BACE GINE configuration

- Oracle/classifier: task-specific frozen GNN; `oracle_backend=gnn`,
  `classifier_type=gnn`, `rf_oracle_used=false`.
- Primary backbone: five-layer GINE, hidden dimension 256, mean pooling,
  residual connections, batch normalization, dropout 0.2, and a two-layer
  readout.
- Output classes: 2.
- Optimization: class-weighted cross entropy, AdamW, learning rate `1e-3`,
  weight decay `1e-5`, batch size 64, gradient clipping 5.0, at most 200
  epochs, early-stopping patience 20, seed 7.
- Checkpoint selection: validation ROC-AUC.
- Full health gate: validation ROC-AUC at least 0.65, more than one predicted
  class, positive source-class recall, and finite probabilities/metrics.
- Registered sensitivity backbones: GIN, GCN, and GATv2; none is presented as
  the primary BACE result.

## 10. BACE classifier metrics

B3 completed with a passing validation-only health gate:

| Metric | Value |
|---|---|
| run ID | `20260821T180839Z-bace-B3_GNN_FULL-95434` |
| best epoch | `44` |
| validation ROC-AUC | `0.9010548039440496` |
| health gate | `PASS` |
| checkpoint identity | `4edd23cd...1bc47` (abbreviated; full identity is in the frozen model card/manifest) |
| test loaded | `false` |
| test evaluated | `false` |

Other exact validation fields remain in `training_metrics.json`; they are not
reconstructed from rounded console output. No held-out test metric is claimed.

## 11. Calibration metrics

B4 copied the B3 bundle to a fresh persistent directory, calibrated on
`val.csv` only, and passed without loading test:

| Metric | Before | After |
|---|---:|---:|
| temperature | — | `1.5447202081060156` |
| validation NLL | `0.47404` | `0.44296` |
| validation ECE | `0.15040` | `0.04744` |
| stage/gate | — | `PASS` |
| test loaded/evaluated | — | `false / false` |

The unrounded authoritative values and argmax-invariance evidence are in
`b4_calibration.json`; the B3 bundle remains unchanged.

## 12. RF provenance gate

The gate is fail-closed for BACE and TasteMolNet:

```text
oracle_backend = gnn
classifier_type = gnn
rf_oracle_used = false
```

The audit classifies the legacy BACE Morgan-RF teacher and every candidate,
verification, selector, or final artifact bound to it as `RF_CONTAMINATED`.
A legacy GCFExplainer checkpoint without a complete task-specific GINE model
card is `UNKNOWN_ORACLE_PROVENANCE`; MolCLR is `ORACLE_NEUTRAL` because it is
only the WNode distance encoder. Exact classifications are in the v2 audit's
`bace_existing_artifacts.csv` and `bace_oracle_provenance.csv`.

The B5 runtime RF guard returned `true`. B5 also kept
`test_loaded=false`. No RF artifact entered B0--B6.

## 13. Current GPU allocation

There is no current BACE or TasteMolNet GPU allocation: all B0--B6 and Taste
CPU-smoke workers completed or reached their terminal gate and released their
resources.

| Route/stage | Allocation | State |
|---|---|---|
| BACE B2, B3, B5 (historical) | GPU 0, `GPU-0e4e08dd-f7cc-da83-c0f6-a663440c0732` | completed `PASS`; lock released |
| BACE B6 diagnostic (historical) | GPU 0, same UUID | terminal `BLOCKED`; lock released |
| BACE B4 | CPU | completed `PASS` |
| TasteMolNet CPU smoke | CPU | completed `PASS` |
| TasteMolNet heavy route | none | blocked; not launched |
| Mutagenicity recovery | independent/read-only | not resampled or changed by this route |
| AIDS recovery | independent/read-only | not resampled or changed by this route |

The historical BACE launcher used the stable-idle/UUID lock policy and did not
kill or displace existing work.

## 14. AutoDL PIDs and tmux sessions

All PIDs below are historical completed processes, not current allocations.
No active BACE or TasteMolNet tmux session is required now.

| Stage | Run ID | Historical PID / child PID | Current state |
|---|---|---|---|
| B2 | `20260821T180530Z-bace-B2_GNN_SMOKE-94533` | `94627 / 94632` | `PASS`, completed |
| B3 | `20260821T180839Z-bace-B3_GNN_FULL-95434` | `95529 / 95536` | `PASS`, completed |
| B4 | `20260821T181040Z-bace-B4_GNN_CALIBRATED-97689` | `97699 / 97702` | `PASS`, completed |
| B5 | `20260821T181237Z-bace-B5_ORACLE_SMOKE-97877` | `97969 / 97974` | `PASS`, completed |
| B6 | `20260821T183322Z-bace-B6_PPO_SMOKE-98865` | `98965 / 98970` | `BLOCKED`, exit 78, completed |
| Taste CPU smoke | `20260821T180648Z-tastemolnet-GNN_CPU_SMOKE-94932` | `94942 / 94943` | `PASS`, completed |

Mutagenicity and AIDS belong to their independent recovery controller. This
route deliberately did not refresh, stop, or overwrite their PID/session
records; use that controller's handoff for their live identities.

## 15. BACE stage status

| Stage | Formal state | Evidence |
|---|---|---|
| B0_AUDIT | `PASS` | marked `2026-08-21T18:03:09Z`; v2 migration audit |
| B1_DATA_READY | `PASS` | marked `2026-08-21T18:04:07Z`; frozen split and 1,513-graph cache |
| B2_GNN_SMOKE | `PASS` | run `20260821T180530Z-bace-B2_GNN_SMOKE-94533` |
| B3_GNN_FULL | `PASS` | run `20260821T180839Z-bace-B3_GNN_FULL-95434` |
| B4_GNN_CALIBRATED | `PASS` | run `20260821T181040Z-bace-B4_GNN_CALIBRATED-97689` |
| B5_ORACLE_SMOKE | `PASS` | run `20260821T181237Z-bace-B5_ORACLE_SMOKE-97877` |
| B6_PPO_SMOKE | `BLOCKED` | run `20260821T183322Z-bace-B6_PPO_SMOKE-98865`; scoring diagnostic PASS, PPO not performed |
| B7_PPO_FULL | `NOT_STARTED` | B6 PASS and real GNN-PPO manifest required |
| B8_POOL_BASE | `NOT_STARTED` | downstream-gated |
| B9_POOL_HIGHTEMP | `NOT_STARTED` | downstream-gated |
| B10_POOL_MERGED | `NOT_STARTED` | downstream-gated |
| B11_CROSS_PARENT_VERIFIED | `NOT_STARTED` | downstream-gated |
| B12_SELECTOR | `NOT_STARTED` | downstream-gated |
| B13_FINAL_EVAL | `NOT_STARTED` | downstream-gated |
| B14_FROZEN | `NOT_STARTED` | downstream-gated |

B5 selected exactly 16 correctly predicted source-class parents. All 16 had
valid real deletion evidence, producing 64 deletion records. Batch/single
maximum probability difference was `5.14e-08`; the RF guard passed and test was
not loaded.

B6 loaded the calibrated GNN once and scored 32 bounded candidates. The
diagnostic found 6 strict flips, finite probabilities, and batch/single maximum
difference `9.2241e-08`; `test_loaded=false`, `rf_oracle_used=false`. It did
not train PPO (`ppo_training_performed=false`, `ppo_update_count=0`) and did not
claim a PPO pass. Its primary blocker is
`BLOCKED_MISSING_GNN_PPO_INTEGRATION`; the secondary blocker is
`BLOCKED_NO_GNN_CLEAN_BACE_POLICY_INITIALIZATION`. Therefore downstream release
is false and B7 was not launched.

## 16. BACE checkpoint and output paths

- Persistent runtime root:
  `/autodl-fs/data/counterfactual-subgraph-runtime`.
- Persistent control root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control`.
- B0--B5 fast execution clone:
  `/root/autodl-tmp/worktrees/bace-tastemolnet-gnn-autodl-861ba55`; despite the
  directory suffix, its deployed HEAD for those runs was `8b9628c`.
- B6 execution clone:
  `/root/autodl-tmp/worktrees/bace-tastemolnet-gnn-autodl-8b17fb1`, exact HEAD
  `8b17fb1096666852b0680f899073dd82f207cce1`.
- B2 smoke bundle:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/smoke-20260821T180529Z-94533`.
- B3 uncalibrated full bundle:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/full-20260821T180836Z-95434`.
- B4 calibrated frozen-GNN bundle:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689`.
- B5 oracle-smoke evidence:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/oracle-smoke-20260821T181237Z-97877`.
- B6 diagnostic/blocker evidence:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/bace/ours_gnn_gine/b6-ppo-smoke-20260821T183221Z-98865`.
- BACE final output: not produced; B14 is `NOT_STARTED`.

All state, checkpoints, outputs, and logs are persistent under
`/autodl-fs/data`; the fast NVMe clone is an execution copy only.

## 17. TasteMolNet `READY_NOT_RUN` status

- Prepared rows and cached graphs: `13,421 / 13,421`.
- CPU smoke run:
  `20260821T180648Z-tastemolnet-GNN_CPU_SMOKE-94932`.
- CPU smoke output:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/cpu-smoke-20260821T180646Z-94932`.
- CPU smoke state: `PASS`; `num_classes=3`; test was not loaded/evaluated.
- Heavy switch: `RUN_TASTEMOLNET=0`.
- Heavy GNN, PPO, candidate pools, verification, selectors, and full baselines:
  not launched.
- Authoritative route status: `BLOCKED_LICENSE_REVIEW`.

The data and bounded foundation are technically ready, but the route does not
claim `[TASTEMOLNET_FOUNDATION_READY_NOT_RUN]` while reuse terms remain
unresolved. No historical result is relabeled.

## 18. Exact status and resume commands

Read-only status command for the persistent BACE/Taste control plane:

```bash
AUTODL_DATA_ROOT=/autodl-fs/data \
AUTODL_CONTROL_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control \
AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python \
RUN_TASTEMOLNET=0 \
PYTHONPATH=/root/autodl-tmp/worktrees/bace-tastemolnet-gnn-autodl-8b17fb1 \
/root/miniconda3/envs/smiles_pip118/bin/python \
/root/autodl-tmp/worktrees/bace-tastemolnet-gnn-autodl-8b17fb1/scripts/autodl/status.py \
  --project-root /root/autodl-tmp/worktrees/bace-tastemolnet-gnn-autodl-8b17fb1 \
  --data-root /autodl-fs/data --format table --gpu --limit 20
```

There is no running process to resume. Do not rerun B0--B6. There is no safe
launcher for B7 yet: first implement and review both a provenance-clean BACE
policy initializer and a GNN-backed PPO reward/training adapter, then replace
the B6 diagnostic with a real PPO smoke that produces at least one update and
a frozen GNN-reward manifest. Until then, the exact resume command is `NONE`.

## 19. Incomplete items and blockers

- B6 is terminal `BLOCKED`; its GNN scoring is diagnostic evidence, not PPO
  training. B7--B14 remain `NOT_STARTED` behind immutable dependency gates.
- BACE final output does not exist because B14 has not run.
- TasteMolNet heavy work is blocked by `RUN_TASTEMOLNET=0` and
  `BLOCKED_LICENSE_REVIEW`.
- The five paper-facing edits remain inside a pre-existing untracked paper
  tree and require separate review; do not bulk-stage that tree.
- The earlier repository-wide diagnostic failures are pre-existing
  environment/data issues; they do not justify repeated full-suite runs or any
  AIDS/Mutagenicity change.

## 20. Next minimum action

1. Implement a provenance-clean BACE policy initializer and GNN-backed PPO
   reward/training adapter without changing the frozen B0--B5 artifacts.
2. Add a real B6 PPO smoke contract with at least one optimizer update and a
   GNN reward manifest; only a reviewed B6 `PASS` may release B7.
3. Keep `RUN_TASTEMOLNET=0`; do not launch TasteMolNet heavy work until the
   data license is resolved explicitly.

## Final machine-readable handoff fields

```text
current_stage=B6_PPO_SMOKE_BLOCKED__B7_NOT_STARTED
current_pid=none__B6_launcher_and_child_exited
tmux_session=none_active_for_BACE_or_TasteMolNet
assigned_gpus=none_current__B6_GPU0_lock_released__historical_UUID_GPU-0e4e08dd-f7cc-da83-c0f6-a663440c0732
bace_gnn_checkpoint=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689
bace_final_output=NOT_PRODUCED__B14_NOT_STARTED
tastemolnet_foundation=CPU_SMOKE_PASS__BLOCKED_LICENSE_REVIEW__HEAVY_NOT_LAUNCHED
handoff_path=docs/BACE_TASTEMOLNET_GNN_AUTODL_HANDOFF.md
resume_command=NONE__IMPLEMENT_CLEAN_BACE_POLICY_INITIALIZATION_AND_GNN_PPO_ADAPTER_FIRST
status_command=see_section_18_exact_read_only_command
```
