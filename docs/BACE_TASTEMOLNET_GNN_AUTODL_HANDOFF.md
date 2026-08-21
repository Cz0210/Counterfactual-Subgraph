# BACE / TasteMolNet Frozen-GNN AutoDL handoff

Date: 2026-08-22

Scope: AutoDL only; no HPC GPU or CPU was used for this route.

Draft status: final release deployment is still in progress. Every dynamic
field that depends on that deployment is deliberately recorded as
`PENDING_FINAL_DEPLOY`; it must be replaced from the persistent AutoDL state
exactly once before this handoff is declared final.

## 1. Git branch and commit

- Branch: `feat/bace-tastemolnet-gnn-autodl`.
- Foundation commit currently deployed/audited:
  `861ba55179ef3107b06e8634a98e1070426dfc4a`.
- Logical commits through that foundation:
  - `3bc8bb4` — TasteMolNet data and multiclass semantics;
  - `b3dcc30` — generic frozen molecular-GNN oracle;
  - `80c068b` — AutoDL frozen-GNN scheduler;
  - `861ba55` — BACE/TasteMolNet Frozen-GNN route documentation.
- Final release commit: `PENDING_FINAL_DEPLOY`.
- The final handoff must record one exact final commit after the current
  B4/B5, persistent-control, graph-cache, and launcher changes are committed
  and deployed. Do not repeatedly poll or re-verify Git SHA while experiments
  are running.

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

The changes currently present after `861ba55` in the feature worktree are task
changes, not pre-existing user changes. Their final file list belongs to the
pending final release commit.

## 3. Files added or modified by this route

The commits through `861ba55` add or update the following coherent groups:

- active registry/configuration: `README.md`, `configs/datasets/{bace_gnn,tastemolnet}.yaml`,
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
- AutoDL control plane: `src/utils/autodl_runtime.py` and
  `scripts/autodl/{bootstrap_env,common,detect_runtime,exp_run,gpu_inventory,gpu_lock,status}.py/.sh`
  plus the BACE/Taste launchers;
- provenance guard: `scripts/audit_oracle_provenance.py`;
- paired CLI wrappers under `scripts/slurm/`;
- focused registry, data, oracle, semantics, provenance, scheduler, and vertical
  smoke tests under `tests/` and `tests/autodl/`;
- route documentation: `README.md`, `docs/cf_subgraph_v3_spec.md`,
  `docs/decisions.md`, `docs/refactor_plan.md`,
  `docs/AUTODL_EXPERIMENT_LOG.md`,
  `docs/BACE_TASTEMOLNET_GNN_AUTODL.md`,
  `docs/BBBP_TO_TASTEMOLNET_MIGRATION.md`, and
  `docs/TASTEMOLNET_MULTICLASS_BASELINE_AUDIT.md`.

The pending final release additionally includes the persistent control-root
contract, molecular graph cache, explicit BACE B4/B5 stage entrypoints, their
paired Slurm wrappers, and focused tests. The exact final diff is
`PENDING_FINAL_DEPLOY` and must be captured from the final commit rather than
from a moving worktree.

Verification already completed before the pending final deploy:

- local scoped release gate: `459 passed, 9 skipped` with one harmless joblib
  warning;
- AutoDL Linux targeted GNN/Taste gate: `16 passed`;
- one local full-suite diagnostic found `18 failed, 1557 passed, 10 skipped,
  1 error`; those failures were pre-existing data/checkout/platform collection
  issues and were not repeatedly rerun.

## 4. BBBP replacement scope

The active four-dataset matrix is now intended to be AIDS, Mutagenicity, BACE,
and TasteMolNet. BBBP is removed only from the active dataset registry, active
configs, new automation, paper-facing active dataset table, and future formal
experiment matrix. AIDS and Mutagenicity are unchanged and remain read-only.

At migration base there was no tracked active runtime BBBP registry; tracked
BBBP mentions were historical. The one-time migration audit passed at:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_tastemolnet_gnn_migration_20260822_0119_v2`

The earlier sibling directory ending in `_0119` is an incomplete diagnostic
attempt caused by `rg` being absent in the AutoDL image. No scientific stage
used that incomplete directory.

## 5. BBBP historical preservation

No BBBP data, checkpoint, log, output, paper draft, or Git record was deleted
or rewritten. Historical tracked material remains in repository Git history;
pre-existing runtime artifacts remain at their original data/output/log paths.
The exact repository references observed during the migration are frozen in:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_tastemolnet_gnn_migration_20260822_0119_v2/bbbp_reference_inventory.csv`

There is no newly invented consolidated BBBP directory. The inventory above
is the source of truth for historical locations.

## 6. TasteMolNet source, hash, and license

- Public source repository:
  `https://github.com/MujeebOnawole/Taste_Prediction_RGCN`.
- Fixed upstream commit:
  `16af8ead8a17b6bd3941d9eb5879c5be75c14114`.
- Selected upstream file: `processed_data/taste_scaffold_split.csv`.
- One-time SHA-256 of the fixed-commit source CSV:
  `b7308b3277fd07ed6af4b861c0d2ce2d843f92cc81a9e5e4efd65cf4040a291b`.
- AutoDL upstream copy:
  `/autodl-fs/data/counterfactual-subgraph-runtime/data/tastemolnet/upstream/16af8ead8a17b6bd3941d9eb5879c5be75c14114/taste_scaffold_split.csv`.
- Prepared foundation:
  `/autodl-fs/data/counterfactual-subgraph-runtime/data/tastemolnet/prepared/16af8ead8a17b6bd3941d9eb5879c5be75c14114`.
- License status: `LICENSE_REVIEW_REQUIRED`.

The upstream repository and CSV do not provide a standalone explicit data
license. The CSV and derived records therefore remain untracked. Preparation,
graph foundation, tests, and a bounded forward smoke are allowed, but full
TasteMolNet training and publication of derived data remain fail-closed until
reuse terms are confirmed.

## 7. TasteMolNet cleaning and split statistics

- Input rows: 14,158.
- Input labels: Bitter `3,165`, Sweet `6,085`, Tasteless `4,908`.
- Upstream groups: train `11,330`, validation `1,415`, test `1,413`; these
  groups are provenance only, not the project four-way split.
- Retained conflict-free supported molecules: `13,421`.
- Excluded during canonicalization/support/conflict filtering: `737`.
- Project scaffold-disjoint split (seed 7):
  - train: `9,437`;
  - validation: `1,328`;
  - calibration: `1,328`;
  - test: `1,328`.
- All three labels occur in every project split.
- Cross-split scaffold overlap gate: `PASS`.
- Exact per-file hashes and detailed exclusion/component counts remain in the
  prepared directory's manifest/provenance artifacts; do not replace those
  artifacts with numbers copied from this prose.

## 8. BACE split and source label

- Source split root:
  `/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/data/processed/BACE`.
- Frozen split sizes: train `959`, validation `187`, calibration `129`, test
  `238`; total `1,513` unique valid molecules.
- Scaffold overlap between every split pair: zero.
- Source label: `1` (`Active`). Destination is untargeted class `0`.
- Validation and test remain unchanged. Validation selects the classifier and
  fits temperature; calibration is reserved for downstream thresholds/rules;
  test is held out until final frozen evaluation.

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
- BACE checkpoint selection override: validation ROC-AUC.
- Full health gate: validation ROC-AUC at least 0.65, more than one predicted
  class, positive source-class recall, and finite probabilities/metrics.
- Alternative registered sensitivity backbones: GIN, GCN, and GATv2. They are
  not primary BACE results.

## 10. BACE classifier metrics

No formal B3 metric is claimed in this draft.

| Metric | Value |
|---|---|
| selected epoch | `PENDING_FINAL_DEPLOY` |
| validation ROC-AUC | `PENDING_FINAL_DEPLOY` |
| validation accuracy | `PENDING_FINAL_DEPLOY` |
| validation macro-F1 | `PENDING_FINAL_DEPLOY` |
| validation source-class recall | `PENDING_FINAL_DEPLOY` |
| predicted-class support | `PENDING_FINAL_DEPLOY` |
| health gate | `PENDING_FINAL_DEPLOY` |

Test metrics must not be added here until B4/B5 and all downstream choices are
frozen.

## 11. Calibration metrics

Planned B4 calibration uses temperature scaling on `val.csv` only, copies the
uncalibrated B3 bundle to a fresh persistent directory, and must leave B3
unchanged. Formal evidence is pending.

| Metric | Before | After |
|---|---:|---:|
| temperature | — | `PENDING_FINAL_DEPLOY` |
| validation NLL | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| validation ECE | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| validation Brier | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| argmax invariant | — | `PENDING_FINAL_DEPLOY` |

B4 state/gate: `PENDING_FINAL_DEPLOY`.

## 12. RF provenance gate

The gate is fail-closed for BACE and TasteMolNet:

```text
oracle_backend = gnn
classifier_type = gnn
rf_oracle_used = false
```

The audit classifies the legacy BACE Morgan-RF teacher and every candidate,
verification, selector, or final artifact bound to it as `RF_CONTAMINATED`.
They remain diagnostic history only. A legacy GCFExplainer checkpoint without a
complete task-specific GINE model card is `UNKNOWN_ORACLE_PROVENANCE` and is
also excluded. MolCLR is `ORACLE_NEUTRAL` because it is only the WNode distance
encoder. Exact classifications are recorded in the v2 audit's
`bace_existing_artifacts.csv` and `bace_oracle_provenance.csv`.

Focused RF/oracle guard tests passed locally. Formal B5 runtime guard evidence:
`PENDING_FINAL_DEPLOY`.

## 13. Current GPU allocation

| Route | GPU index | GPU UUID | State |
|---|---|---|---|
| pre-existing Mutagenicity recovery | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| pre-existing AIDS recovery | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| BACE Frozen-GNN | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| TasteMolNet heavy route | none authorized | none authorized | `RUN_TASTEMOLNET=0` |

The BACE runner may claim at most two GPUs that are stably idle for 60 seconds,
uses UUID-bound locks, and must not kill or displace existing work. A fresh
one-time `nvidia-smi` snapshot must replace the placeholders immediately before
final handoff.

## 14. AutoDL PIDs and tmux sessions

| Process/session | PID or child PID | tmux session | State |
|---|---|---|---|
| BACE controller/launcher | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| BACE active stage worker | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| Mutagenicity recovery | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| AIDS recovery | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` | `PENDING_FINAL_DEPLOY` |
| TasteMolNet heavy worker | none | none | disabled |

Do not copy old PID samples into this table. Fill it from the final persistent
registry and process snapshot after deployment.

## 15. BACE stage status

The audit/data evidence exists, but B0/B1 are not reported as formal state
machine passes until their evidence is registered by the final deployed
runner. No B0-B5 PASS is claimed in this draft.

| Stage | Formal state | Planned evidence/purpose |
|---|---|---|
| B0_AUDIT | `PENDING_FINAL_DEPLOY` | v2 migration audit |
| B1_DATA_READY | `PENDING_FINAL_DEPLOY` | frozen four-way BACE split |
| B2_GNN_SMOKE | `PENDING_FINAL_DEPLOY` | train/reload/batch-single/deletion smoke |
| B3_GNN_FULL | `PENDING_FINAL_DEPLOY` | full GINE training and health gate |
| B4_GNN_CALIBRATED | `PENDING_FINAL_DEPLOY` | validation-only temperature bundle |
| B5_ORACLE_SMOKE | `PENDING_FINAL_DEPLOY` | 16 correctly predicted source parents and real residuals |
| B6_PPO_SMOKE | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B7_PPO_FULL | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B8_POOL_BASE | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B9_POOL_HIGHTEMP | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B10_POOL_MERGED | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B11_CROSS_PARENT_VERIFIED | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B12_SELECTOR | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B13_FINAL_EVAL | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |
| B14_FROZEN | `PENDING_FINAL_DEPLOY` | downstream, not executed in this draft |

## 16. BACE checkpoint and output paths

- Persistent data/runtime root:
  `/autodl-fs/data/counterfactual-subgraph-runtime`.
- Persistent control root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control`.
- Fast code worktree used for execution:
  `/root/autodl-tmp/worktrees/bace-tastemolnet-gnn-autodl-861ba55`
  (the final deployed worktree/commit must replace this if it changes).
- B2 smoke output: `PENDING_FINAL_DEPLOY`.
- B3 uncalibrated checkpoint bundle: `PENDING_FINAL_DEPLOY`.
- B4 calibrated checkpoint bundle: `PENDING_FINAL_DEPLOY`.
- B5 oracle-smoke evidence: `PENDING_FINAL_DEPLOY`.
- BACE final output: `PENDING_FINAL_DEPLOY`.

All scientific state, checkpoints, outputs, and logs must remain on
`/autodl-fs/data`; the fast NVMe clone is an execution copy only.

## 17. TasteMolNet `READY_NOT_RUN` status

- Data foundation: prepared at the fixed upstream commit.
- Multiclass semantics/config/tests: implemented.
- Heavy-run switch: `RUN_TASTEMOLNET=0`.
- Heavy GNN training, PPO, candidate pools, verification, selectors, and full
  baselines: not launched.
- Authoritative blocker: `LICENSE_REVIEW_REQUIRED`.

Therefore this draft does **not** claim the
`[TASTEMOLNET_FOUNDATION_READY_NOT_RUN]` pass marker. The accurate state is
foundation prepared, heavy route disabled, license review required. After
reuse terms are confirmed, a fresh bounded foundation check may promote it to
`READY_NOT_RUN`; no historical result should be relabeled retroactively.

## 18. Exact status and resume commands

These commands depend on the final deployed commit, run ID, and persistent
state. They must not point to `861ba55` after a newer release is deployed.

```text
status_command=PENDING_FINAL_DEPLOY
resume_command=PENDING_FINAL_DEPLOY
```

Required environment contract for both commands:

```text
AUTODL_DATA_ROOT=/autodl-fs/data
AUTODL_CONTROL_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control
AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
RUN_TASTEMOLNET=0
```

The final author must paste the exact single status command and exact single
resume command that were actually exercised; do not provide a menu of guessed
commands.

## 19. Incomplete items and blockers

- Final release commit and deployment: `PENDING_FINAL_DEPLOY`.
- Formal B0-B5 stage registration/execution: `PENDING_FINAL_DEPLOY`.
- BACE classifier, calibration, PID/GPU, checkpoint, and output evidence:
  `PENDING_FINAL_DEPLOY`.
- B6-B14 were not executed in this draft and remain downstream-gated.
- TasteMolNet full training is blocked by `RUN_TASTEMOLNET=0` and
  `LICENSE_REVIEW_REQUIRED`.
- The five paper-facing edits live inside a pre-existing untracked paper tree;
  they require separate review and must not be bulk-staged accidentally.
- The repository-wide diagnostic failures described in section 3 are
  pre-existing environment/data issues; they do not justify repeatedly running
  the full suite or changing AIDS/Mutagenicity.

## 20. Next minimum action

1. Finish one final release commit, push it, and deploy that exact commit to the
   fast AutoDL execution clone while retaining persistent state under
   `/autodl-fs/data`.
2. Register the existing audit/split evidence as B0/B1, run B2 once, and launch
   B3 immediately if B2 passes. Do not wait for unrelated task lines and do not
   start TasteMolNet heavy work.
3. Once B3 is a stable long-running job, capture one status/GPU/PID snapshot,
   replace every `PENDING_FINAL_DEPLOY` field, write the exact exercised
   status/resume commands, and stop monitoring for the handoff interval.

## Final machine-readable handoff fields

```text
current_stage=PENDING_FINAL_DEPLOY
current_pid=PENDING_FINAL_DEPLOY
tmux_session=PENDING_FINAL_DEPLOY
assigned_gpus=PENDING_FINAL_DEPLOY
bace_gnn_checkpoint=PENDING_FINAL_DEPLOY
bace_final_output=PENDING_FINAL_DEPLOY
tastemolnet_foundation=LICENSE_REVIEW_REQUIRED
handoff_path=docs/BACE_TASTEMOLNET_GNN_AUTODL_HANDOFF.md
resume_command=PENDING_FINAL_DEPLOY
status_command=PENDING_FINAL_DEPLOY
```
