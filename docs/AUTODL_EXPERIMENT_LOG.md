# AutoDL experiment log

This file is the human-readable companion to
`outputs/autodl/experiment_registry/runs.jsonl`. Runtime entries are appended by
the AutoDL runner and must record the run ID, dataset, stage, exact Git commit,
GPU index/UUID, command, config identity, expected output, PID/session, state,
and exit code.

## 2026-08-22 — BACE/TasteMolNet frozen-GNN foundation

- Branch: `feat/bace-tastemolnet-gnn-autodl`.
- BACE legacy RF artifacts classified as historical/contaminated.
- BACE frozen scaffold split selected as the only new-route data split.
- Taste upstream source fixed to `MujeebOnawole/Taste_Prediction_RGCN` commit
  `16af8ead8a17b6bd3941d9eb5879c5be75c14114`.
- Taste data license status: `LICENSE_REVIEW_REQUIRED`; data remains untracked.
- `RUN_TASTEMOLNET=0`; no heavy TasteMolNet task authorized.
- No HPC resource is used by this route.

### AutoDL foundation execution

- One-time migration audit: `PASS` at
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_tastemolnet_gnn_migration_20260822_0119_v2`.
  The earlier `_0119` directory is an incomplete diagnostic attempt caused by
  the AutoDL image not providing `rg`; no scientific stage used it.
- Upstream Taste CSV was privately materialized from the fixed commit above and
  prepared at
  `/autodl-fs/data/counterfactual-subgraph-runtime/data/tastemolnet/prepared/16af8ead8a17b6bd3941d9eb5879c5be75c14114`.
- Preparation consumed 14,158 rows and retained 13,421 conflict-free supported
  molecules.  The train/validation/calibration/test sizes are
  9,437/1,328/1,328/1,328, all three labels occur in every split, and the
  scaffold-overlap gate passed.
- The upstream repository and CSV do not carry an explicit data license.
  `LICENSE_REVIEW_REQUIRED` therefore remains authoritative: foundation and a
  bounded CPU smoke are allowed, but full TasteMolNet training and publication
  of derived data remain disabled.
- The AutoDL Linux Frozen-GNN core/Taste preparation gate passed 16 targeted
  tests.  Full-suite reruns are intentionally avoided; the frozen local release
  gate already covered 459 tests (9 platform skips).
