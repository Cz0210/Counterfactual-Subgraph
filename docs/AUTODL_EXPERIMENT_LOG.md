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
