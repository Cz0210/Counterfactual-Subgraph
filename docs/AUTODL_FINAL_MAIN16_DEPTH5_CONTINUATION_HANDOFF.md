# Final main16 depth-5 continuation handoff

Updated 2026-09-04 01:29 CST.  The only matrix authority is
`/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority`;
it reports 12/16.  Missing cells are Mutagenicity/ComRecGC and TasteMolNet
GCFExplainer, GlobalGCE, and ComRecGC.  LLM/GNN ablations remain disabled.

## HPC T8 chain

- Historical canary 2536033: TIMEOUT; follow-up 2536034: COMPLETED.
- Current depth-5 canary 2536148: RUNNING at the final snapshot; its only legal
  follow-up is 2536149 with `afterany:2536148`.
- Stable pointer:
  `/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/control/t8-refinement-chain/current.json`.
- Do not submit another follow-up.  Let 2536149 either admit the full array or
  deterministically refine the current prefix.
- Read-only status:
  `scripts/local/status_hpc_autodl_offload.sh` from commit
  `02c8e032593e19893f7562ae9b9a8aa7ea72c3f0`.

## AutoDL owners

- Mut fresh successor: owner 165790, worker 165793, robust-v2, CPUs 1/2,
  fresh-vs-fresh.  Root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut-clean-successor-9c17bf4d-9c17bf4d-d756-4a16-a41a-bf63d6d0eed1`.
- T12 reference owner 162844 / science 162847 remains healthy on GPU3.  Its
  heartbeat counter is a segment boundary; use first-seen records for progress.
- Old AutoDL T8 controller 82588 / science 82680 remains protected on GPU1.
- T14 has no science owner.  Route A is rejected: its sealed audit has zero
  qualifying convergence windows, and lightweight artifacts cannot close
  postprocess.  Route B is unavailable because the cgroup-v1 memory mount is
  read-only and the 480-GiB hard limit is below the historical restore gate.
  Route C must therefore start fresh on GPU2; never load the legacy 12.5k
  `data.pkl`.

## Git

Feature branch `fix/final-main16-depth5-continuation-20260904` contains the
dynamic HPC chain pointer and Mut successor.  AutoDL immutable worktree:
`/root/autodl-tmp/worktrees/final-main16-02c8e032`.

Do not restart the healthy Mut/T12/T8 owners.  End a status turn instead of
polling any hour-scale job when no further code or launch work is ready.
