# AutoDL main-table and ablation handoff (v1)

Last live audit: 2026-09-03 00:47--01:02 CST.  Re-run the status commands below;
the values in this note are a handoff snapshot, not a substitute for the unique
matrix authority.

## Priority and authority

The immutable priority is `MAIN_16_OF_16 > LLM_ABLATION > GNN_ABLATION`.
The only matrix authority is:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json
```

At the audit time it reported 12/16.  The missing cells were
`Mutagenicity/ComRecGC`, `TasteMolNet/GCFExplainer`,
`TasteMolNet/GlobalGCE`, and `TasteMolNet/ComRecGC`.

## Live main owners

- Mut continuation: PID 104363, start ticks 10141317, state
  `PROTECTED_BASELINE_RUNNING`, robust-v2 1800-second baseline.  It must remain
  the only Mut owner.  It will run trace-on and trace-off sequentially and does
  not rebuild the historical 50k, pair store, or DBSCAN unless adoption fails.
- Taste GlobalGCE recovery: manager PID 82588 and science PID 82680, start
  ticks 7319071, GPU1.  This is the sole seed-7/100-epoch recovery and must not
  be restarted.  The valid-zero fallback is permitted only after both targets
  finish, no writer remains, and typed chemistry replay proves zero valid
  rules with no engineering error.
- Taste GCF T12: PID 66459, start ticks 5586300, GPU3.  It has no early
  checkpoint suitable for a restart; observe PID and output growth only.
- Taste ComRecGC T14: PID 7224, start ticks 361595, GPU2.  The latest audit saw
  step 11000 and committed checkpoints through 10000.  The external auditor
  must never open the active SQLite writer and first becomes numerically useful
  when the committed 12500 checkpoint exists.

All four owners are protected.  A missing cell whose exact PID/start ticks no
longer match is conservatively treated as `READY_WAITING_GPU`; this blocks
early LLM admission until the relevant publisher appends the cell.

## LLM proposer ablation

The actual BACE/Ours path is `BASE_PLUS_PPO_LORA`.  There is no independent
project SFT checkpoint.  The four core rows are:

```text
BRICS_FIXED
CHEMLLM_7B_OFF_THE_SHELF
CHEMLLM_7B_PPO_MAIN
CHEMLLM_2B_OFF_THE_SHELF
```

The 7B PPO row is adoption-only.  The scale claim is limited to 2B versus 7B
off-the-shelf under the same proposal budget and downstream evaluation.  The
matched-SFT study stays disabled and must be reported as not applicable to the
current main pipeline.

The train-only BRICS vocabulary has 472 entries.  Its candidate pool and
shortfall receipt are already present under the BACE stage-v2 output root.
The 2B snapshot is pinned at revision
`215c0dbc89417a06bbc3bae43a3ad61e58f0a56e`; it contains 1,889,110,016
parameters by safetensors headers.  The isolated-load report is still required
before the 2B science row can run.

Early LLM science may use at most one GPU only after matrix >=13, Mut no longer
needs GPU0, every remaining main cell has a healthy owner, no main task is
waiting for a GPU, the target has runtime evidence and checkpoint/resume, and
one GPU has been idle for 1200 seconds.  A new main GPU waiter pauses the LLM
run at its next committed stage boundary.

## GNN proposal-fixed ablation

The five rows are `gine,gin,gcn,gatv2,gps`.  GraphGPS uses PyG `GPSConv`, local
GINE, four-head global attention, five layers, dropout 0.2, mean pooling, and
topology-only random-walk PE length 16.  Parameter-only dry-run selected hidden
dimension 160: 1,608,327 parameters versus the reloaded GINE's 1,432,583
(12.2676% difference).  No validation/test metric selected this width.

GNN science remains blocked until matrix 16/16 and final matrix audit, Figure
3, Figure 4, and Table 2 receipts are all PASS.  Seed 7 runs first with at most
two GPUs; seeds 17 and 27 are extensions only when the measured per-model ETA
is no more than two hours.  Graph-Mamba is pinned metadata only and never runs
under this controller.

## Status commands

After deployment, use the immutable execution worktree recorded in the
controller receipt:

```bash
scripts/autodl/launch_main_and_ablations_v1.sh status
python scripts/autodl/status_llm_ablation_core_v1.py --help
scripts/autodl/status_gnn_five_backbone_ablation_v1.py --help
```

The active science PIDs must be checked by exact PID plus `/proc/<pid>/stat`
start ticks.  Do not infer ownership from fuzzy command matching, and do not
query the active T12/T14/T8 SQLite files.
