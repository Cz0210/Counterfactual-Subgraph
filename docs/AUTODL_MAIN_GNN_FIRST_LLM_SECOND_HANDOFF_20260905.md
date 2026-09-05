# Main-table protection and BACE GNN seed7 CPU handoff

## Execution and scope

- Scientific execution commit: `532e83733971701b0709086469d2ed8955a96e25`.
- Private branch: `feat/early-gnn-first-ablation-20260905`.
- Mac development worktree: `/private/tmp/early-gnn-first-ablation-20260905`.
- HPC immutable worktree: `/share/home/u20526/czx/worktrees/bace-gnn-ablation-532e8373`.
- AutoDL input-only worktree: `/root/autodl-tmp/worktrees/bace-gnn-input-532e8373`.
- No healthy main owner was restarted, signaled, reconfigured or copied.
- No main matrix write, GPU ablation, deletion or model download was performed.
- The user's dirty Mac/HPC checkouts and external-disk recovery copies are unchanged.

## Main authority and remaining cells

Read-only observation: 2026-09-05 10:39:38 UTC / 18:39 CST.
The unique authority remains **12/16** at
`/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`.
All BACE cells and Taste Ours are registered. Four missing cells remain:

| Cell | Existing live route | Remaining gate |
|---|---|---|
| Mut ComRecGC | owner193161, science193180, post-AB193450, executor222378; trace-on231, trace-off not started | Complete same-contract sequential trace parity/reload; adopt only after evidence PASS; reuse pair store/DBSCAN |
| Taste GlobalGCE | import272454, T13 owner219876; import rchar17.28GB, positive growth | Full HPC import validation has not emitted release; T13 waits for that release |
| Taste GCF | reference owner162844/science173495; sealed250, positive output growth | Complete reference251–500 and501–510 parity before authorized fresh-zero production; diagnostic checkpoint is not production |
| Taste ComRecGC | retry2 owner268102/science268321; reference162 | Finish reference500/parity and the existing production continuation |

PID/start-ticks/command identities matched the earlier audit. T8 import has not
yet emitted its first heartbeat, so it is **active with positive I/O**, not
independently declared terminal PASS. T12 is in D state but has positive I/O;
do not infer process death or restart it. No main READY GPU waiter was observed.
GPU0=Mut; GPU1=reserved for T13 despite physical idleness; GPU2=T14; GPU3=T12.
Reliable 16/16 completion time cannot yet be derived from these incomplete
diagnostic/import gates. They are not four nearly-complete full experiments.

Local audit evidence:
`/private/tmp/final16-main-llm-preparation-audit-20260905T103938Z.json`.
Persistent free space:1,685,825,503,232 bytes; free inodes99,828;
cgroup headroom410,992,877,568 bytes; failcnt4306 unchanged.

## Real BACE input contract and transfer

- AutoDL source package root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/hpc-input-20260905T105400Z-532e8373`.
- Mac external copy:
  `/Volumes/DireRaven/counterfactual-hpc-offload/bace-gnn-seed7-20260905T105400Z/bace_gnn_hpc_input.tar.gz`.
- HPC archive:
  `/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/incoming/bace-seed7-532e8373.tar.gz`.
- HPC extracted immutable inputs:
  `/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/inputs/bace-seed7-532e8373`.
- Archive bytes:14,651,604. Payload bytes:17,835,639.
- Archive SHA:`39b0ccf488aaaece032a866f72c2150f5de78111a73be9e61153f488556d66bc`.
- Bundle self SHA:`3dca2f56501e1b32c63a6b80c59f01f2cce3be1c5541bad641153c2c394619e6`.
- Reference contract: extracted `reference_contract.json`;
  self SHA:`e2e373d2aabcd81c063eb8726373c80620484a3fbcbe84a4f32dbc2e9ae7a14a`.

The package retains the source GINE's own checksum inventory, all four frozen
BACE splits, 66 canonical rule identities, merged proposal pool, MolCLR and
its model source, exact selector/thresholds/WNode contracts. It contains no
ChemLLM, credentials, live SQLite/WAL, main matrix state or active output tree.
Train/validation alone are parsed during classifier training; calibration and
test file hashes in the provenance inventory do not authorize their use in
training. The GINE training config, not stale generic YAML, fixes AdamW,
200-epoch maximum, validation ROC-AUC checkpoint selection and validation-fitted
temperature. No calibration-trained temperature is silently substituted.

## Submitted CPU dependency chain

Recovery history: the initial preflight2557215 failed before Python because
site `/etc/bashrc` read optional `BASHRCSOURCED` under nounset. Its six exact
PENDING children2557216/17/18/19/20/22 were removed without starting science.
The failed campaign and all logs remain under the earlier
`bace-seed7-20260905T104500Z` root. Commit532e8373 corrects only shell bootstrap,
covered by five executable regression tests. The table below is the **new**
active chain; do not confuse it with those failed/cancelled historical IDs.

Transport review initially declined the second package for allegedly missing
payload/destination authorization. The user's actual attachment sections2/5
explicitly authorize these exact BACE contents and AutoDL→externaldisk→HPC
roots. After showing those passages and the45-file inventory, the same action
was approved. No permission workaround was used. The final adopted package is
the native AutoDL-built archive; an independent local execution-only repack
was diagnostic evidence only and is not the submitted input.

Campaign:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z`.

| Job | Role | Dependency |
|---|---|---|
|2557523|five-backbone real CPU forward/environment preflight|none|
|2557524|GatedGCN+ seed7 benchmark then admitted training|afterok2557523|
|2557525|GIN seed7 benchmark then admitted training|afterok2557523|
|2557526|GCN seed7 benchmark then admitted training|afterany2557524|
|2557527|GATv2 seed7 benchmark then admitted training|afterany2557525|
|2557528|five-model train-only CPU evaluation timing, then admitted native/common evaluation|afterok all four training jobs|
|2557529|independent core audit verification and result package|afterok2557528, requires real core marker|

The two training lanes guarantee at most two simultaneous training jobs.
Each uses intel,8CPU,32GiB,12h and **no GPU**. Benchmarks are the first real
epochs of the same run: pause after five epochs or1200s at a safe committed
epoch, retain optimizer/RNG/best-state, then resume exactly if measured training
ETA≤12h. An over-budget route preserves its checkpoint and emits
`READY_GNN_GPU_FALLBACK`; it does not receive unconditional AutoDL GPU access.
Inspect `submission.json` for exact fresh UUID attempt roots/commands, and each
`benchmark.json` for measured ETA. No CPU benchmark PASS is claimed merely
because submission succeeded.

HPC path admission at submission:free29,433,589,760 bytes,
reserve5,886,717,952 bytes, projected persistent allowance2,147,483,648 bytes.
No cleanup was necessary. No `/ssdfs` use was necessary.

## Backbones and scientific evaluation

GINE is adopted, never retrained. Real CPU forward on the transported input
passes on Mac for all five backbones; the separate HPC preflight job must also
pass before its first training jobs run.

| Backbone | Actual parameters |
|---|---:|
|GINE|1,432,583|
|GIN|1,432,583|
|GCN|446,978|
|GATv2|1,431,298|
|GatedGCN+|1,219,138|

GatedGCN+ source pin:`0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28`;
RWPE16; selected width160 among64/96/128/160/192/256;
relative count difference0.14899311244095456. Architecture width was selected
by parameter count, not performance. Edge handling is explicit for every model.

All candidates use the existing hard-deletion/strict-flip/MolCLR-WNode science.
The main selected variant and thresholds are fixed across classifiers.
Calibration-native and five-way-common selectors all freeze before parsing
test. Empty calibration cohorts block; empty test cohorts report N/A; nonempty
zero-flip cohorts report valid zero coverage, not fabricated rules or costs.
CPU evaluation probes use train-only rows and an explicit non-self WNode timing
probe; missing timing evidence or projected cost>12h yields the GPU fallback
state instead of full evaluation. Per-parent hash-bound checkpoints support
resume. Final audit and marker require all five native/common results.

## LLM readiness and remaining engineering

Existing BRICS CPU preparation is adopted:472 train-only vocabulary entries,
959 train molecules,386parents×8=3088 candidates,shortfall0, no oracle or test.
2B exact snapshot is already present at revision
`215c0dbc89417a06bbc3bae43a3ad61e58f0a56e` with1,889,110,016 parameters.
7B source revision:`b8b2ea19e48f53d190fe8dced94572717f8e89a2`;
base7,737,708,544 parameters, PPO LoRA18,874,368,300 updates.
The actual main chain is **base + fresh LoRA + PPO**; no independent project
SFT checkpoint exists. Core variants are BRICS,7B off-the-shelf,7B PPO-main
adoption,2B off-the-shelf. No SFT/full-factorial/20B experiment is invented.

Formal LLM GPU science remains blocked until the complete hash-verified GNN
seed7 audit plus live main-owner/idle-GPU/memory/storage/resume gates pass.
Secondary seeds17/27 are not a prerequisite. The existing model comparability
report only supports off-the-shelf proposal sensitivity and still lacks a
confirmed isolated remote-code import; resolve that loader preflight before
2B inference, without enabling unreviewed remote code. Existing model-specific
launch manifests/live owner checks remain in force. The new gate is not a
claim that a GPU worker or cross-host result-transfer watcher is running.

## Validation and commands

86 focused tests PASS; compileall, git diff --check and relevant shell syntax
checks PASS. Tests include epoch-resume equivalence, own-source GINE SHA,
common architecture preservation, five backbones, pre-test selector freeze,
empty cohorts, CPU admission, native/common evaluation and no-GPU dependencies.
No real7B/20B model was loaded by a unit test.

One-shot GNN status (do not poll continuously):

```bash
ssh tongji-hpc '/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python /share/home/u20526/czx/worktrees/bace-gnn-ablation-532e8373/scripts/hpc/gnn/status_bace_gnn_seed7.py --campaign-root /share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z'
```

Read-only main count:

```bash
ssh autodl-a800 'jq . /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
```

LLM gate (after importing the completed ablation result to a fresh local root):

```bash
python scripts/autodl/status_llm_after_gnn_v1.py --gnn-evaluation-root /path/to/verified/evaluation --main-resource-evidence /path/to/current-readonly-resource-evidence.json
```

Do not resubmit the campaign launcher while these job IDs exist. If a job
fails, inspect its exact stderr/terminal/checkpoint; do not recreate successful
classifiers or restart any main owner. If all remaining work is queued or
long-running, end the interactive turn instead of hourly polling.
