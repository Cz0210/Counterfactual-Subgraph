# GNN evaluation closeout and LLM successor — 2026-09-05

This is a continuation of the actual
`AUTODL_MAIN_GNN_FIRST_LLM_SECOND_HANDOFF_20260905.md`, not a replacement for
missing history. **The final temperature-contract review supersedes the earlier
core-PASS interpretation. No LLM science was launched.**

## Outcome and required owner decision

Main remains 12/16. All five seed7 classifiers were evaluated without retraining,
changing weights or temperatures, or regenerating the 66-rule candidate pool.
Exact native calibration parent units:288; exact test parent units:614.
Common cohorts:calibration41/test96. All ten calibration selections froze before
test. The complete scientific-row replay and package hashes passed the first
verifier, but a subsequent model-temperature review found **four `not_fit` T=1
files**, versus the adopted GINE's fitted T=1.5447202081060156. This is not a
completed validation-calibrated five-backbone scientific contract.

The old audit/package is preserved, not rewritten. Its integrity PASS must not
trigger LLM or be described as five calibrated models. Current core state is
`BLOCKED_TEMP_CALIBRATION_CONTRACT`. A strengthened verifier rejects this
case. The owner explicitly prohibited changing sealed temperatures this turn;
therefore no fitting or waiver was performed. The proposed remedy requires
permission: first-time validation-only temperature fitting for the four already
trained alternatives in fresh overlays, no GINE change/no training, followed by
an exact evaluation reconciliation. Reuse raw match/embedding/WNode computation
only when its content and numerical contract bind; do not assume probability
metrics or selectors remain unchanged.

Root cause: all four actual trained configs omit
`calibration.fit_on_validation=true`; `scripts/train_molecular_gnn.py` defaults
the missing flag to false, and `src/oracles/gnn_oracle.py` writes the exact
`not_fit/T=1` placeholder. This is not a fit whose optimum happens to equal one.
The corrective CLI `scripts/hpc/gnn/audit_bace_gnn_temperature_promotion.py`
reads the sealed archive and writes only a fresh independent review receipt;
exit 2 means this diagnosed scientific-contract blocker, not a retry request.

## Main snapshot: 2026-09-05 21:08:36 CST

Authority:
`/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`.
Matrix SHA:`fa6c85263b28e25bddf2e89e6d71f91a9fbb46667e81edd5c35af98ddce25cf8`.
No writes were made to it. Missing cells are Mutagenicity/ComRecGC and Taste
GCFExplainer, GlobalGCE and ComRecGC.

| Route | Real state | Protected PID |
|---|---|---|
|Mut|trace-on238/500, trace-off not started|owner193161/science193180; continuation193450/executor222378|
|T8 import|HPC package deep verification; no release|272454|
|T13|WAITING_HPC_IMPORT_PASS; GPU1 PREDEPLOYED reservation retained|219876|
|T12|reference250→500; last sealed250; I/O growth, not formal production|owner162844/science173495|
|T14|retry2 reference239/500; not 20k generation|owner268102/science268321|

GPU0=Mut, GPU1=T13 reservation/no CUDA worker, GPU2=T14, GPU3=T12.
All protected PID/start-tick/normalized-command identities matched.
Container memory limit515396075520bytes, usage109824237568bytes,
headroom405571837952bytes; failcnt4306 unchanged. Persistent free
1685726150656bytes,99576inodes. Host MemAvailable was not used as headroom.

Mut fallback remains a **conditional engineering blocker**, not authorization
to interrupt current A/B: its Route-B closeout writes BLOCKED_ADAPTER_MISSING;
fresh generation, new-universe pair/DBSCAN, standardized evaluation and publisher
still need concrete linkage if adoption fails. Its default100000-inode gate
also exceeds this snapshot. Current successful-adoption path remains running.

## GNN exact continuation

Original trained science:`532e83733971701b0709086469d2ed8955a96e25`.
Original publication repair:`31391b261750fd901d953d46f7769a597ad3d7e9`.
Executed exact evaluator/verifier:`fd98c5f23bf835f2b68799d03b7a2fd8b8b713f7`.
Immutable worktree:`/share/home/u20526/czx/worktrees/bace-gnn-exact-eval-fd98c5f2`.
Campaign:`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z`.
Current result suffix:`exact-parent-closeout-fd98c5f2`.

2558288/2558289 were COMPLETED but admission-only/no package. Their measured
104822-second serial projection was not core PASS. New exact chain:

2558894 train-only exact regression →2558895 calibration partition
→2558896 calibration array0–24%2 →2558897 global freeze/test partition
→2558898 test array0–39%2 →2558899 metric closeout →2558901 audit/package.

All completed. No historical Slurm FAILED status was changed. No old jobs were
cancelled. At most two heavy CPU jobs ran; every shard used the original exact
scientific functions and hash-bound per-parent checkpoints. The regression
matched both train parents on all five models, ignoring only observational
`distance_cache_hit`; actual parent JSON bytes also matched.

The package contains1160files,27013606bytes. SHA256:
`e40c9ee7a3e53f0db9635040b7fb7f09cf3fac22174444a16f743a7696e8cf63`.
HPC receipt:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z/exact-parent-closeout-fd98c5f2/verified/result_package.json`.

## Transfer and provisional tables

Mac relay immutable code:`/private/tmp/gnn-relay-ff77288a`.
Real relay PID75804, UUID`8af242e0-f773-4f8a-a23b-b884ef0df65c`.
The first shell-background PID75707 vanished before creating any relay root;
one diagnosed retry used a detached session. It did not restart any main task.

Mac archive/control:
`/Volumes/DireRaven/counterfactual-hpc-offload/gnn-seed7-closeout-fd98c5f2/8af242e0-f773-4f8a-a23b-b884ef0df65c`.
AutoDL incoming and import roots:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/seed7-import-fd98c5f2-20260905/incoming-8af242e0-f773-4f8a-a23b-b884ef0df65c`
and the same parent plus`import-8af242e0-f773-4f8a-a23b-b884ef0df65c`.
Source archives and every sealed science artifact remain preserved. The relay
does no LLM launch and no main-matrix writes. Its import is package integrity,
not an override of the temperature review above.

At 2026-09-05 21:22:41 CST the scoped relay finished normally:
`VERIFIED_PACKAGE_IMPORTED`. Both transport hops matched bytes/SHA, the fresh
AutoDL import completed, and its terminal receipt records no LLM launch or main
matrix write. The relay exits after this transfer; no permanent service was
installed. The imported historical package is still provisional under the new
temperature-contract review.

Mac provisional GNN tables/review:
`/private/tmp/gnn-seed7-paper-results-20260905`.
Do not use NLL/ECE/Brier as a matched five-calibrated-backbone comparison yet.
The result's scope is PROPOSAL_FIXED_BACKBONE_SENSITIVITY, not end-to-end.

## LLM preparation

Executed preparation code:`ff77288a1b82f0b6da0711cc0563a85e4e4ddcc5`.
AutoDL immutable worktree:`/root/autodl-tmp/worktrees/bace-llm-common-ff77288a`.
Four real task specs:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-ready-ff77288a-20260905`.
BRICS472 vocabulary/386×8 attempts adopted through exact manifest chains.
No new vocabulary or proposal pool was generated.

L1=7B off-the-shelf; L2=existing7B PPO LoRA (300updates, no project SFT);
L3=2B off-the-shelf. L2 uses the original adapter but requires matched native
prompt regeneration: the old main plain-prompt pool is not falsely adopted.
Generation uses one RNG initialization per shard/regime and full four-sequence
parent-call checkpoints. Tiny real Torch/Python/NumPy resume parity passed;
actual 7B GPU load/resume and GPU2B NF4 execution remain unmeasured.

2B isolated CPU load/finite forward/native4-token generation PASS:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/isolated-2b-cpu-25d0c562-20260905/isolated_load_receipt.json`.
Receipt SHA:`dc1b1b93f3304b63aae6eb6c68b6dbf655ad4398f105fc357592c0a0bce442af`.
Actual parameters1889110016, BF16 bytes3778220032, frozen trainable0.
This is readiness, not LLM science PASS.

Native generator and a single frozen-GINE common verifier/selector/test
entrypoint are implemented and tested. There is no GNN-secondary-seed or
main-matrix-count gate for LLM. However the **new temperature contract blocker
keeps all LLM core science, including L0 common evaluation, unstarted**.
GPU1 revocable borrowing is NOT_SUPPORTED: existing reservation and owner have
no <=120-second validated release protocol. Normal GPU use needs a real
unreserved1200-second idle slot and an existing owner-passed locked FD. No new
GPU platform or reservation clearing was implemented. Secondary17/27 remain
plan-only and do not block LLM.

## Published-main partial paper artifacts

AutoDL:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_staging/partial-aids-bace-fd98c5f2-20260905`.
Mac:`/private/tmp/gnn-closeout-paper-partial-20260905`.
AIDS/BACE each contain Figure3,Figure4(PDF/PNG),Table2(CSV/TeX/Markdown).
All8source hashes close; titles/manifests explicitly say PARTIAL. AIDS legal
zero coverage/undefined cost is preserved. Missing cells are PENDING, not0.
No final four-dataset output or changes to the paper user worktree were made.

## Exact immediate status commands

```bash
ssh -o BatchMode=yes autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
ssh -o BatchMode=yes tongji-hpc 'sacct -X -j 2558894,2558895,2558896,2558897,2558898,2558899,2558901 --format=JobID,State,ExitCode,Elapsed -P'
cat /Volumes/DireRaven/counterfactual-hpc-offload/gnn-seed7-closeout-fd98c5f2/8af242e0-f773-4f8a-a23b-b884ef0df65c/control/heartbeat.json
ssh -o BatchMode=yes autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-ready-ff77288a-20260905/llm_readiness.json'
```

Do not rerun the campaign launcher, training, or old evaluation jobs. There is
no authorized temperature-repair command yet; ask for the narrowly scoped
first-fit overlay decision, not a broad experimental redesign.
