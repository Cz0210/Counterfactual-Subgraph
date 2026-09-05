# BACE GNN temperature correction and LLM successor — 2026-09-06

## Outcome at handoff

Main matrix: **12/16**, read at 2026-09-05 23:54:59 CST. The matrix SHA remains
`fa6c85263b28e25bddf2e89e6d71f91a9fbb46667e81edd5c35af98ddce25cf8`.
Missing: Mutagenicity/ComRecGC and TasteMolNet/GCFExplainer, GlobalGCE, ComRecGC.
No main process was signalled or restarted and no matrix was written this turn.

Four first validation fits, probability reconciliation for all 902 parent-model
units, ten global calibration selector freezes, and corrected test reports have
completed. The independent scientific replay passed. The final **portable**
acceptance job is still RUNNING; do not yet publish `GNN_CORE_SEED7_CORRECTED_PASS`.

Current chain: completed 2560774 fit → 2560775 calibration → 2560776 freeze →
2560777 test → 2560778 finish. Original final verification 2560779 FAILED because
it counted ten `parents/progress.json` files as parent checkpoints. All original
902 scientific checkpoints were present. Narrow verifier-only repair **2560832**
reopens the existing corrective archive without repeating fitting, inference,
OT or packaging. Its dependent L0 CPU job is **2560839**. Neither submission nor
an intermediate audit constitutes final core acceptance.

The obsolete, never-started L0 job 2560792 was precisely verified PENDING with
`DependencyNeverSatisfied` and cancelled. Its receipt remains preserved.
No completed science was cancelled or deleted.

## Provenance and real fit results

- Original science: `532e83733971701b0709086469d2ed8955a96e25`.
- Original publication: `31391b261750fd901d953d46f7769a597ad3d7e9`.
- Original exact evaluation: `fd98c5f23bf835f2b68799d03b7a2fd8b8b713f7`.
- First temperature correction: `65d767c63dd795370f5331ba72a5f14fa419a4e1`.
- Verifier-only correction: `896be79dc595b4c6503299613c791ef8d5aad8d9`.
- Private feature: `feat/early-gnn-first-ablation-20260905`; pushed fast-forward.
- All work used new isolated worktrees. Main execution and sealed roots unchanged.

| Backbone | Fitted T | Validation examples | Validation NLL before | After |
|---|---:|---:|---:|---:|
| GIN | 1.141163362671653 | 187 | 0.4434133306314786 | 0.44149238982035993 |
| GCN | 1.3824019233729015 | 187 | 0.436542537913518 | 0.420315335478469 |
| GATv2 | 1.1259209687482514 | 187 | 0.4636377689943545 | 0.46164323464901585 |
| GatedGCN+ | 1.5573810153872865 | 187 | 0.43663582751988755 | 0.4068922606203846 |

GINE remains **1.5447202081060156**, with its original receipt SHA
`a3e48f0c31014c714a10292af804e28919867d3c50c51e83e4c2ed81c801dbef`.
The four old `not_fit` files and all five weights are unchanged. All four raw
validation-logit files were available; no validation inference was necessary.
Inputs bind ordered IDs, labels, SMILES, mapping and actual selected best_state.
Only scalar log-temperature was optimized using the historical float64 LBFGS
contract; no model training occurred.

The original 27,013,606-byte package is preserved with SHA
`e40c9ee7a3e53f0db9635040b7fb7f09cf3fac22174444a16f743a7696e8cf63`.
The distinct corrective archive already exists with SHA
`10f7a32e9bcb95c529d52cf8b3c442085dd16fd98d1355178866186a3d28e579`,
but remains unaccepted until the portable verifier completes. Do not rename the
old package or relabel its historical blocked audit.

Original test results were previously evaluated. The repair contract explicitly
records that fact and `repair_selected_using_test=false`; this is not a claim
of first-ever unseen test evaluation. Fitting used validation only, new selectors
were frozen before corrected test probabilities were read.

Calibration: 288 units, 1,144 residual inference-only records, 140 raw OT reuse
records. Test: 614 units, 2,580 residual inference-only records, 203 raw OT reuse
records. Total: **902 units, 3,724 residual inference-only records, 343 raw OT reuse
records, zero OT recomputation, zero cache provenance gaps**. Positive-temperature
argmax, source cohort and strict flip checks all reported zero changes. All ten
ordered-rule hashes remain the same after deterministic replay. Probability
metrics changed; improvement was not an acceptance condition (GIN Brier slightly
worsened and is retained). Missing historical training wall-time/RSS is **N/A**;
empty fields in the original machine CSV schema must not be plotted as zero or
used to claim an efficiency advantage. Current frozen trainable count=0 describes
evaluation, not training-time parameter count.

## Persistent continuation

Correction root:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z/temperature-repair-20260905T144600Z`

Within it, `submission.json` preserves the six original jobs;
`verification_retry_submission.json` binds 2560832 and its diagnosed failure;
`verification-attempt-2/verification.json` and `verified/result_package.json` will
only be created after portable acceptance. Do not resubmit the original chain.

Narrow authorization:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/authorizations/first-fit-four-20260905T151313Z.json`

Authorization file SHA: `a299709e6da8d9edc6e3df2e92837e2314e2b9720955d2678b5a05a411b66c85`.
Repair contract file SHA: `f6cc398f6b1e3362315547bcfd78ef1b30fd65730730018f76c1d6dddece21e0`.

HPC immutable verifier worktree:
`/share/home/u20526/czx/worktrees/bace-gnn-verify-896be79d`.
AutoDL immutable import/LLM worktree:
`/root/autodl-tmp/worktrees/bace-gnn-verify-896be79d`.

Scoped Mac relay PID **82454**, caffeinate PID **82455**. It waits only for 2560832,
then transfers this corrective package HPC → external disk → fresh AutoDL import,
checks bytes/SHA and safely extracts. It exits on completion or a real failure;
no permanent daemon, source deletion, main matrix write or T8 retransmission.

Relay root:
`/Volumes/DireRaven/counterfactual-hpc-offload/gnn-seed7-corrected-20260905T144600Z/90844a40-f255-47cd-97b9-361478b75bac`

Reserved fresh import destination (not yet completed):
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/seed7-corrected-20260905T144600Z/import-90844a40-f255-47cd-97b9-361478b75bac`

First relay attempt `f666e309-8637-47c4-a165-a2f021fec8e3` exited on the failed
predecessor before transferring anything. Its log/terminal is retained.

## LLM: real CPU dependency, GPU interface still incomplete

L0 **2560839**, `afterok:2560832`, intel CPU-only, 8 CPU/32GiB/12h. The common
evaluator itself reopens the corrected package before science. It uses the
unchanged main GINE, 472 existing BRICS vocabulary items and 386×8 attempts.
The portable inputs contain 4,943,849 bytes; inner manifest SHA is
`0660ebd8b4a6ea64a270598e53f65dbded7cae5c1bbfc15c5da676d675316f75`.
No vocabulary regeneration, LLM weights transfer or old T8 retransmission.

L0 control root:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/llm/runs/bace-l0-corrected-65d767c6-20260905`

`successor_submission.json` is authoritative; scientific output is `science-verify2`.
Independent registry:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/llm/registry/core-corrected-65d767c6`.

Final fresh GPU specs:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-ready-896be79d-20260905`

L1→L2→L3 order is sealed. L2 reuses the existing 300-update PPO-LoRA weights but
requires matched native-chat generation; its old plain-prompt pool is not adopted.
No SFT checkpoint or training is invented. 2B CPU 4-token evidence is not a GPU
smoke or full generation result. No main-cell-count/secondary-seed gate remains.

**Unfinished GPU engineering:** existing `gpu_lock.py run` does not pass its held
FD, sets device index instead of UUID, and does not maintain the ≤120s live
resource evidence required by the native generator. `run_bace_llm_successor.py`
is a real one-shot adapter, not a running owner. Thus GPU specs are
`PREPARED_NON_DISPATCHABLE / BLOCKED_GPU_OWNER_INTERFACE`, not unattended GPU-ready.
Do not manufacture FD/evidence, clear T13 reservation, borrow, co-locate or start
a new generic GPU platform. The next implementation must be a reviewed narrow
extension to the existing owner/dispatcher, retaining actual lease and main
priority checks. L1/L2/L3 have no science PID this turn.

## Main blockers and remaining work

| Line | Last actual state | Safe next step |
|---|---|---|
| Mut | trace-on245/500, off not started; owner193161/start23395179; science193180/start23395555 | Preserve sequential A/B and existing success/adoption chain |
| T12 | reference250→500; owner162844/start18577652; science173495/start20206493; slow positive I/O | Preserve original PID, wait full parity; diagnostic250 cannot promote |
| T14 | retry2 reference306/500; owner268102/start32824581; science268321/start32838718 | Existing reload/parity→authorized full; no retry3/old monolith |
| T13 | FAILED -9 at14:21UTC after preprocessing3823/3823; owner284476/science284493 gone; no training checkpoint | Diagnose/repair eager augmented-data materialization before any fresh successor |

T13 likely hit a memory peak (`torch.stack`/`repeat_interleave` of augmented graph
tensors), but PID-specific OOM is not proven: kernel log unavailable and cgroup
OOM counters are historical. Preserve the imported 20 patterns, no mining or
6.10GB transfer repeats. Proposed repair: lazy/indexed augmented dataset sharing
parent tensors, exact sample/RNG/order/batch tensor parity, then a scoped fresh
successor under the same target0/2, seed7,100-epoch contract. This repair and launch
have **not** been performed. GPU1 is empty but T13 reservation remains HELD.

Mut Route-B is **not closed**: its trace-off generation payload lacks the selected
action causal history required by the existing exact lineage consumer. Old trace-on
lineage, node-origin metadata and final graphs are not substitutes. The committed
preflight identifies this instead of launching a guaranteed-failing fresh50k.
Complete an observational lineage producer and prove unchanged RNG/science plus
checkpoint parity, then wire fresh universe→pair store→exact DBSCAN→evaluation→
publisher. Do not run it unless actual A/B science fails. No speculative adoption.

Last resources: AutoDL free1,684,373,798,912 bytes;98,214 inodes. Existing100,000
inode gate lacks1,786; known compact additional peak2,068 implies3,854 shortfall
while preserving the guard. Unknown generation/checkpoint/evaluation peaks remain
unmeasured, so this is not full admission. No threshold reduction or cleanup.
Cgroup limit515,396,075,520; usage10,333,831,168; headroom505,062,244,352 bytes.
GPU0 Mut, GPU1 reserved T13/no CUDA, GPU2 T14, GPU3 T12.

No new algorithm/temperature authorization is needed for the submitted correction
and L0. Remaining human direction concerns T13's newly failed route and scope of
its memory-repair successor, not permission to repeat already-approved fits.
The Mut/LLM interface gaps still require engineering; they are not science PASS.

## Paper outputs and tests

Existing AIDS/BACE four-method PARTIAL artifacts remain at:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_staging/partial-aids-bace-fd98c5f2-20260905`.
Mac copy: `/private/tmp/gnn-closeout-paper-partial-20260905`.
Missing cells remain PENDING, never zero. No final four-dataset figures exist yet.
Corrected GNN tables are generated under the correction `verified/` directory,
but remain provisional until portable acceptance/import. LLM tables still await
actual L0/L1/L2/L3 results. Secondary seeds17/27 remain queued and do not block LLM.

Focused tests: prior integrated86 PASS; final verifier-only set30 PASS; CLI help,
compile and shell syntax/diff checks passed. No full7B/20B unit test, model training,
environment upgrade or login-node model inference. Original user checkout untouched.

## Exact status commands (read only; no recovery launch implied)

```bash
/Users/cz0210/miniconda3/envs/smiles_local/bin/python /private/tmp/gnn_temp_main_readonly_audit_20260905.py
ssh -o BatchMode=yes autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json'
ssh -o BatchMode=yes tongji-hpc 'sacct -X -j 2560774,2560775,2560776,2560777,2560778,2560779,2560832,2560839 --format=JobID,State,ExitCode,Elapsed -P'
ssh -o BatchMode=yes tongji-hpc '/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python -I -B /share/home/u20526/czx/worktrees/bace-gnn-verify-896be79d/scripts/hpc/gnn/repair_bace_seed7_temperature_contract.py --config configs/hpc.yaml --action status --output-root /share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z/temperature-repair-20260905T144600Z'
cat /Volumes/DireRaven/counterfactual-hpc-offload/gnn-seed7-corrected-20260905T144600Z/90844a40-f255-47cd-97b9-361478b75bac/control/heartbeat.json
ssh -o BatchMode=yes autodl-a800 'cat /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/core-native-ready-896be79d-20260905/llm_readiness.json'
ssh -o BatchMode=yes tongji-hpc 'sacct -X -j 2560839 --format=JobID,State,ExitCode,Elapsed -P'
```

Do not rerun completed jobs or create duplicate owners. A new error must be diagnosed
and recorded; at most two evidence-based engineering retries per failed stage.
Once only these persisted jobs/relay remain waiting, end interactive polling.
