# AutoDL Root-Cause Acceleration Continuation Handoff

Snapshot basis: 2026-08-24 07:23 UTC. This document is a rolling, truthful
handoff for the active AutoDL continuation. A marker is recorded only after its
own immutable terminal proof exists. Historical failures and frozen adoptions
are never rewritten.

## Scientific boundary

- The paper matrix remains **7/16**: the six user-approved AIDS/Mutagenicity
  frozen-v4 cells plus frozen BACE Ours.
- BACE uses the frozen calibrated GINE checkpoint at
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689`;
  RF is not used.
- TasteMolNet remains `RUN_TASTEMOLNET=0` and
  `BLOCKED_LICENSE_REVIEW`.
- No old output root has been deleted, overwritten, or resumed with new code.
  The old BACE 50k writers remain the scientific fallback.

## Controllers and current ownership

| Route | Controller / worker | Current state | Scientific ownership |
|---|---:|---|---|
| Root-cause read-only health monitor | `root_cause_acceleration_v1_17f8688`, PID `457778` | running, 60 s heartbeat | none; read-only observation only |
| AIDS ComRecGC repair-v4 | controller PID `273356`, science PID `273939` | old brute DBSCAN running but unviable | CPU only; old root protected |
| Mut ComRecGC trace-off parity | controller PID `273766` | waiting for exact AIDS terminal PASS | no GPU |
| BACE GlobalGCE v6 | `bace_globalgce_frozen_gine_v6_9646874`, PID `450901`, science PID `451688` | formal training running | GPU1 exclusive |
| BACE GCF CPU lockstep | `bace-gcf-cpu-lockstep-284cc78d-20260824T060014Z`, worker PID `456488`, current science PID `466792` | Quick-50 PASS; Quick-100 legacy-A running | CPU only |
| BACE ComRecGC deterministic M500 | `bace-comrecgc-deterministic-m500-17f8688-20260824T060331Z`, worker PID `456429`, pair PID `456435`, science PID `456442` | legacy half running | GPU2 exclusive |
| Deferred frozen-GINE benchmark | `bace-gine-deferred-1983ad0d-20260824T070726Z`, PID `466744` | `WAITING_RESOURCE/PAIR_REGISTRY_RUNNING` | none until the complete M500 pair releases GPU2 |
| Original four-method continuation | controller PID `138716`, GCF science PID `139725` | mixed historical state; GCF full is genuinely running | GPU0 exclusive |
| Repair-v1 continuation | controller PID `144695`, ComRecGC science PID `169008` | mixed historical state; ComRecGC full is genuinely running | GPU3 exclusive |

Fresh takeover evidence is under
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/root_cause_acceleration_proc_20260824T050616Z`.
It contains controller, worker, lock, CPU, I/O, progress, ETA, and filesystem
snapshots. The 120-second host sample observed 112 logical CPUs, 4.397% mean
aggregate utilization, 5.943% p95 utilization, load-one 5.899 mean, and
negligible iowait. The slow scientific routes are single-process CPU/RDKit
bottlenecks, not whole-host CPU saturation.

## Four GPU slots

| GPU | UUID suffix | Current task | Live CUDA PID | Observed memory/utilization |
|---:|---|---|---:|---|
| 0 | `...0c0732` | old BACE GCFExplainer 50k | `139725` | 763 MiB / instantaneous 0% |
| 1 | `...7a690` | BACE GlobalGCE v6 formal train | `451688` | 46,513 MiB / instantaneous 15% |
| 2 | `...e1484` | deterministic ComRecGC M500 legacy | `456442` | 887 MiB / instantaneous 0% |
| 3 | `...d3693b` | old BACE ComRecGC 50k | `169008` | 899 MiB / instantaneous 0% |

The instantaneous zeroes on GPUs 0/2/3 do not imply dead work: their science
processes are in `R`, consume about one CPU core, retain the expected CUDA
context, and advance atomic progress. All four cards are currently occupied,
so no co-location benchmark is running and no extra GPU task may be inserted.

## AIDS and Mutagenicity ComRecGC

The immutable old AIDS pair store is terminally promoted with 560 chunks and
91,916,686 rows. The old exact sklearn-brute route has
`eps=0.02`, `min_samples=3`, Euclidean distance, self-neighbour semantics, and
deterministic minimum-core-component border assignment. At this snapshot its
neighbour-count checkpoint is `14960/91,916,686`; the rolling health monitor
classifies the ETA as unviable.

The replacement is a fresh, exact, promotion-only external route. Its release
candidate includes authenticated crash recovery, source SHA/stat closure,
adaptive all-core/one-component proof, deterministic downstream streaming,
and strict RSS/headroom gates. It is **not launched yet**: final production
builder/supervisor review must additionally bind the exact old read-only PID
identity, reject every other high-memory ComRecGC process, and prove the live
128 GiB headroom / 96 GiB route cap. The old PID is not signalled until the
fresh route passes source, equivalence, checkpoint, and throughput gates.

Mut remains correctly blocked on the exact AIDS replacement PASS. It has not
allocated a GPU and will not run concurrently with the high-memory AIDS route.

## BACE GCFExplainer

The earlier patched Quick-50 preserved every selected transition, RNG state,
canonical graph sequence, frequency, and coverage but differed in a few raw
frozen-GINE embedding bits. A fresh repeated-cold benchmark isolated the cause:
five identical CUDA evaluations of the same full ordered 64-row batch produced
five different hidden/logit byte digests, while CPU was byte-identical 5/5.
Labels remained equal and the maximum probability/logit-level numerical drift
was `3.576e-7`. The formal audit is
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/bace_gcf_cuda_raw_byte_nondeterminism_20260824T054301Z/audit.json`.

A CPU frozen-GINE lockstep route ran legacy-A, legacy-B, and patched Quick-50
from fresh roots. Legacy-A versus legacy-B is `LOCKSTEP_EXACT`; legacy-A versus
ordered-v2 is `CANONICAL_EXACT` with nested `LOCKSTEP_EXACT`, identical trace
SHA256 `3c7bdc9272e3530ddf08ddef32f1f0fd0caa7e12f061d1a93004bbaa0c3b1198`,
`first_divergence=null`, and no failures. Quick-50 is therefore a real
diagnostic equivalence PASS and released CPU Quick-100; Quick-100 legacy-A was
at 30/100 at this snapshot. M500 remains dependency-blocked. The old full
continues and is not eligible for replacement on Quick-50 evidence alone.

## BACE ComRecGC

The first M500 divergence has been identified. The prior legacy/optimized pair
had 2,420 versus 2,422 candidates; the first stable transition divergence was
row 1, while the first raw embedding hash drift appeared at row 0. This is the
same CUDA raw-byte frozen-GINE identity problem, not a buffered-writer tail.

The fresh pair keeps NeuroSED on CUDA but computes the official GINE graph key
on CPU. Both roles use the same frozen device contract. The pair is currently
in its legacy half at 100/500, about 90.2 steps/hour; optimized and the final
audit are automatically sequenced after it. The formal equivalence audit also
compares the complete device contract, trace, payload, lineage, checkpoint,
and serialization identities. No M500 PASS has been claimed.

The old 50k fallback is live at 4,400/50,000, about 108.6 steps/hour. Sharding
is not authorized before M500 PASS; the current algorithm remains stateful and
must use ordered planning/collection unless a later proof establishes index
independence.

## Frozen-GINE scoring benchmark

The diagnostic scorer preserves full duplicate-containing ordered batches,
loads the frozen checkpoint once, uses stable sequence IDs, and keeps any cache
key bound to graph/checkpoint/temperature/feature/device identity. CPU is the
only observed byte-stable device for the official raw embedding key. The full
CPU/GPU matrix for batch sizes 1, 8, 32, 128, and 512 is queued behind the
complete ComRecGC M500 pair. Persistent controller
`bace-gine-deferred-1983ad0d-20260824T070726Z` binds the pair run ID, launch-spec
SHA, worker start-ticks, UUID lock, thread environment, input semantic hash,
and a fresh benchmark root. It remains `WAITING_RESOURCE`, rather than treating
the transition from legacy to optimized as a free GPU. The benchmark reports
pure-model, batching, end-to-end, argmax, calibrated-probability differences,
normalization, and the best device/batch summary. A faster allclose CUDA result
is not an exact-identity waiver.

## BACE GlobalGCE v6

The bridge now treats official affine edge-class outputs as logits and applies
class-axis softmax before the hard argmax codec. Negative finite logits are
valid; there is no clamp, RF, or surrogate. The production-shaped bridge smoke
passed with negative scores, non-zero transformation gradients, zero GINE
parameter gradients, hard-oracle parity, and unchanged checkpoint hash.

The v5 mining adoption was correctly rejected because its old manifest did not
prove the exact-v2 flag. V6 performed a fresh exact stable-top-k mining pass:
19/19 roots, 1,601 reported patterns, 20 retained, 1,371 pruned branches, and a
hash-bound selected identity. GPU1 now runs official rule/candidate training.
Source expansion 353/353 is complete. Epoch 0 published the first atomic
checkpoint (`next_epoch=1`, SHA256
`8ac602e3d3ae00e02da2315142e71c3f88bc7218a6ba8c62b3e743306aad4139`)
after about 100.5 minutes, including the first full validation. Epochs 1--4
and the next validation at epoch 5 are still needed for an honest cyclic ETA.
The persistent 16-task manifest already includes calibration, frozen
selection, held-out test shards, merges, standardization, and final freeze.
No BACE GlobalGCE cell PASS has been claimed.

## Replacement and co-location decisions

- Old BACE GCF/ComRecGC 50k: continue. Neither equivalence nor a 30% earlier
  projected optimized finish gate has passed.
- AIDS old brute: continue read-only until the fresh exact route is independently
  released and demonstrates a safe checkpoint plus viable throughput; then use
  project stop/SIGTERM only, never SIGKILL.
- GPU co-location: not run. All four GPUs have exclusive work, and no semantic
  equivalence gate authorizes shared-lowmem scheduling yet.
- Paper staging: frozen partial. No 12-cell figure/table is generated at 7/16.

## Matrix and license

`matrix_complete_cells=7`, `matrix_total_cells=16`. The authoritative registry
must not be advanced until a real standardized cell publishes its terminal
closure. AIDS/Mutagenicity/BACE completion will trigger the existing strict
three-dataset supervisor; it cannot create figures or tables while any of its
12 required owner-bound roots is absent.

TasteMolNet remains the only user-action gate: an exact-data license or explicit
research-reuse approval file is required. No preparation smoke is counted as a
paper cell.

## Status and restart commands

Read-only root-cause health status:

```bash
cat /autodl-fs/data/counterfactual-subgraph-runtime/control/root_cause_acceleration/root_cause_acceleration_v1_17f8688/heartbeat.json
cat /autodl-fs/data/counterfactual-subgraph-runtime/control/root_cause_acceleration/root_cause_acceleration_v1_17f8688/state.json
```

GlobalGCE v6 status:

```bash
PYTHONPATH=/root/autodl-tmp/worktrees/run-globalgce-v6-96468740 \
  /root/miniconda3/envs/smiles_pip118/bin/python \
  /root/autodl-tmp/worktrees/run-globalgce-v6-96468740/scripts/autodl/status_four_by_four.py \
  --project-root /root/autodl-tmp/worktrees/run-globalgce-v6-96468740 \
  --data-root /autodl-fs/data \
  --control-root /autodl-fs/data/counterfactual-subgraph-runtime/control \
  --controller-id bace_globalgce_frozen_gine_v6_9646874 --format table
```

Do not execute a restart while the matching PID/start-ticks and heartbeat are
fresh. For a proven-dead generic controller, rerun its immutable manifest with
`scripts/autodl/launch_four_by_four.sh`; for an experiment-registry pair, use
the frozen `launch_spec.json` and its registered launcher rather than manually
starting a second writer. Preserve every output/controller root.

## Current release markers

- `TASTE_LICENSE_BLOCKED`
- `ROOT_CAUSE_ACCELERATION_CONTROLLER_RUNNING`
- `GCF_QUICK50_EQUIVALENCE_PASS`

All other requested completion/equivalence markers remain pending.
