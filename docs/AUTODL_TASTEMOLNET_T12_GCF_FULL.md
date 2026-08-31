# AutoDL TasteMolNet T12 GCFExplainer Full Route

## Scope and current release state

T12 is the TasteMolNet `GCFExplainer` main-table cell.  Its scientific route
remains the pinned official full-graph VRRW edit walk, the frozen calibrated
three-class GINE, and generated-query-to-original-target NeuroSED coverage.
It is not a deletion-fragment generator and does not use an RF oracle.

The deterministic restart substrate is implemented in
`src/baselines/tastemolnet_gcf_full_resume.py`.  The real bounded replay
producer is implemented in
`src/baselines/tastemolnet_gcf_replay_canary.py` and exposed by
`scripts/run_tastemolnet_gcf_replay_canary_worker.py`.  It reconstructs the
same descriptor-held T7 GINE/NeuroSED/official-source authority and runs the
official VRRW loop; it is not a mock producer.  This release is still not a
T12 paper-result release and exposes no T12 cell PASS marker.  Production
remains closed until this producer passes on a real A800 and the later 20k
train/calibration/test/export worker is implemented.

## Authorized identity change

The vendored official implementation uses
`hash(graph_embedding.tobytes())`.  That integer changes between isolated
Python processes and cannot identify a restored registry.  For T12 only, the
project boundary replaces that registry key with
`canonical_parent_free_attributed_graph_sha256_v1`:

- node attributes are exact atomic numbers decoded from the frozen one-hot
  vocabulary;
- edges are the exact symmetric, untyped official native edges;
- RDKit supplies only canonical graph labelling;
- source-parent metadata, raw embedding bytes, and Python's built-in hash are
  excluded;
- a queued embedding digest still proves that each official hash request
  consumes the graph from the corresponding scorer call.

Official edit actions, teleportation, frequency reinforcement, GINE
probabilities, candidate predicate (`pred != Sweet`), score (`1-p(Sweet)`),
and NeuroSED coverage are unchanged.  When the same structural graph is
rescored with CUDA low-bit drift, T12 applies the already-reviewed Taste
canonical-row envelope (`rtol=1e-5`, `atol=1e-7`) and reuses the first row;
any discrete prediction, validity, candidate, coverage, or graph change is a
failure.

## Persistent checkpoint contract

`capture_checkpoint_payload`, `write_checkpoint`, `reopen_checkpoint`, and
`restore_checkpoint_payload` persist and revalidate:

- the complete official `graph_map`, `graph_index_map`, ordered candidates,
  coverage state, covering set, transitions, traversed trace and walk cursor;
- stable bridge records and scorer/adapter counters;
- native action counters and the current graph identity;
- Python, NumPy, Torch CPU, and every Torch CUDA RNG state;
- fresh attempt UUID and generation token;
- train cohort/split, frozen GINE model/config, NeuroSED model/threshold,
  official source inventory, execution commit/tree, runtime, GPU UUID and
  fixed walk parameters.

Production has exactly 20,000 steps.  Durable production checkpoints are
accepted only at cursors 10,000 and 20,000.  Checkpoint payloads and manifests
are same-directory fsynced and published with no-replace hard links.  Reopen
rehashes the full payload before deserialization and proves the restored live
state and RNG digests.

Production does not use the canary's 128/512 resource values.  It binds the
pinned official `sample_size=10000` and `candidate_capacity=100000`.  Complete
bridge rows are limited to the live official graph/candidate/current domain;
historical observations use a fixed 272-byte append-only hash chain and a
rebuildable, non-authoritative disk index.  For the 3,778-parent production
cohort the checked bridge proof is:

- at most 200,000,001 scored observations and a 54,400,008,488-byte journal
  prefix under the 64-GiB disk cap;
- at most 20,001 complete live rows plus one 10,000-row transient batch;
- a 15,863,382,016-byte bridge RAM formula under 16 GiB;
- a 5,310,251,008-byte bridge checkpoint formula under 8 GiB; and
- a 32-GiB hard cap on any complete checkpoint before publication.

This is not yet a full-route RAM proof.  The unchanged official transition
dictionary can reference 200,020,000 coverage rows: 3,022,702,240,000 bytes as
dense float32, or 94,459,445,000 bytes even if coverage alone is bit-packed.
The production checkpoint orchestrator therefore rejects a raw dict and
requires a later dataset-specific external compact transition store with a
bounded expanded LRU and exact checkpoint export.  Lowering `sample_size` or
`candidate_capacity` is not an authorized workaround.

## Mandatory real-GPU replay gate

Before a production worker can consume the checkpoint, run the same fixed
eight-parent, 16-step bounded
train-only cohort twice on the same physical A800 and same immutable inputs:

1. uninterrupted from step 1 through the bounded terminal step;
2. a first process through the intermediate cursor, durable checkpoint and
   clean exit, followed by a distinct process that reopens the checkpoint and
   reaches the same terminal step.

The sequence uses three distinct science Python processes: uninterrupted,
checkpoint prefix, and resumed suffix.  The gate requires all three
`(pid,start_ticks)` pairs to differ, rehashes the prefix manifest, and binds
its checkpoint identity/state/RNG digests to the resumed observation.  The
gate compares exact values—not `allclose`—for the full official mutable state
(including covering graphs), traversed stable-ID trace, ordered candidate
frequencies and importance, graph/index maps, transitions, current cursor,
bridge/adapter/action/RNG states, generated-to-original coverage, and the
official native result.

Every process requires these controls before importing Torch:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
```

The worker additionally enables deterministic algorithms in error mode,
disables cuDNN benchmarking and TF32/reduced-precision reductions, sets the
highest float32 matmul precision, and records the complete setting in the
runtime identity.  If an official PyG/CUDA kernel has no deterministic
implementation, the real canary fails with that kernel as the blocker; it is
never downgraded to approximate equality.

The vendored NeuroSED helper retries every `RuntimeError` forever, including
deterministic-kernel errors and persistent batch-size-one OOMs.  T12 patches
only that resource boundary: explicit CUDA OOM may halve the batch down to
one, batch one re-raises, and every other `RuntimeError` re-raises immediately.
The attempted batch schedule is checkpointed and included in exact replay;
distance normalization, direction and threshold comparison remain official.

The threshold argument must point at the real calibration-only selector file
`t7_neurosed_threshold_authority.json`.  The worker also reopens and rehashes
its sibling `input_authority.json`, `selection_receipt.json`, `sha256sums.txt`,
`PASS`, calibration distance inventory, and shared WNode contract.  It checks
the exact held T3, NeuroSED and official-source hashes.  A scalar or hand-made
JSON is rejected.

### Recommended one-allocation command

The sequence script guarantees that all three science processes use the same
allocated physical A800 while retaining real cross-process restart:

```bash
export T12_CANARY_OUTPUT_BASE=/share/home/u20526/czx/counterfactual-subgraph/outputs/tastemolnet/gcfexplainer/t12-replay-$(date -u +%Y%m%dT%H%M%SZ)
export TASTE_MANAGED_NEUROSED_ROOT=/absolute/managed/neurosed/final
export TASTE_T3_ROOT=/absolute/managed/t3/root
export TASTE_OFFICIAL_GCF_ROOT=/share/home/u20526/czx/counterfactual-subgraph/baselines/gcfexplainer_official
export TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY=/absolute/threshold-selector/t7_neurosed_threshold_authority.json
sbatch scripts/slurm/run_tastemolnet_gcf_replay_canary_sequence.sh
```

The job resolves the allocated GPU UUID once, launches three separate worker
processes in order, and runs the independent gate last.  `T12_GPU_UUID` may be
set as an additional external pin; if set, a different allocated UUID fails.
The example uses the repository's mandatory Tongji Slurm checkout.  On AutoDL,
use the direct commands below and a fresh `/autodl-fs/data/...` output base.

### Exact direct process commands

Within one already-allocated A800 shell, use two fresh UUID/token pairs and
one fresh base.  The checkpoint and resume commands must reuse the second
pair exactly:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=7
export CUDA_VISIBLE_DEVICES=1
export T12_GPU_UUID=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
export T12_BASE=/absolute/fresh/t12-replay
export T12_U_ATTEMPT=$(python -c 'import uuid; print(uuid.uuid4())')
export T12_U_TOKEN=$(python -c 'import secrets; print(secrets.token_hex(32))')
export T12_R_ATTEMPT=$(python -c 'import uuid; print(uuid.uuid4())')
export T12_R_TOKEN=$(python -c 'import secrets; print(secrets.token_hex(32))')

python scripts/run_tastemolnet_gcf_replay_canary_worker.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode uninterrupted --output-root "$T12_BASE/uninterrupted" --observation "$T12_BASE/uninterrupted.json" --attempt-id "$T12_U_ATTEMPT" --generation-token "$T12_U_TOKEN" --gpu-uuid "$T12_GPU_UUID" --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" --t3-root "$TASTE_T3_ROOT" --official-root "$TASTE_OFFICIAL_GCF_ROOT" --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY"
python scripts/run_tastemolnet_gcf_replay_canary_worker.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode checkpoint --output-root "$T12_BASE/resumable" --attempt-id "$T12_R_ATTEMPT" --generation-token "$T12_R_TOKEN" --gpu-uuid "$T12_GPU_UUID" --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" --t3-root "$TASTE_T3_ROOT" --official-root "$TASTE_OFFICIAL_GCF_ROOT" --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY"
python scripts/run_tastemolnet_gcf_replay_canary_worker.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode resume --output-root "$T12_BASE/resumable" --observation "$T12_BASE/resumed.json" --checkpoint-manifest "$T12_BASE/resumable/checkpoints/checkpoint-00000008.manifest.json" --attempt-id "$T12_R_ATTEMPT" --generation-token "$T12_R_TOKEN" --gpu-uuid "$T12_GPU_UUID" --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" --t3-root "$TASTE_T3_ROOT" --official-root "$TASTE_OFFICIAL_GCF_ROOT" --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY"
```

Then close the gate:

Close the independent gate with:

```bash
python scripts/run_tastemolnet_gcf_replay_canary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --uninterrupted /absolute/canary/uninterrupted.json \
  --cross-process-resumed /absolute/canary/resumed.json \
  --checkpoint-prefix-receipt /absolute/canary/resumable/prefix_receipt.json \
  --output /absolute/canary/replay_gate.json
```

or set `T12_UNINTERRUPTED_OBSERVATION`, `T12_RESUMED_OBSERVATION`,
`T12_CHECKPOINT_PREFIX_RECEIPT`, and `T12_CANARY_GATE`, then submit:

```bash
sbatch scripts/slurm/run_tastemolnet_gcf_replay_canary.sh
```

Only an exact gate prints
`[TASTE_T12_GPU_CROSS_PROCESS_REPLAY_CANARY_PASS]`.  This is a replay marker,
not a method-cell marker, and its receipt explicitly records
`production_released=false`.

The gate compares the recursively loaded official native result through
`tastemolnet_t12_native_result_recursive_exact_v1`: all mapping content,
sequence/tensor order, tensor dtype/shape and scalar values are exact, with no
`allclose`.  Raw `torch.save` SHA-256 values remain separate serialization
evidence.  Attempt `91b` produced equal canonical native-result SHA-256
(`4c6d4df28e9435905bd22c95bb55abcc9f7e367b762f425024a24d8758da9f10`)
but unequal raw archive SHA-256 values; gate v2 records this as
`NON_SEMANTIC_SERIALIZATION_REPRESENTATION_ONLY`.  A canonical difference is
still a hard scientific replay failure.

No real A800 timing is fabricated here.  For scheduling only, the conservative
pre-measurement envelope is 6--12 GiB peak VRAM and 20--70 minutes total for
all three phases.  Replace that envelope with the first real receipt/log
measurement; the 16/8-step walk is bounded, but model/source loading dominates
and cluster load can vary.

## Remaining production work

The following must still be implemented and reviewed before launching the T12
20k main-table worker:

1. run gate v2 over the completed real A800 `91b` observations and retain its
   exact replay receipt (the three workers need not be repeated);
2. implement the remaining external compact official-transition store and its
   bounded expanded LRU; the bridge history and 10k/20k orchestration are now
   implemented, but the raw transition state is deliberately rejected;
3. run the production resource preflight against the real 3,778-parent route
   and publish the measured checkpoint/RSS receipt below the hard caps;
4. lossless native candidate graph persistence and train-only candidate
   materialization;
5. calibration-only ordering with the externally frozen shared WNode
   threshold contract;
6. post-freeze held-out test evaluation, standardized Figure 3/Figure 4/Table
   2 exports, and a separate replaying terminal verifier.

The NeuroSED distance threshold and shared WNode threshold contract remain
required external pins.  No value is inferred from a test fixture or generic
default, and test data must not be opened before calibration freeze.
