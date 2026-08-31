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
official VRRW loop; it is not a mock producer.  The train-side 20k producer now
uses the dataset-specific exact external transition store in
`src/baselines/tastemolnet_gcf_transition_store.py`, the lossless native
candidate archive in `src/baselines/tastemolnet_gcf_candidate_store.py`, and
the fresh/resume entry point `scripts/run_tastemolnet_gcf_full.py`.  This is
still not a T12 paper-result release: generation verification prints only
`[TASTE_T12_GCF_GENERATION_PASS]`.  The paper marker `[TASTE_GCF_PASS]`
remains closed until calibration-only freeze, held-out test, standardized
exports and their separate verifier are implemented and pass.

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

The official transition mapping is now replaced only at its persistence
boundary by a T12-specific append-only journal.  Already-computed binary
coverage is row-major bit-packed; graph hashes, action order, importance dtype
and values are retained, and one expanded official tuple is held in a
deterministic LRU.  Checkpoints contain only the authenticated journal prefix,
active insertion order and LRU keys.  Independent reopen scans and rehashes
the complete prefix and reconstructs the exact numeric tuple without model
calls, action enumeration or RNG.  A raw dictionary, non-binary coverage,
more than two expanded rows, or any `sample_size`/`candidate_capacity` change
remains a hard failure.  The external store is capped at 128 GiB; the
persistent launcher requires 220 GiB free for all T12 generation artifacts.

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

## Direct 10k -> 20k production and generation verification

Use one fresh UUID/root and one generation token across two distinct science
processes. `T12_EXACT_REPLAY_GATE` must name the existing real-A800 gate-v2
JSON, not only its marker file. The other paths are the adopted T7/T3
authorities already used by that canary.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=1
export T12_ATTEMPT_ID=$(python -c 'import uuid; print(uuid.uuid4())')
export T12_GENERATION_TOKEN=$(python -c 'import secrets; print(secrets.token_hex(32))')
export T12_OUTPUT_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/gcfexplainer/t12-production/attempt-$T12_ATTEMPT_ID
export T12_GPU_UUID=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')

python scripts/run_tastemolnet_gcf_full.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode fresh --output-root "$T12_OUTPUT_ROOT" --attempt-id "$T12_ATTEMPT_ID" --generation-token "$T12_GENERATION_TOKEN" --gpu-uuid "$T12_GPU_UUID" --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" --t3-root "$TASTE_T3_ROOT" --official-root "$TASTE_OFFICIAL_GCF_ROOT" --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY" --exact-replay-gate "$T12_EXACT_REPLAY_GATE"

python scripts/run_tastemolnet_gcf_full.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode resume --checkpoint-manifest "$T12_OUTPUT_ROOT/checkpoints/checkpoint-00010000.manifest.json" --output-root "$T12_OUTPUT_ROOT" --attempt-id "$T12_ATTEMPT_ID" --generation-token "$T12_GENERATION_TOKEN" --gpu-uuid "$T12_GPU_UUID" --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" --t3-root "$TASTE_T3_ROOT" --official-root "$TASTE_OFFICIAL_GCF_ROOT" --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY" --exact-replay-gate "$T12_EXACT_REPLAY_GATE"

python scripts/verify_tastemolnet_gcf_full_generation.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --production-root "$T12_OUTPUT_ROOT" --output-root "$T12_OUTPUT_ROOT/generation_verification"
```

The verifier independently rescans both transition/history prefixes, reopens
both checkpoints, proves that the 20k trace and journals retain the committed
10k prefix, binds both official native archives to their checkpoint states,
and reopens the terminal candidate archive. Its PASS is generation-only and
records `paper_cell_pass=false`.

### Persistent GPU1 handover after T11

For the currently frozen T11 manager, launch the narrow sidecar once from the
deployed immutable checkout. It polls `/proc/605212/stat` for exact start ticks
`763435090`, never signals T11, waits for GPU1 to have no compute process, runs
fresh 10k, resume 20k in a new process, then the independent verifier.
Heartbeats and logs survive the launching shell.

```bash
export T12_REPO_ROOT=/absolute/deployed/integration-checkout
export T12_WAIT_PID=605212
export T12_WAIT_PID_START_TICKS=763435090
export T12_GPU_INDEX=1
export T12_MIN_FREE_GB=220
export TASTE_OFFICIAL_GCF_ROOT=$T12_REPO_ROOT/baselines/gcfexplainer_official
# Also export the adopted NeuroSED root, T3 root, threshold JSON and gate JSON.
bash scripts/autodl/launch_tastemolnet_t12_after_t11_v1.sh
```

The launcher prints the controller PID/root and status command. A concurrent
second launch is rejected by a dedicated T12 lock; no general controller
platform is introduced.

## Remaining paper-cell work

The following remains before publishing the T12 method cell:

1. retain and rehash the already-passed real A800 gate-v2 receipt at launch;
2. run the production resource preflight against the real 3,778-parent route
   and publish the measured checkpoint/RSS receipt below the hard caps;
3. calibration-only ordering with the externally frozen shared WNode
   threshold contract;
4. post-freeze held-out test evaluation, standardized Figure 3/Figure 4/Table
   2 exports, and a separate replaying terminal verifier.

The NeuroSED distance threshold and shared WNode threshold contract remain
required external pins.  No value is inferred from a test fixture or generic
default, and test data must not be opened before calibration freeze.
