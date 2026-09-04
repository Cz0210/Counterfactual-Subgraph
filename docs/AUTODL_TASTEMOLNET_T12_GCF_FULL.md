# AutoDL TasteMolNet T12 GCFExplainer Full Route

## Independent GPU3 release relay

`scripts/autodl/launch_tastemolnet_t12_release_relay_v1.sh` supersedes the old
process-coupled queue.  The successor has exactly four release dependencies:
the calibrated T3 terminal, managed T7 smoke PASS, managed NeuroSED PASS, and
the typed managed release root.  It validates all four before writing
`[TASTE_T12_DEPENDENCY_DECOUPLED]`; it does not inspect or wait for another
Taste full process.

The relay is pinned to physical GPU3.  After the card is naturally idle it
creates a fresh UUID and output root, writes its PID/heartbeat/launch receipt,
and starts the unchanged 10k-to-20k production route.  Launching real science
writes `[TASTE_T12_GCF_FULL_LAUNCHED]`; generation is not successful until the
existing independent verifier publishes its generation PASS.

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
the fresh/resume entry point `scripts/run_tastemolnet_gcf_full.py`.
Generation verification still prints only
`[TASTE_T12_GCF_GENERATION_PASS]`.  The dataset-specific paper continuation is
implemented in `src/baselines/tastemolnet_gcf_full_postprocess.py`: it reopens
that PASS, freezes a calibration-only WNode order, loads held-out test only
after the freeze, exports Figure 3/Figure 4/Table 2, and requires a distinct
terminal verifier invocation before `[TASTE_GCF_PASS]` can exist.

## Authorized identity change

The vendored official implementation uses
`hash(graph_embedding.tobytes())`.  That integer changes between isolated
Python processes and cannot identify a restored registry.  For T12 only, the
project boundary replaces that registry key with
`canonical_parent_free_gine_and_neurosed_graph_sha256_v2`:

- the adapter first decodes the full molecule with the retained atom/bond
  sidecars, then hashes its canonical chemistry identity and the exact
  normalized node/edge tensors sent to frozen GINE;
- the same key also binds the canonical parent-free one-hot/untyped-edge graph
  used as the generated NeuroSED query, including canonical SMILES and exact
  node/undirected-edge counts;
- source-parent metadata, raw embedding bytes, and Python's built-in hash are
  excluded;
- a queued embedding digest still proves that each official hash request
  consumes the graph from the corresponding scorer call.

The stronger model-input identity supersedes the earlier attributed-native
identity after a real production attempt proved that identical raw tensors
can decode differently from different retained parent sidecars.  Any replay
gate or checkpoint carrying the earlier identity contract is therefore not a
release authority for this implementation; run a fresh bounded three-process
gate before a fresh production root.

Official edit actions, teleportation, frequency reinforcement, GINE
probabilities, candidate predicate (`pred != Sweet`), score (`1-p(Sweet)`),
and NeuroSED coverage are unchanged.  When the same exact model input is
rescored with CUDA low-bit drift, T12 applies the already-reviewed Taste
canonical-row envelope (`rtol=1e-5`, `atol=1e-7`) and returns the first row to
the official walk.  Any model-input, discrete prediction, validity, candidate,
coverage, or out-of-envelope numeric change is a failure with the exact
mismatching fields named in the terminal exception.

### Canonical NeuroSED query ordering

The official walk may emit two raw `x`/`edge_index` tensors for the same
canonical attributed graph with different node or directed-edge ordering.
Those raw byte hashes are encoding evidence, not distinct molecular queries.
T12 records the first raw SHA-256 as the representative.  Canary and other
non-production bridges retain every ordered unique raw SHA-256 variant in the
bridge checkpoint and report.  The production bridge instead keeps only the
representative in each live record, counts later variant observations, and
authenticates every observed raw SHA-256 in compact-history v2.  This keeps the
live record constant-sized even if one canonical graph is observed through an
unbounded number of permutations.  A batch containing multiple encodings of
the same exact canonical identity and collision payload performs one NeuroSED
evaluation and reuses its binary coverage row.

Before NeuroSED, T12 reparses the already-bound canonical attributed-graph
SMILES, reconstructs the frozen one-hot features, and sorts both directions of
every edge.  The resulting deterministic tensor bytes are separately hashed.
This makes an evicted production identity reconstruct the same mathematical
query after restart.  Each compact observation also retains its raw query SHA,
so the first observation is recovered exactly by the rebuilt disk index while
the checkpoint hash chain binds every later encoding observation for offline
audit.  Production checkpoint/report metadata explicitly marks the live
variant list as representative-only and the journal as the complete variant
evidence; canary metadata marks the in-record list as complete.  The existing
historical coverage SHA/count comparison still fails closed if a recomputed
coverage row changes.  A different canonical graph, collision
payload, feature vocabulary, target cohort, or threshold cannot reuse the
row.  Hash/collision inconsistency is rejected before any cached coverage is
returned.  This changes only tensor ordering at the permutation-invariant
NeuroSED boundary; generated-to-original direction and threshold math remain
unchanged.

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
historical observations use a fixed 304-byte append-only hash chain and a
rebuildable, non-authoritative disk index.  For the 3,778-parent production
cohort the checked bridge proof is:

- at most 200,000,001 scored observations and a 60,800,008,520-byte journal
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
but unequal raw archive SHA-256 values; gate v3 records this as
`NON_SEMANTIC_SERIALIZATION_REPRESENTATION_ONLY`.  A canonical difference is
still a hard scientific replay failure.  Gate v3 also records the exact graph
identity and NeuroSED permutation contracts; production rejects a stale v2
receipt even when its earlier exact-replay fields are otherwise PASS.

No real A800 timing is fabricated here.  For scheduling only, the conservative
pre-measurement envelope is 6--12 GiB peak VRAM and 20--70 minutes total for
all three phases.  Replace that envelope with the first real receipt/log
measurement; the 16/8-step walk is bounded, but model/source loading dominates
and cluster load can vary.

## Direct 10k -> 20k production and generation verification

Use one fresh UUID/root and one generation token across two distinct science
processes. `T12_EXACT_REPLAY_GATE` must name a fresh real-A800 gate-v3
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

Production releases each complete transient neighbour batch at the official
restart/move boundary, after all hashes are consumed and the selected graph is
committed.  Only rows admitted to the official live `graph_map` remain in the
full bridge cache; every observation remains authenticated by the compact
history.  This is a cache-lifetime correction only: it adds no RNG draw or
model call and changes no graph, transition, candidate, score, coverage or
ordering.  A live graph whose full row was previously evicted must be present
in compact history at checkpoint closure, otherwise the run fails closed.

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

## Durable generation-to-paper continuation

The paper follower may be started while the generation sidecar is still
waiting for T11.  It reads the exact generated root from that controller's
`launch.env`, rejects controller exit/PID reuse before generation PASS, and
then runs the resumable calibration/test stage followed by a distinct
terminal verifier process on GPU1.

```bash
export T12_REPO_ROOT=/absolute/deployed/integration-checkout
export T12_GENERATION_CONTROLLER_ROOT=/absolute/controller-for-running-generation
export T12_PAPER_CONTROLLER_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control/t12-paper-$(date -u +%Y%m%dT%H%M%SZ)
export T12_GPU_INDEX=1
export T12_TRAIN_CSV=/absolute/prepared/tastemolnet_train.csv
export T12_CALIBRATION_CSV=/absolute/prepared/tastemolnet_calibration.csv
export T12_TEST_CSV=/absolute/prepared/tastemolnet_test.csv
export T12_GNN_CHECKPOINT=/absolute/t3/artifacts/checkpoint
export T12_MOLCLR_ROOT=/absolute/pinned/molclr/source
export T12_MOLCLR_CHECKPOINT=/absolute/pinned/molclr/checkpoint.pth
export T12_WNODE_THRESHOLD_CONTRACT=/absolute/shared-thresholds/tastemolnet.json
bash scripts/autodl/launch_tastemolnet_t12_paper_after_generation_v1.sh
```

After exact terminal PASS, the matrix-consumable locator is written atomically
at `$T12_PAPER_CONTROLLER_ROOT/cell_root_locator.json` with schema
`fast16_matrix_cell_root_locator_v1`.  The terminal root itself contains exact
bytes `[TASTE_GCF_PASS]\n`; the distinct verifier root contains the same marker
and `tastemolnet_t12_terminal_verification_v1` evidence.

For a manual already-generated root, run
`scripts/run_tastemolnet_gcf_full_postprocess.py` with the same authority paths,
`--generation-root`, `--generation-verification-root`, `--output-root`, WNode
cache paths and `--device cuda:0`; add `--resume` when `checkpoint.json` exists.
Then invoke `scripts/verify_tastemolnet_gcf_full.py` with the same authority,
paper output, and one fresh `--verification-root`.  The verifier is the only
process authorized to write the paper PASS.

The NeuroSED distance threshold and shared WNode threshold contract remain
required external pins.  No value is inferred from a test fixture or generic
default, and test data is not opened before the fsynced calibration freeze.

## Accelerated fork from the sealed step-250 reference

`build_t12_accelerated_from250_v1.py` binds the current immutable reference
task spec, checkpoint manifest/payload, generation receipt, compact-history
prefix, and first-seen embedding prefix into one fresh accelerated task spec.
The current sealed evidence is deliberately pinned by SHA-256; a replacement
reference cannot silently enter this route.  The builder also emits a
non-executable downstream descriptor.  It is explicitly
`BLOCKED_PENDING_PRODUCTION_IDENTITY_REFRAME`; it is not a sealed full,
postprocess, or publisher task spec.

The sealed task spec binds an immutable Mut GPU0 release-receipt path.  The
science owner itself requires the physical receipt to contain `status=PASS`,
`gpu_index=0`, and `gpu_released=true` before taking the lease or creating the
science root; this gate cannot be bypassed by invoking the Python entrypoint
without its convenience launcher.  It never signals or restarts the GPU3
reference.  The accelerated owner copies the exact committed binary prefixes
to a fresh root, changes only their absolute storage-root fields, retains the
reference checkpoint identity, and records GPU0 as a separate transport
identity.  Its disposable SQLite history index is placed under the explicitly
bound local-scratch root; all append-only journals remain authoritative in the
fresh output root.

After both arms have checkpoints 500 and 510, the owner's `parity` action can
write `endpoint_250_500_510_comparison.json`.  It compares complete endpoint
checkpoint state, but the current checkpoint schema does not retain every
per-step selected action, pre-softmax logit, and NeuroSED distance.  Therefore
the receipt is `ENDPOINT_ONLY_PASS_PROMOTION_BLOCKED`, never a required
251--500 per-step parity PASS.  Promotion from diagnostic bounds to the
10k/20k resource contract requires both a separately implemented per-step
proof and a reviewed prefix-identity/bounds reframing step.  Until then, no
full/postprocess/publisher task is dispatchable.

The builder writes `promotion_blocker.json` for this exact limitation.  The
existing compact journal authenticates graph identity, probabilities,
prediction, coverage digests and endpoint frequency/lineage state, but not an
ordered per-step selected-action ledger, pre-softmax logits, or normalized
NeuroSED distances.  Because the live reference did not record those fields,
a correct comparison needs a shadow reference replay from the same sealed
step-250 checkpoint, endpoint-bound back to the live reference at 500/510,
plus an independently recorded accelerated ledger.

The diagnostic history, first-seen and transition segment headers also bind
the 510-step contract and bounds.  They cannot be relabeled as a 20k
checkpoint.  A future promotion must re-emit the authenticated prefix into a
fresh root under the 20k contract, prove the scientific projection and RNG
unchanged, and explicitly support cursor 500 as a promotion seed followed by
the 2500-through-20000 checkpoint schedule.  Until all of those code points
exist and pass, the blocker remains fail closed and no GPU full owner exists.
