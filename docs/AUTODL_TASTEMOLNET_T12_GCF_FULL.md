# AutoDL TasteMolNet T12 GCFExplainer Full Route

## Scope and current release state

T12 is the TasteMolNet `GCFExplainer` main-table cell.  Its scientific route
remains the pinned official full-graph VRRW edit walk, the frozen calibrated
three-class GINE, and generated-query-to-original-target NeuroSED coverage.
It is not a deletion-fragment generator and does not use an RF oracle.

The deterministic restart substrate is implemented in
`src/baselines/tastemolnet_gcf_full_resume.py`.  This first release is not a
T12 paper-result release and exposes no T12 PASS marker.  It deliberately
stops before production until a real A800 canary has proved exact
uninterrupted-versus-new-process replay and the external NeuroSED and shared
WNode threshold authorities are pinned.

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

## Mandatory real-GPU replay gate

Before a production worker can consume the checkpoint, run the same bounded
train-only cohort twice on the same physical A800 and same immutable inputs:

1. uninterrupted from step 1 through the bounded terminal step;
2. a first process through the intermediate cursor, durable checkpoint and
   clean exit, followed by a distinct process that reopens the checkpoint and
   reaches the same terminal step.

Each producer must build its terminal scientific snapshot with
`build_replay_scientific_state`, capture `/proc` evidence with
`capture_linux_process_identity`, construct an observation with
`build_canary_observation`, and publish it with
`write_canary_observation`.  The two processes must bind the same canary
identity and GPU UUID.  The gate compares exact values—not `allclose`—for the
full traversed stable-ID trace, ordered candidate frequencies and importance,
graph/index maps, transitions, current cursor, bridge/adapter/action/RNG
states, generated-to-original coverage, and the official native result.

Close the independent gate with:

```bash
python scripts/run_tastemolnet_gcf_replay_canary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --uninterrupted /absolute/canary/uninterrupted.json \
  --cross-process-resumed /absolute/canary/resumed.json \
  --output /absolute/canary/replay_gate.json
```

or set `T12_UNINTERRUPTED_OBSERVATION`, `T12_RESUMED_OBSERVATION`, and
`T12_CANARY_GATE`, then submit:

```bash
sbatch scripts/slurm/run_tastemolnet_gcf_replay_canary.sh
```

Only an exact gate prints
`[TASTE_T12_GPU_CROSS_PROCESS_REPLAY_CANARY_PASS]`.  This is a replay marker,
not a method-cell marker, and its receipt explicitly records
`production_released=false`.

## Remaining production work

The following must still be implemented and reviewed before launching T12:

1. a dataset-specific worker that reconstructs the held T12 input authority,
   invokes the stable bridge around the official walk, and writes both real
   canary observations;
2. production 10k/20k segment orchestration using the checkpoint API;
3. lossless native candidate graph persistence and train-only candidate
   materialization;
4. calibration-only ordering with the externally frozen shared WNode
   threshold contract;
5. post-freeze held-out test evaluation, standardized Figure 3/Figure 4/Table
   2 exports, and a separate replaying terminal verifier.

The NeuroSED distance threshold and shared WNode threshold contract remain
required external pins.  No value is inferred from a test fixture or generic
default, and test data must not be opened before calibration freeze.
