# TasteMolNet T4 multiclass oracle smoke, managed release v3

## Frozen science boundary

T4 consumes the exact published managed T3 root:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed
```

It validates the generic managed gate, nested T3 scientific verification, and
the complete `artifacts/checkpoint` inventory. It opens only authenticated
graph-cache `manifest.json` and `calibration.pt`; train, validation, test, CSV,
and RF-oracle payloads are forbidden.

The managed release-v3 authority binding assigns T4 to physical GPU 1,
`CUDA_VISIBLE_DEVICES=1`, visible `cuda:0`, and the exact `nvidia-smi` GPU UUID.
The smoke deterministically selects calibration molecules with true label 1
and frozen-GINE prediction 1. It expands only when the current round has not
reached the terminal gate:

```text
round 1:  16 source parents, at most  8 connected deletions per parent
round 2:  64 source parents, at most 16 connected deletions per parent
round 3: 128 source parents, at most 32 connected deletions per parent
```

The first round with at least 16 real strict flips from at least 8 distinct
parents passes. Both `1 -> 0` and `1 -> 2` are valid strict flips. Seeing both
records `DESTINATION_DIVERSITY_PASS`; seeing only one records
`DESTINATION_DIVERSITY_SINGLE_CLASS_WARNING` and does not block T4 or downstream
Taste stages. Failure to reach the flip/parent minima after round 3 blocks T4.

The worker and independent verifier check batch/single equivalence, all three
probabilities, invalid/full deletion controls, and one model load per scientific
process. Deterministic unit fixtures independently assert that `1 -> 0` and
`1 -> 2` flip while `1 -> 1` does not. Only `calibration.pt` is loaded: train,
validation, test, CSV source payloads, and RF-oracle use remain forbidden.

## Managed execution

The only production entrypoint is the outer runner:

```text
scripts/autodl/tastemolnet_t4_managed_runner_v2.py
```

It owns one UUID managed attempt and continuously holds the canonical GPU1 UUID
lock across:

```text
worker -> SEALED -> independent verifier -> atomic terminal publish -> release ACK
```

The worker can write only aggregate method documents and candidate evidence,
including the three-row aggregate `destination_distribution.csv`.
The verifier independently repeats the science and is the only caller allowed
to publish `verification.json`, `gate.json`, and `PASS`. Direct worker and
verifier Slurm scripts are static AutoDL-only refusals.

The managed attempt input hashes are exactly:

```text
t3_gate
t3_verification
graph_cache_manifest
calibration_cache
controller_launcher_receipt
controller_receipt
controller_anchor_heartbeat
gpu1_lease
```

The anchor heartbeat is historical immutable input H1. Worker and verifier
holders independently prove H1 belongs to the full sequence-1-to-terminal
chain, while adopting fresh H2+ terminal generations. Candidate evidence
records worker-initial and worker-final terminal heartbeats separately; a
normal heartbeat advance therefore cannot create a false quarantine.

The activation phases are `WORKER_ACTIVE`, `WAITING_VERIFIER`,
`VERIFIER_ACTIVE`, and `RELEASE_REQUESTED`. A science child blocks for at most
45 seconds waiting for its exact phase acknowledgement. The runner requests an
append-only renewal after acquiring GPU1, giving the bounded smoke two lease
windows. Release occurs only after the independent verifier exits successfully
and the final path exists.

## CLI shape

Run only from a reviewed clean immutable AutoDL checkout after the controller
has acknowledged a fresh GPU1 lease:

```bash
python -I -B scripts/autodl/tastemolnet_t4_managed_runner_v2.py \
  --config "$PWD/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  --stage-root /absolute/control/T4_ORACLE_SMOKE \
  --final-path /autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/t4-oracle-smoke-UUID \
  --t3-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed \
  --graph-cache-root /absolute/private/graph-cache \
  --gpu-uuid GPU-REVIEWED-GPU1-UUID \
  --controller-launcher-receipt /absolute/launcher/launcher_receipt.json \
  --controller-receipt /absolute/controller/controller_receipt.json \
  --controller-anchor-heartbeat /absolute/controller/heartbeats/00000000000000000001-00000000-0000-4000-8000-000000000001.json \
  --expected-controller-id taste-main-v2-UUID \
  --expected-git-commit COMMIT \
  --expected-git-tree TREE \
  --expected-controller-launcher-receipt-sha256 SHA256 \
  --expected-controller-receipt-sha256 SHA256 \
  --expected-controller-anchor-heartbeat-sha256 SHA256 \
  --gpu-lease /absolute/controller/gpu_leases/T4_ORACLE_SMOKE-UUID.json \
  --expected-gpu-lease-uuid UUID \
  --expected-gpu-lease-sha256 SHA256
```

The reviewed controller/runner environment records
`ALLOW_TASTE_T4_ADAPTIVE_CALIBRATION_SEARCH=1`,
`TASTE_T4_REQUIRE_BOTH_DESTINATIONS=0`, `TASTE_T4_MIN_STRICT_FLIPS=16`, and
`TASTE_T4_MIN_FLIPPED_PARENTS=8`. These policy values describe the frozen code
contract; changing them does not override the constants in an immutable
execution tree.

This repository change does not launch the command. A real successful terminal
verification uses `[TASTE_T4_ORACLE_SMOKE_PASS]`; a successful single-destination
run also emits `[TASTE_T4_SINGLE_DESTINATION_WARNING]` before PASS.
Implementation tests do not emit either runtime marker. T4 is an
oracle-interface prerequisite, not a paper matrix cell and not authorization
for a GNN ablation.
