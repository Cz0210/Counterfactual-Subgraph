# TasteMolNet T4 multiclass oracle smoke, managed v2

## Frozen boundary

This successor consumes the independently published managed-v2 T3 root:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/
tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed
```

The line break above is display-only. The physical path is one direct
`calibrated-*` child of the TasteMolNet `gine/seed7` artifact root. T4 accepts
only the generic managed `gate.json` plus its SHA-bound `verification.json`,
then validates the nested `tastemolnet_t3_calibration_v2` scientific PASS. Its
only checkpoint is `<T3>/artifacts/checkpoint`.

T4 opens only the authenticated graph-cache `manifest.json` and
`calibration.pt`. It never opens train, validation, test, or a CSV payload. It
uses physical GPU index 2, `CUDA_VISIBLE_DEVICES=2`, visible `cuda:0`, and the
controller-provided physical GPU UUID. The process verifies the index/UUID
again with `nvidia-smi` and requires exactly one visible CUDA device.

The deterministic smoke selects sixteen calibration rows with true label 1
and frozen-GINE prediction 1, retaining exactly four real connected one- or
two-atom deletions per parent. It validates every three-wide probability
vector, parent and deletion-pair batch/single parity, full-parent and invalid
deletion controls, and observed `1->0`, `1->2`, and no-flip outcomes. One model
object is constructed per scientific process.

## Publication ownership

The method worker writes only:

- four aggregate JSON documents;
- `t4_oracle_smoke_candidate.json`; and
- `sha256sums.txt`.

The generic managed worker adds only `raw_evidence.json`, `worker_exit.json`,
and `SEALED.json`. It cannot write PASS, a gate, or verification evidence.

A separate method verifier repeats the science from the retained T3 and cache
authorities. Only after the replay matches the SEALED candidate does the
managed-v2 terminal publisher write `verification.json`, generic `gate.json`,
and `PASS`, then atomically publish the complete directory with no replacement.
Every attempt, worker staging root, and generation token is a permanent UUID;
partial names are burned. `AUTO_TERMINATE_UNCONTROLLED_CHILDREN` is fixed to
`0`; any anomaly is quarantined without a process signal.

Published method evidence is aggregate-only. It contains no SMILES, molecule
ID, per-example prediction, graph payload, CSV, or matrix-cell claim.

## AutoDL worker launch

Run from the reviewed clean immutable checkout. First fill the physical graph
cache root and GPU-2 UUID. The four managed input hashes are shell-computable;
the T3 gate and verification recursively bind the model, fresh temperature,
feature schema, T2 receipt, and exact checkpoint inventory.

```bash
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
export AUTODL_PHYSICAL_GPU_INDEX=2
export AUTODL_PHYSICAL_GPU_UUID=GPU-REVIEWED-GPU2-UUID
export CUDA_VISIBLE_DEVICES=2

T3_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed
GRAPH_CACHE_ROOT=/absolute/existing/tastemolnet/graph-cache
STAGE_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-main-v2/stages/T4_ORACLE_SMOKE
CONTROLLER_ID=tastemolnet-main-v2
GIT_COMMIT="$(git rev-parse HEAD)"
CONFIG_HASH="$(sha256sum configs/hpc.yaml | awk '{print $1}')"
T3_GATE_HASH="$(sha256sum "$T3_ROOT/gate.json" | awk '{print $1}')"
T3_VERIFICATION_HASH="$(sha256sum "$T3_ROOT/verification.json" | awk '{print $1}')"
CACHE_MANIFEST_HASH="$(sha256sum "$GRAPH_CACHE_ROOT/manifest.json" | awk '{print $1}')"
CALIBRATION_CACHE_HASH="$(sha256sum "$GRAPH_CACHE_ROOT/calibration.pt" | awk '{print $1}')"

/root/miniconda3/envs/smiles_pip118/bin/python -B scripts/autodl/managed_worker_v2.py \
  --stage-root "$STAGE_ROOT" \
  --controller-id "$CONTROLLER_ID" \
  --task-id T4_ORACLE_SMOKE \
  --git-commit "$GIT_COMMIT" \
  --config-hash "$CONFIG_HASH" \
  --input-hash "t3_gate=$T3_GATE_HASH" \
  --input-hash "t3_verification=$T3_VERIFICATION_HASH" \
  --input-hash "graph_cache_manifest=$CACHE_MANIFEST_HASH" \
  --input-hash "calibration_cache=$CALIBRATION_CACHE_HASH" \
  --cwd "$PWD" \
  -- \
  /root/miniconda3/envs/smiles_pip118/bin/python -I -B \
    scripts/autodl/tastemolnet_t4_oracle_smoke_worker_v2.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --t3-root "$T3_ROOT" \
    --graph-cache-root "$GRAPH_CACHE_ROOT" \
    --gpu-uuid "$AUTODL_PHYSICAL_GPU_UUID" \
    --batch-size 32
```

Record the emitted `attempt_id`, `generation_token`, and `staging_path`. A
nonzero worker or anything other than `SEALED` is not verifier input.

## Independent verifier launch

Use the same clean commit, input authorities, GPU-2 UUID, and environment. The
final path must be a fresh direct `t4-oracle-smoke-*` sibling of T3.

```bash
SEALED=/absolute/value/from/worker/staging_path
ATTEMPT_ID=UUID-FROM-WORKER
GENERATION_TOKEN=UUID-FROM-WORKER
FINAL_PATH="$(dirname "$T3_ROOT")/t4-oracle-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

/root/miniconda3/envs/smiles_pip118/bin/python -I -B \
  scripts/autodl/tastemolnet_t4_oracle_smoke_verifier_v2.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --sealed "$SEALED" \
  --final-path "$FINAL_PATH" \
  --t3-root "$T3_ROOT" \
  --graph-cache-root "$GRAPH_CACHE_ROOT" \
  --gpu-uuid "$AUTODL_PHYSICAL_GPU_UUID" \
  --expected-attempt-id "$ATTEMPT_ID" \
  --expected-generation-token "$GENERATION_TOKEN" \
  --expected-controller-id "$CONTROLLER_ID" \
  --expected-git-commit "$GIT_COMMIT" \
  --batch-size 32
```

Success requires a generic managed PASS gate whose nested verification has
stage `T4_ORACLE_SMOKE`, marker `[TASTE_T4_ORACLE_SMOKE_PASS]`, both flip
destinations, 16 parents, 64 deletions, GPU2/UUID binding, and all access flags.
The scripts under `scripts/slurm/` are mandatory static CLI parity and always
refuse HPC science.

This stage is an oracle-interface smoke, not one of the four paper methods and
not authorization for a GNN ablation.
