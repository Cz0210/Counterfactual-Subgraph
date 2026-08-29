# AutoDL BACE GlobalGCE K20 raw-rule extension

## Scope

This route repairs the failed BACE GlobalGCE candidate-training stage without
changing the paper selector. The final candidate universe remains exactly
`K=20`. The extension only increases the number of official decoded raw
LHS-to-RHS rules available before native-rule validation and canonical
deduplication.

The bounded schedule is cumulative:

| Round | Cumulative seeds | Incremental raw budget | Cumulative budget |
|---|---|---:|---:|
| 1 | `7` | 80 | 80 |
| 2 | `7,17` | 120 from seed 17 | 200 |
| 3 | `7,17,27` | 300 from seed 27 | 500 |

The controller stops after the first round with at least 20 semantically unique
valid native rules. Candidate IDs and `native_rule_index` are excluded from the
semantic deduplication key. It never lowers K, duplicates a transformation,
accepts an invalid rule, or reads calibration/test data to decide whether to
continue.

## Safety and evidence

The controller:

- runs from an exact clean immutable Git worktree and hash-checks its critical
  implementation files;
- acquires `/autodl-fs/data/counterfactual-subgraph-runtime/locks/gpu-2.lock`
  with a non-blocking exclusive `flock` for the entire science-child lifetime;
- discovers and records the physical GPU2 UUID itself and requires GPU2 to be
  idle at launch; it does not accept a caller-signed GPU UUID;
- binds the protected GPU0 BACE GCFExplainer and GPU3 BACE ComRecGC process
  generations to the live `nvidia-smi` GPU UUID/PID inventory and expected task
  command roles before launch and throughout every child;
- sets `CUDA_VISIBLE_DEVICES=2`, while the isolated child uses `cuda:0`;
- samples the complete GPU2 compute-PID set at every heartbeat, rejects any PID
  other than the bound science child, and requires the set to be empty after
  child exit and at release;
- holds no child-signal authority and never calls a process termination API;
  before any OS thread exists it process-wide blocks SIGINT, SIGTERM, and
  SIGHUP, synchronously drains them into a deferred stop request, and keeps the
  lock until the child exits naturally; the exec'd raw-round process restores
  normal delivery for itself and never receives a controller-forwarded signal;
- uses only the frozen 360-parent train source manifest and native train CSV,
  the same frozen BACE GINE, min-frequency 7, 100 epochs, and pinned official
  GlobalGCE implementation;
- launches fresh exact-top-k raw-rule roots for each incremental seed/budget;
- accepts a nonzero child only as exit code 20 plus a structured, hash-closed
  `K20_RAW_ROUND.json` receipt and final `RAW_SHORTFALL` marker; log phrases and
  arbitrary files named `PASS` are not evidence;
- reopens every catalog row through `GlobalGCENativeRule.from_payload`,
  recomputes the native content hash and selector chemistry, then deduplicates
  the transformation payload without caller-controlled ID/index fields;
- re-audits Git, config, Python, all train-contract manifests and hashes, every
  required frozen-GINE bundle file, and every tracked official Python source
  (including `GTGNN.py` and `gSpan.py`) after every child and twice at release;
  official imports require the captured inode/byte/hash authority and the final
  signal-drain/check occurs immediately before `PASS`.

The child PID/start-tick identity and fresh round root are printed with
`[BACE_GLOBALGCE_K20_EXTENSION_LAUNCHED]`. This is a launch marker, not a
scientific PASS. Pre-marker JSON uses `SEALED_CANDIDATE` or `RELEASE`, never
`PASS`, and no `_RUN_COMPLETE.json` is created. Only an exactly 20-row,
semantic-unique, hard-validated publication may write
`[BACE_GLOBALGCE_K20_PASS]`; that marker is the final filesystem write.

## AutoDL command

Run through `nohup` or `tmux` from a reviewed immutable execution checkout.
All paths below must be replaced with reviewed physical paths; the output root
must be absent and be a direct child of the `bace_globalgce_k20` namespace.

```bash
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
export PYTHONDONTWRITEBYTECODE=1

/root/miniconda3/envs/smiles_pip118/bin/python -I -B \
  scripts/autodl/run_bace_globalgce_k20_extension.py \
  --config "$PWD/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  controller \
  --controller-id bace-globalgce-k20-UUID \
  --output-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/bace_globalgce_k20/run-UUID \
  --source-manifest /absolute/train/source_manifest.jsonl \
  --native-train-csv /absolute/train/native.csv \
  --official-root /absolute/pinned/GlobalGCE \
  --gnn-checkpoint /absolute/frozen/bace/gine \
  --protected-gpu0-process GPU0_PID:GPU0_START_TICKS \
  --protected-gpu3-process GPU3_PID:GPU3_START_TICKS
```

The paired Slurm file exists for CLI parity and exits 75 unconditionally before
its documentation-only command. Slurm can allocate an arbitrary visible GPU,
so it cannot establish this route's physical GPU2 contract. The paper recovery
run is an AutoDL physical-index execution, not an HPC job or GNN ablation.

## Release state

Code review and local focused tests are prerequisites only. Do not claim either
runtime marker until the exact AutoDL child/root and final immutable artifacts
exist. A failed round or fewer than 20 unique rules after raw budget 500 is a
real blocker and does not authorize a smaller K or fabricated replacement.
