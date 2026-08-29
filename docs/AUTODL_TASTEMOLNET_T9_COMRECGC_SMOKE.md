# AutoDL TasteMolNet T9 COMRECGC smoke

## Current route

T9 now has a runnable bounded successor under the explicit execution semantic
`TRUSTED_SINGLE_OPERATOR_ROOT`. This removes the unfinished mutable-controller
receipt from the launch boundary; it does not relax scientific provenance.
The worker and the independent verifier separately retain and cross-bind:

- the fresh seven-file T2 adoption receipt and its three supplied SHA-256 pins;
- managed-v2 T3 and T4 PASS roots, which must bind one T2 receipt and one GINE;
- the exact frozen GINE checkpoint inventory and payload hashes;
- the checkpoint-declared train CSV and its SHA-256;
- this clean Git commit/tree and `configs/hpc.yaml` bytes; and
- official COMRECGC commit
  `122f9341a360e9f06bb58a2f5823bb596021f6bf` plus all seven reviewed source
  hashes.

The old `tastemolnet_t9_comrecgc_smoke_release_v1.json` remains immutable
historical GPU2/controller-draft evidence and stays disabled. It is not an
alternate release path. The successor runs on physical GPU1 only, after T4,
through the existing UUID-scoped AutoDL GPU lock. GPU0 and GPU3 are untouched.

## Scientific contract

This is a smoke, not a full or paper result:

- three-class TasteMolNet, source `Sweet=1`;
- random-walk importance `1-p(Sweet)`;
- counterfactual membership `argmax([p0,p1,p2]) != 1`;
- canonical attributed-graph SHA-256 identity, independent of parent metadata,
  lineage, embeddings, and Python `hash()`;
- first eight distinct GINE-predicted Sweet sources from the first 64 sorted
  train-only Sweet rows;
- native serial COMRECGC with five heads, seed 7, `M=500`, and an actual
  checkpoint/reload after fully completed step 250;
- candidate capacity 2048, sample size 10000, teleport 0.1, theta 0.1,
  delta 0.02, DBSCAN min-samples 3, and common-recourse size 5;
- official source-major pair order, medoid, coverage, and greedy ordering; and
- `M=50000`, RF, validation payload, calibration payload, test, and paper
  eligibility remain false.

The terminal contains aggregate JSON only. It contains no SMILES, molecule ID,
graph tensor, checkpoint, row-level prediction, or reconstructable dataset.

## Managed-v2 publication

The scientific worker creates a never-reused UUID attempt/staging generation,
writes only:

```text
artifacts/comrecgc_smoke.json
artifacts/input_authority.json
raw_evidence.json
worker_exit.json
SEALED.json
```

plus managed generation tokens. It cannot write verification, gate, or PASS.
A separate verifier process reopens the SEALED inventory and all scientific
inputs, validates the exact M500/checkpoint250 evidence, then calls the existing
managed-v2 atomic no-replace publisher. The final outer marker is
`[MANAGED_EXECUTION_V2_PASS]`; the exact method marker
`[TASTE_T9_COMRECGC_SMOKE_PASS]` is nested in independent verification and
printed only after publication.

Same-filesystem publication is a single no-replace directory rename.
Cross-filesystem publication uses the existing copy/fsync/rehash plus
no-replace directory rename path. No partial final directory is reusable.

## AutoDL invocation

Set the existing scoped Taste permissions and the following exact paths/hashes:

```bash
export RUN_TASTEMOLNET=1
export TASTE_RESEARCH_COMPUTE_ALLOWED=1
export TASTE_PAPER_RESULTS_ALLOWED=1
export TASTE_DATA_REDISTRIBUTION_ALLOWED=0
export RUN_GNN_ABLATION=0
export TASTEMOLNET_T9_STAGE_ROOT=/persistent/fresh/t9-managed-v2
export TASTEMOLNET_T9_OUTPUT=/persistent/fresh/t9-final
export TASTEMOLNET_T9_RUN_ID=taste-t9-comrecgc-m500-v1
# T2_GATE is gate.json SHA-256; T2_RECEIPT is the canonical SHA-256 of
# the complete seven-file {name: sha256} map; T2_SOURCE is the SHA-256 of
# source_evidence.json itself.
export TASTEMOLNET_T2_ADOPTION_ROOT=/persistent/exact/t2-receipt
export TASTEMOLNET_T2_ADOPTION_GATE_SHA256=<T2_GATE>
export TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256=<T2_RECEIPT>
export TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256=<T2_SOURCE>
export TASTEMOLNET_T3_OUTPUT_ROOT=/persistent/exact/t3-managed-final
export TASTEMOLNET_T4_OUTPUT_ROOT=/persistent/exact/t4-managed-final
export TASTEMOLNET_TRAIN_CSV=/persistent/exact/train.csv
export COMRECGC_OFFICIAL_ROOT=/persistent/exact/official-comrecgc
bash scripts/autodl/run_tastemolnet_comrecgc_smoke.sh
```

The checkpoint is not a separately selectable input: the wrapper derives it
as `TASTEMOLNET_T3_OUTPUT_ROOT/artifacts/checkpoint`, and the retained T3
authority verifies every payload before either process uses it.

The wrapper requires T4 to exist, waits only for the GPU inventory's bounded
idle check, takes physical GPU1's standard UUID lock, and holds it across the
worker, SEALED handoff, independent verifier, and atomic publication. It never
signals another task. If GPU1 is not idle, it exits 75 for a later operator
retry; it does not poll indefinitely.

The paired Slurm entrypoints are static AutoDL-only refusals and exist solely
for CLI synchronization. No HPC submission is authorized.
