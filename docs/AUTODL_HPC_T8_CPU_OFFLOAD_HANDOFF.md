# AutoDL to HPC Taste GlobalGCE T8 CPU-offload handoff

## 2026-09-04 storage-safe merge/package replacement

The legacy merge/package chain materializes a full uncompressed merge under
`/share` and then creates another uncompressed tar.  Keep those jobs held when
the user-path free space cannot cover that peak.  Do not modify successful
array shards and do not release the legacy chain merely to test this route.

The replacement is
`scripts/hpc/t8/slurm_storage_safe_merge_package.sh`.  It is one CPU-only
`afterok:${T8_FULL_ARRAY_JOB_ID}` job.  Exact merge, duplicate checks, complete
partition/order validation, and deterministic gzip creation occur in
`$SLURM_TMPDIR` (or a task-private `/tmp` directory).  The fresh persistent
`T8_STORAGE_SAFE_RESULT_ROOT` ends with exactly:

```text
t8_exact_result_bundle.tar.gz
result_manifest.json
```

Required bindings are:

```text
T8_FULL_ARRAY_JOB_ID
T8_EXECUTION_WORKTREE
T8_EXPECTED_COMMIT
T8_PYTHON
T8_INPUT_MANIFEST
T8_EXPECTED_INPUT_MANIFEST_SHA256
T8_EXPECTED_CONFIG_SHA256
T8_EXPECTED_HPC_CONFIG_SHA256
T8_PARTITION_MANIFEST
T8_EXPECTED_PARTITION_MANIFEST_SHA256
T8_FULL_SHARDS_ROOT
T8_CANARY_PARITY_RECEIPT
T8_EXPECTED_CANARY_PARITY_SHA256
T8_STORAGE_SAFE_RESULT_ROOT
T8_ENVIRONMENT_MANIFEST
T8_SLURM_INVENTORY
T8_RESOURCE_METRICS
```

`T8_EXPECTED_COMMIT` is the immutable packaging-code commit.  The full
partition manifest separately retains the original exact-science commit and
input identities.  Optional admission settings default to
`T8_MIN_PERSISTENT_RESERVE_BYTES=2147483648` and
`T8_PERSISTENT_RESERVE_FRACTION=0.20`; do not weaken them for production.

After transfer, AutoDL runs
`scripts/hpc/t8/stream_verify_storage_safe_bundle.py` with externally recorded
archive SHA, packaging commit, scientific-input SHA, and full-partition SHA.
It streams every member, recomputes each raw JSONL hash/count, checks typed DFS
identities and official preorder, and never extracts the archive.  PASS allows
only a fresh-root import; it does not authorize GINE inference,
calibration/test, or matrix publication on HPC.

## 1. Purpose and authority boundary

This route moves only the exhaustive CPU gSpan stage of the single authorized
TasteMolNet GlobalGCE T8/T13-grade recovery from AutoDL to the Tongji HPC.  It
does not replace the frozen classifier, change typed chemistry, select rules,
evaluate calibration/test data, or publish a matrix cell.

The priority and ownership rules are fixed:

1. the AutoDL 4 x 4 main campaign remains the scientific authority;
2. the HPC may create a candidate/result bundle only;
3. AutoDL independently verifies and imports that bundle into a fresh root;
4. only the unique AutoDL `fast16_matrix_authority` may publish a cell;
5. no HPC command receives a matrix-authority path or matrix write capability.

The route is fail-closed.  A faster wall clock, a matching top-K count, or a
plausible pattern catalog is not parity evidence.

## 2. Audited snapshot (2026-09-03)

This is a dated inventory, not a substitute for the live status command.

### Mac relay host

- macOS 13.3.1, arm64.
- Local proxy endpoint: `127.0.0.1:7897`; TCP, HTTP CONNECT, and SOCKS checks
  reached GitHub during the audit.
- `tongji-hpc` uses the existing ControlMaster and a reverse tunnel from HPC
  `127.0.0.1:39394` to the Mac proxy.  Agent forwarding is disabled.
- `autodl-a800` is a key-only, BatchMode-capable alias.  Host-key policy is
  `ask`; it is never weakened to `StrictHostKeyChecking=no`.
- External relay disk: `/Volumes/DireRaven/counterfactual-hpc-offload`, with
  about 413 GiB free after the local cleanup.  It is the preferred transfer
  staging area; it is not a scientific authority.

### Tongji HPC

- Login host audit: 96 logical Xeon Max 9468 CPUs, about 629 GiB RAM and
  about 530 GiB available at inspection time.
- The global `/share` filesystem had about 20 PiB free, but the path-specific
  user view for `/share/home/u20526/czx` exposed only about 11 GiB free after
  cleanup. Treat that smaller number as the admission limit: use compute-node
  `/tmp` for fine-grained mining state and persist only sealed checkpoints and
  compact results.
- `/ssdfs/datahome/u20526` had only about 1.3 GiB free and was 100% utilized.
  **Do not place the worktree, input bundle, checkpoints, shard output, merge
  output, Python environment, or temporary files on `/ssdfs`.**
- No general `/scratch` mount was present on the login node.  Use Slurm's
  node-local `$SLURM_TMPDIR` when the job provides it; on the audited `intel`
  compute node it was empty while `${TMPDIR:-/tmp}` resolved to a roughly
  795-GiB local filesystem, so wrappers fall back to a task-private `mktemp`
  directory there.  They do not use the capacity-limited `/share` path for
  high-frequency transient state.
- The usable Python is
  `/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python` (Python 3.10,
  RDKit and PyG present at audit time).
- Slurm is available.  The confirmed CPU partition is `intel`; the T8 wrappers
  default to it, allow an explicit partition override, and never request
  `--gres=gpu`.
- The ordinary HPC repository was dirty and is not an execution root.  The
  offload must use a separate detached or branch-pinned worktree.

### AutoDL main campaign

- The unique matrix authority reported 12/16 cells at the audit point.  The
  incomplete cells were Mutagenicity/ComRecGC and TasteMolNet
  GCFExplainer/GlobalGCE/ComRecGC; the exact live set must be reopened before
  any import.
- The active T8/T13-grade recovery and the earlier serial canary were still
  working in root 0.  Both are protected: submission of an HPC canary is not
  permission to signal, restart, or write into either root.
- T12 and T14 have their own recovery/parity gates.  This offload does not
  change or restart them.
- LLM and GNN ablation work remains lower priority than the main matrix.

## 3. One Git source of truth

The implementation, input manifest, Slurm jobs, merge, and result-bundle
builder must all bind one exact Git commit.  A branch name is transport
metadata and is not an execution identity.

The allowed synchronization order is:

1. build and test in the isolated Mac worktree;
2. commit to the private feature branch;
3. push/fetch that commit on the HPC through the existing proxy tunnel;
4. create an isolated HPC worktree at the exact commit;
5. record `git rev-parse HEAD` in every Slurm and result manifest.

If authenticated Git fetch is unavailable, create a Git bundle containing the
same feature commit, relay it through the external disk, verify its SHA-256 on
HPC, and create the worktree from the bundled commit.  Do not merge three
independent Mac, AutoDL, and HPC histories.  Do not pull into either dirty
repository, push `main`, force-push, `reset --hard`, or `clean`.

The pinned official GlobalGCE/gSpan source remains commit
`157e65c2850bc787f229a1ee8c60564906b933f2`.  The execution manifest must bind
both the project commit and this submodule commit.

## 4. Minimal, train-only input bundle

Only immutable data needed by the CPU miner may leave AutoDL.  The audited
primary graph input is:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/audits/
t8_root_sharding_canary_b65bc40/production_graphs_target0.jsonl
```

At the inventory point it had:

- SHA-256 `7e0d77f4a64d9f2ef0d2be7bab625927854f5baa159bc8648741725b8b4cdc1d`;
- 4,435,201 bytes;
- 3,058 graph records;
- regular-file, non-symlink, stable-mtime, no-open-writer evidence.

The accompanying frozen input manifest had SHA-256
`651f2a319ed28d474e083b8b54aeaad8082691458b128eefb55fd901e69158f3`.
The prepared train CSV is **not transferred**.  Its SHA-256 remains a
provenance-only CLI/source-manifest binding.  The train CSV and transferred
split-manifest hashes audited at this point were respectively
`eac05f7003c37a24554aa2c22e1051edb90eb4a12f9b62ae6fd47d73efa59564`
and `841f3b...51af`.  The bundle builder must record the complete values it
actually verifies rather than copying this abbreviated documentation value.

The graph database is target-independent for this full T13-grade route.  The
bundle therefore carries one frozen transaction JSONL plus an explicit
dual-target binding for `source_label=1`, `targets=[0,2]`; it must not pretend
that two distinct graph databases were exported.

The bundle allowlist is deliberately small:

- frozen graph JSONL;
- exact input/provenance manifests and detached SHA-256 identities (the raw
  train CSV remains excluded);
- project-owner authorization receipt overriding the older no-HPC-copy policy
  only for this private, train-only, non-redistributed offload;
- graph codec/feature-schema identities and the calibrated GINE identity as
  hashes or small manifests, never model weights;
- run contract: support 2, minimum vertices 3, maximum vertices 20, K=20,
  seed 7, 100 epochs, targets 0 and 2, and exact/no-prune mining.

It must contain no calibration/test rows, SMILES source corpus beyond the
frozen derived transaction JSONL, GINE/ChemLLM weights, active SQLite/WAL/SHM,
live journals, temporary checkpoints, credentials, or a runtime-tree copy.
Never copy the active sibling `canary-root0-root22/` directory.

Transfer is strictly:

```text
AutoDL fresh export root
  -> /Volumes/DireRaven/counterfactual-hpc-offload
  -> HPC fresh input root under /share
```

Use resumable copies without `--delete`.  At each boundary verify the detached
bundle SHA-256, the inner manifest self-hash, every allowlisted file hash, the
train-only split binding, regular-file/no-symlink policy, and total byte count.

## 5. Exact parity gate

The first Slurm work is a bounded canary, not the full array.  The exact
reference and sharded implementations must consume the same immutable bundle
and exact project/official-source commits.

The canary comparison includes, for every selected real root and in stable DFS
order:

- typed DFS code and canonical graph identity;
- emitted pattern identity and support;
- candidate-input identity;
- every typed rejection category and count;
- ordered accepted catalog;
- exact K=20 result and its deterministic tie-breaking inputs;
- checkpoint reload and replay at a prefix-partition boundary;
- complete serial and sharded manifests plus SHA-256 closure.

The exhaustive route uses no scientific pruning.  Work may be partitioned only
at deterministic typed DFS prefixes.  A lost shard resumes from a completed
prefix boundary or replays the active prefix; it does not skip an uncommitted
subtree.  The merge orders typed prefixes and records the complete partition
coverage proof before computing a deterministic union.

The full array may be submitted only when the independent parity verifier says
PASS with no first semantic divergence.  Approximate support, tolerance-based
pattern equality, ANN, heuristic truncation, root-count substitution, or a
top-K-only comparison cannot authorize production.

Canary PASS alone is not a storage or wall-time admission receipt.  Before a
full array is submitted, use the canary's observed bytes/events/patterns and
per-unit runtime to show that the longest planned prefix fits the 48-hour
array wall time and that sealed shards plus merge and bundle staging fit the
path-specific free-space budget with reserve.  Until those measurements exist,
the full array remains intentionally unsubmitted.

## 6. HPC filesystem and Slurm rules

Recommended roots are:

```text
/share/home/u20526/czx/worktrees/t8-hpc-cpu-offload-v1
/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/inputs/<bundle-sha>/
/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/canary/<attempt>/
/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/shards/<attempt>/
/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/merge/<attempt>/
```

The wrappers under `scripts/hpc/t8/` are intentionally CPU-only.  This is a
task-specific exception to the repository's ordinary A800 one-GPU Slurm
baseline: they default to the audited `intel` CPU partition, accept an explicit
partition override, request no GPU, and never import a frozen GINE for
inference.  The canary is statically capped at 64 GiB and one hour.  Keep full
jobs bounded by explicit CPU, memory, time, array-size, and array-concurrency
settings.  Run only compute-light status and transfer commands on the login
node.

The bounded canary and final merge are fresh, fail-closed jobs and do not ask
Slurm to requeue them.  Only the full array is requeueable, because its writer
can reopen sealed persistent prefix boundaries.  A failed canary or merge is
retried under a fresh root after preserving the old evidence.

All outputs are fresh.  A shard writes its own directory and checkpoint; no two
array tasks share a writer.  Merge and result packaging are dependency jobs
that read only completed shards.  Neither receives AutoDL credentials or a
matrix path.

### Wrapper and bundle contracts

`scripts/hpc/t8/build_input_bundle.py` never walks a directory.  The operator
must supply exactly one `--source ROLE=PATH` and one
`--expected-sha256 ROLE=SHA256` for each of `graph_jsonl`,
`source_input_manifest`, `split_manifest`,
`train_cohort_manifest`, `feature_schema`, and `data_use_authorization`, plus
one or more explicit `--allowed-source-root` values.  The non-transferred
native train CSV is represented only by required
`--native-train-csv-sha256` provenance.  The project-owner
authorization is itself hash-checked and must bind the exact source graph,
train-only scope, source label 1, targets 0 and 2, shared target-independent
transaction database, no redistribution, no calibration/test payload, and no
HPC matrix publication.  The fixed science contract is seed 7, 100 epochs,
support 2, vertex range 3--20, K=20, and 50 roots.  Any different value fails
before a tar is written.

The primary composite wrappers require explicit environment bindings rather
than inferring a live root.  All three require
`T8_EXECUTION_WORKTREE`, `T8_EXPECTED_COMMIT`, `T8_PYTHON`,
`T8_INPUT_MANIFEST`, `T8_EXPECTED_INPUT_MANIFEST_SHA256`,
`T8_EXPECTED_CONFIG_SHA256` (the scientific mining-config hash), and
`T8_EXPECTED_HPC_CONFIG_SHA256` (the byte hash of `configs/hpc.yaml`).  The
canary additionally requires the frozen graph JSONL, official source checkout,
and fresh canary root.  The production array additionally requires the
full-universe partition manifest and hash, shard count, bounded concurrency,
and fresh shards root.  The merge requires the successful array job ID, the
SHA-pinned canary parity receipt, fresh merge/result paths, and submits itself
with `afterok:<array-job-id>`.  Result packaging additionally requires three
regular, immutable JSON evidence files through `T8_ENVIRONMENT_MANIFEST`,
`T8_SLURM_INVENTORY`, and `T8_RESOURCE_METRICS`; the bundle records their byte
and canonical-content hashes.

Invoke the composite files with `bash` to let them choose an explicitly
provided `HPC_CPU_PARTITION` or the first available ordinary CPU partition,
falling back to the audited `intel` CPU partition.
Direct `sbatch --partition=...` remains supported.  Canonical shard directories
are zero-padded (`shard-000`, `shard-001`, ...); requeued array elements reopen
only their own root and rely on the miner's committed prefix boundaries.  The
canary covers exactly one fixed canonical prefix below root 0 plus the complete
small root 22.  `T8_CANARY_PREFIX_UNIT_ID` may pin an already audited root-0
prefix; when unset, the wrapper enumerates root 0 with the pinned official DFS
implementation and deterministically selects the minimum
`(support_hint, partition_id)` `PREFIX_SUBTREE`, favoring the bounded real
prefix.  The bounded manifest validates either selection, and the wrapper
atomically writes a self-hashed `canary_prefix_selection.json` receipt.
Mining and merge transient state is staged under
`$SLURM_TMPDIR`, falling back to a task-private `mktemp` directory under
`${TMPDIR:-/tmp}`, and only validated sealed outputs are copied to durable
shared storage.

## 7. AutoDL import and old-worker handover

Returning an HPC tarball does not close T8.  AutoDL must import into a fresh,
non-active root and independently verify:

1. outer bundle SHA-256 and inner self-hashed manifest;
2. exact project and official GlobalGCE commits;
3. exact input bundle identity and run contract;
4. parity receipt and absence of semantic divergence;
5. complete typed-prefix coverage and deterministic merge;
6. catalog/rule artifacts, hashes, record counts, and no active writer;
7. unchanged three-class GINE, train split, feature schema, source/targets,
   typed chemistry, no calibration/test training leakage;
8. native post-mining validation required by the existing T8/T13 route.

Only after that import is independently PASS may an operator evaluate a
handover.  Immediately before any signal, AutoDL must re-read and match the old
worker's PID, `/proc` start ticks, full command hash, cwd, output root,
controller ownership, last committed checkpoint, and live writer set.  Save a
final progress/output snapshot, request a graceful checkpoint if supported,
send `SIGTERM` to that one exact PID, wait, and verify the preserved old root.
Never use `pkill`, `killall`, a process group, a fuzzy match, or `SIGKILL`.

If the old AutoDL worker naturally finishes or publishes a scientifically
valid result first, it remains authoritative and the HPC result becomes
redundant evidence.  The HPC result must not create a conflicting matrix
append.  A successful HPC import still does not publish directly: the normal
AutoDL verifier/finalizer and the unique matrix authority own publication.

## 8. Recovery and status commands

From the isolated Mac worktree:

```bash
bash scripts/local/status_hpc_autodl_offload.sh
```

Optional status paths can expose task state without process command lines:

```bash
AUTODL_T8_STATUS_PATH=/absolute/path/to/t8/heartbeat.json \
AUTODL_T12_STATUS_PATH=/absolute/path/to/t12/heartbeat.json \
AUTODL_T14_STATUS_PATH=/absolute/path/to/t14/heartbeat.json \
AUTODL_MUT_STATUS_PATH=/absolute/path/to/mut/heartbeat.json \
bash scripts/local/status_hpc_autodl_offload.sh
```

The status tool is read-only and redacted.  It reports connectivity, pinned
worktree commit, T8-named Slurm jobs, matrix counts, aggregate GPU memory, and
optional task states.  It never runs `ssh -vv`, dumps `ssh -G`, reads private
keys/tokens, prints process command lines, or modifies any host.

Before any HPC action, confirm the existing master is alive:

```bash
ssh -S /tmp/tongji-codex.sock -O check tongji-hpc
```

If it is not alive, recreate the ordinary authenticated session manually; do
not use password automation.  On HPC, verify the reverse proxy without
printing credentials:

```bash
curl --fail --silent --show-error \
  --proxy http://127.0.0.1:39394 \
  https://github.com/ -o /dev/null
```

Inspect only the offload jobs:

```bash
ssh tongji-hpc \
  "squeue -u \$(id -un) -o '%i %j %T %M %R' | grep -E 'JOBID|t8|globalgce'"
```

Re-submit only a failed/missing stage after reopening its immutable manifest:

```bash
sbatch scripts/hpc/t8/slurm_canary.sh
sbatch --array=<missing-shards>%<concurrency> scripts/hpc/t8/slurm_array.sh
sbatch --dependency=afterok:<array-job-id> scripts/hpc/t8/slurm_merge.sh
```

Exact arguments and roots must come from the generated run manifest; the
placeholders above are not permission to guess them.  A queued or running
hour-scale job should be left under Slurm ownership rather than polled from a
Codex turn.

## 9. Mac storage cleanup record

The local audit found an incomplete two-shard ChemLLM copy (about 14.4 GiB)
whose exact files already existed on the external disk with matching hashes,
plus an unreferenced `checkpoint-400` (about 74.5 MiB).  They were moved—not
irreversibly erased—under:

```text
/Volumes/DireRaven/counterfactual-local-cleanup-20260903/
```

The source paths are absent and the recovery copies remain available.  The
local checkpoint-300/checkpoint-500, RF, MolCLR, CLEAR, and an unrelated Qwen
model without a verified backup were retained.  Do not purge the recovery copy
until the user explicitly chooses permanent deletion after a second inventory.

## 10. HPC scoped cleanup record

The user separately authorized cleanup **only** below
`/share/home/u20526/czx`. A read-only Slurm/process/Git audit was completed
before deletion. The cleanup removed 4,409,661,901 bytes in total:

- eight stale `*.partial.csv` files byte-identical to retained final CSVs;
- sixteen root-level and six `transfer/` Git bundles whose commits remain in
  the retained repositories;
- thirteen registered, clean, inactive worktrees whose branches/commits remain
  in the main Git repository;
- one `...final copy` directory recursively identical to its retained sibling.

The detailed pre-delete manifests are retained at
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/cleanup-audits/20260903T103500Z/`.
No path outside `/share/home/u20526/czx` was modified. Dirty worktrees, model
weights, scientific results/checkpoints, package caches with uncertain reuse,
and partial files without a final sibling were kept. The path-specific free
space increased from about 8.9 GiB to about 13 GiB.

## 11. Failure policy

- An SSH password prompt or missing key authentication requires a manual user
  login; never add a password to a script.
- A disconnected status check reports `UNREACHABLE` and exits without retries
  or writes.
- A bundle/hash/split/parity mismatch blocks the full array or import.
- A job waiting for an hour is not a reason to poll continuously.
- A full `/ssdfs` or missing `$SLURM_TMPDIR` is not permission to redirect to
  an unreviewed filesystem.
- No offload failure may restart or preempt AutoDL main-table science.
