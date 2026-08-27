# AutoDL TasteMolNet T2 GINE PASS Adoption v1

## Scope

This successor records a completed TasteMolNet GINE training result without
repairing or rewriting its failed controller. It is an evidence adoption, not
a trainer, resume path, controller reconciliation, calibration run, matrix
registration, or scientific-output publisher.

The frozen source identity is:

- execution commit: `583bf668896142d8cc292cd624fbbffc20faf688`;
- identity-classification fix commit:
  `3a90fd8697b58bad4f95f3be9347b327d5c51043`;
- controller CID:
  `tastemolnet_gine_v2_20260827T160626Z_583bf668`;
- controller root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-gine-v2/tastemolnet_gine_v2_20260827T160626Z_583bf668`;
- separate training-state root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-gine-training-v2/tastemolnet_gine_v2_20260827T160626Z_583bf668`;
- formal output root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/full-20260827T160626Z`;
- run ID:
  `20260827T160732Z-tastemolnet-TASTEMOLNET_GINE_FULL_RESEARCH_V1-87809`;
- scientific state: complete three-class GINE output and training-state
  closure;
- controller state: `FAILED`, with exact historical reason
  `WORKER_PROCESS_IDENTITY_DRIFT`;
- process requirement: the deployed FAILED state has the exact terminal schema
  emitted by commit `583bf` (and therefore no invented generation fields), the
  exact trainer authority contains native-integer `exp_run`/trainer
  PID/start/PPID snapshots, and every declared PID is absent from the held,
  physical production `/proc` authority. PID reuse is treated as live, never
  as proof that the reviewed generation exited.

The failed controller is not changed to PASS. Its
`FAILED/WORKER_PROCESS_IDENTITY_DRIFT` value remains accurate control-plane
provenance and is explicitly recorded as a scientific false negative. The
independent authorities instead say that the registry event, runtime state,
runtime log, `training_complete.json`, and closed scientific bundle are PASS.
The receipt preserves both facts; it never turns either one into the other.

## Exact destination and publication boundary

The destination is derived rather than selected by a CLI argument:

```text
<control_root>/tastemolnet-t2-gine-pass-adoption-v1/
  tastemolnet_gine_v2_20260827T160626Z_583bf668
```

This is a fresh, non-scientific, independent control namespace. It is
disjoint from the old controller, old training-state root, old scientific
output, source worktrees, main controller, GPU locks, and matrix. An optional
`--assert-adoption-root` can only assert the derived path; it cannot redirect
publication.

Publication is a fresh one-shot operation. Once the final CID directory
exists, no retry, resume, cleanup, reconciliation, or overwrite is allowed.
An interrupted partial root remains diagnostic evidence and requires a newly
versioned successor after review.

## Retained source authority

The validator walks absolute directory components with `openat` semantics and
`O_NOFOLLOW`, retains the resulting descriptors, and rechecks every named
inode. It retains every file and directory descriptor in the controller,
output, training-state, and run-state inventories. Selected registry, source,
configuration, trainer-authority, completion, and Git-bound files are held as
physical regular files. Publication revalidates all retained sources before
each file and through the final gate commit. Its only mutations are creation
of the exact adoption namespace when absent, its exact fresh CID child, and
the five no-clobber files; after CID creation, file writes use only that held
directory descriptor. The old controller, output, training-state, run-state,
registry, and runtime log remain held read-only.

The first four receipt files remain open as held physical file descriptors.
The gate binds each one's device, inode, mode, UID, GID, link count, size, and
SHA-256, as well as the held adoption-directory identity. `gate.json` is first
written and synced under a non-authorizing prepared name. Only after the four
files, prepared gate, destination directory, release receipt, Git authorities,
and every old source have passed their last checks is
`renameat2(RENAME_NOREPLACE)` used to expose `gate.json`. That rename is the
terminal publication operation: there is no later validation or fsync, and
descriptor cleanup is non-throwing best effort.

The main controller and paper matrix are neither opened nor written. Focused
tests snapshot all of these protected fixture authorities before and after
publication and require identity/content equality.

The source closure requires all of the following:

- exact failed controller state/reason and exact terminal field set,
  controller claim, canonical spec, clean `583bf` execution commit/tree, and
  clean `3a90` identity-fix commit/tree;
- exact config hashes and reviewed wrapper/controller/GINE config hashes;
- typed trainer-child authority bound to source CID, root, run ID, command
  hash, exact durable startup-barrier record/lock, exact wrapper `exp_run`
  argv, and parent/child process snapshots; the reviewed raw Python argv token
  remains distinct from its recorded physical executable identity, matching
  the `3a90` correction;
- all process fields use exact native integer types; all declared PIDs are
  absent from literal production `/proc`, which is not CLI-configurable;
- final matching run-registry event at `PASS` with `exit_code=0`;
- run-state runtime marker at `PASS` with `exit_code=0`, plus the held runtime
  log containing the GINE PASS, molecular-training OK, and AutoDL exit-0
  markers;
- `training_complete.json`, canonical training contract, latest checkpoint,
  output identity, and latest/`last.pt` hash closure;
- exactly nineteen flat output files, with exactly eighteen unique non-self
  entries in `sha256sums.txt` and every content hash matching;
- full physical output and training-state inventories;
- three classes (`Bitter`, `Sweet`, `Tasteless`), `source_label=1`, GINE,
  seed 7, all three predicted classes, positive available per-class recalls,
  checkpoint reload PASS, no RF, train/validation cache use only, no
  calibration/test cache load, policy-v2
  `research_compute_allowed=true`/
  `paper_result_reporting_allowed=true`/
  `data_redistribution_allowed=false`, and no licence-PASS claim.

## Five-file receipt

The final directory contains exactly:

1. `input_hashes.json` — full retained source evidence and release pins;
2. `state.json` — typed `T2_GINE_FULL_PASS_ADOPTED` stage state;
3. `manifest.json` — the adoption receipt;
4. `output_hashes.json` — hashes of the first three files;
5. `gate.json` — terminal PASS hash and physical-inode closure over the
   preceding four held files and their held directory.

There is deliberately no generic `PASS` file. `manifest.json` is the receipt.
T3 calibration may depend only on (a) the validated fresh `gate.json`, whose
`receipt_sha256` equals `SHA256(manifest.json)`, and (b) the exact formal GINE
bundle root, inventory SHA-256, and model SHA-256 recorded in that receipt. It
may not consume the old
controller state, a main-controller shortcut, or matrix state as alternate T2
authority. T4 binds the same receipt SHA and still requires T3's own gate. The
output-hash DAG excludes the gate itself to avoid a self-hash cycle. Every
JSON file uses one deterministic canonical encoding; even same-meaning
whitespace changes fail status validation.

Downstream stages use `hold_t2_gine_pass_adoption(...)` with exactly four
authorities: the fresh adoption root, expected `gate.json` SHA-256, expected
`manifest.json` SHA-256, and expected embedded source-evidence SHA-256. The
holder validates the exact five-file set, canonical JSON/hash DAG, physical
publication binding, fixed source CID/run/commit identities, and the formal
19-file GINE inventory. It deliberately does not reopen the historical GINE
controller, training-state, execution, or identity-fix roots. T3 retains this
holder through marker publication and records its complete downstream binding;
T4 requires byte-for-byte equality with T3. T5 and T6 independently reopen the
same receipt-only authority, with T6 persisting the complete binding and the
three reviewed SHA-256 pins. Release remains disabled until the separate
reviewed external authority is installed.

## Release freeze and external authority

This implementation commit is not publishable. The fixed tracked file
`configs/autodl/tastemolnet_t2_pass_adoption_release_v1.json` has exact native
Boolean `authorization=false` and both external-authority fields are `null`.
CLI flags and environment variables cannot override it. `publish` exits 78
before namespace creation, and `status` cannot endorse a receipt while the
release remains disabled.

`preflight` emits an explicitly `UNREVIEWED_RELEASE_CANDIDATE`. It contains the
full typed source-pin set plus this clean implementation commit/tree and exact
critical blob hashes for the engine, CLI, Slurm parity file, operator document,
and disabled release config. This candidate is inspection material only.

A future release requires a separately reviewed physical JSON authority with
exact schema, `authorization=true`, full absolute-path/string/hash types, the
implementation commit/tree, the exact critical blobs, and every observed
source pin. Its absolute path and SHA-256 must then be placed in the fixed
release config by one clean child commit. The runtime proves that HEAD has
exactly that implementation parent/tree, that the release config is the only
changed path, that all non-config critical bytes still equal the parent, and
that the external receipt is held and unchanged through terminal publication.
This removes the former release-normalized module self-signing pattern.

All Git checks use fixed root-owned `/usr/bin/git`, explicit retained gitdir and
worktree authorities, a minimal environment with replacement objects disabled,
and command-line overrides disabling system/global config, hooks, fsmonitor,
attributes, excludes, and untracked cache. Dirty, staged, untracked, ignored,
`skip-worktree`, and `assume-unchanged` state is rejected. Ignored
`__pycache__`/`.pyc` therefore also blocks release.

## AutoDL CLI

The CLI always requires `--config configs/hpc.yaml` for repository convention;
it reads the config only to validate one physical file. It never loads a
dataset or model.

```bash
python -I -B scripts/autodl/adopt_tastemolnet_gine_pass_v1.py \
  --config configs/hpc.yaml \
  preflight \
  --control-root "$CONTROL" \
  --controller-root "$OLD_CONTROLLER" \
  --scientific-output-root "$OLD_OUTPUT" \
  --training-state-root "$OLD_TRAINING_STATE" \
  --execution-project-root "$EXECUTION_583BF_ROOT" \
  --identity-fix-project-root "$IDENTITY_FIX_3A90_ROOT"
```

Production always opens literal `/proc`; there is no `--proc-root` option. The
private synthetic-proc seam exists only in unit tests. Python isolated mode and
disabled bytecode output are mandatory so inspection cannot silently dirty an
immutable checkout.

Use `status` with the same arguments only after an authorized version has
published the exact receipt. `publish` is documented for interface stability
but is intentionally refused in this revision.

The paired Slurm file is static parity only. It emits an AutoDL-only refusal
and exits 78 before the CLI line, so it cannot submit this route on HPC.

## Non-actions

This implementation does not perform SSH, deploy, commit, controller or GPU
lock acquisition, process signalling, training/resume, calibration, model
loading, matrix writing, old-root writing, or scientific-output writing.
