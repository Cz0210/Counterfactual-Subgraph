# AutoDL managed execution protocol v2

Managed execution v2 is the mandatory terminal protocol for future TasteMolNet
T3--T9 attempts. It supersedes every candidate that exposes a mutable file by
hardlink or lets a scientific worker sign its own PASS.

## Attempt and checkpoint identity

`create_managed_attempt(...)` creates
`<stage_root>/attempts/<attempt_id>/` with an RFC-4122 UUIDv4, an O_EXCL
generation token, and an O_EXCL `attempt_manifest.json`. The manifest binds
`attempt_id`, controller/task IDs, commit/config/input hashes, creation time,
hostname, boot ID, and the canonical attempt path. A partially created or
failed attempt burns its UUID and is never removed or reused. Checkpoints use
the same rule at `checkpoints/<checkpoint_uuid>/`.

The held attempt/staging/checkpoint classes keep directory and evidence-file
descriptors open and compare every named path with `fstat` device/inode
evidence. A same-byte path replacement therefore does not recreate authority.

## Process identity and quarantine

`ProcessSnapshotV2` records PID, PPID, Linux start ticks, boot ID, executable
realpath, canonical argv hash, cwd realpath, and cgroup evidence. A
`ManagedProcessLineageV2` binds both launcher and worker generations to the
controller and attempt. A same-generation launcher exec, one direct managed
child, and an unchanged worker generation that is legitimately re-parented
after launcher exit are accepted. Identity drift, unexpected re-parenting,
unexpected children, orphans, heartbeat loss, and terminal mismatch have a
closed `QUARANTINED` representation with `science_adopted=false`,
`downstream_released=false`, and `manual_review_required=true`.

`AUTO_TERMINATE_UNCONTROLLED_CHILDREN` is fixed to `0`. The v2 modules expose
no signal or termination API. Quarantine retains evidence for manual review.

## Worker and verifier boundary

The worker API is:

```text
create_managed_attempt(...) -> HeldManagedAttemptV2
create_worker_staging(attempt, staging_id=None) -> HeldWorkerStagingV2
write_worker_raw_evidence(staging, payload) -> HeldJSONV2
write_worker_exit(staging, payload) -> HeldJSONV2
seal_worker_staging(staging) -> SealedWorkerArtifactV2
```

The worker may write only `raw_evidence.json`, `worker_exit.json`, and
`SEALED.json` as protocol metadata. Scientific files live below its unique
`artifacts/` directory. The worker cannot write `verification.json`,
`gate.json`, `PASS`, `FAILED`, an adoption receipt, or a release marker.

After all artifact writers close, `seal_worker_staging` traverses through
directory descriptors, rejects symlinks/special files/multiply-linked files,
and records every file and artifact-directory `st_dev`, `st_ino`, size,
`mtime_ns`, SHA-256, attempt ID, and generation token. `SEALED.json` binds the
canonical inventory hash and raw/exit evidence hashes.

Only the independent verifier uses:

```text
open_sealed_worker_artifact(
    sealed_path,
    expected_attempt_id=None,
    expected_generation_token=None,
) -> HeldSealedArtifactV2

verify_and_publish_sealed_attempt(
    held,
    final_path=...,
    verification=...,
    force_cross_filesystem=False,
) -> TerminalPublicationV2
```

The opener retains the SEALED generation, inventory directories, and every
file descriptor and rehashes them. After method-specific input, commit/config,
process, and scientific checks return `status=PASS`, the verifier writes
`verification.json`, `gate.json`, and literal `PASS` into a still-private
directory. Controllers accept only the final verifier gate and its exact
verification/generation/PASS cross-binding.

## Atomic publication

On one filesystem the verifier fsyncs the unique staging tree and publishes it
with `renameat2(RENAME_NOREPLACE)` on Linux or `renameatx_np(RENAME_EXCL)` on
macOS. Across filesystems it creates a unique temporary directory below the
destination parent, copies through held source descriptors, fsyncs every file
and directory, recomputes and records the destination inventory, writes the
verifier outputs, and performs the same atomic no-replace rename. It never
uses copytruncate or a file-link primitive.

A modified file, symlink, inode swap, directory ABA, terminal collision, or
publication mismatch fails closed and releases no dependency. Production must
preflight the exact target filesystem's atomic directory-rename behavior
before any scientific attempt is released.

## Schemas and marker

- `managed_attempt_manifest_v2`
- `managed_generation_token_v2`
- `managed_worker_raw_evidence_v2`
- `managed_worker_exit_v2`
- `managed_worker_sealed_v2`
- `managed_file_evidence_v2`
- `managed_directory_evidence_v2`
- `managed_verification_v2`
- `managed_gate_v2`
- `managed_terminal_publication_v2`
- final marker bytes: `[MANAGED_EXECUTION_V2_PASS]\n`

This local freeze changes no Taste release bit, does not adopt a scientific
artifact, and does not start a controller, GPU process, or experiment.
