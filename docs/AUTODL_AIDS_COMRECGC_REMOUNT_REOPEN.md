# AIDS ComRecGC terminal reconciliation after an AutoDL remount

AutoDL maintenance can remount the same persistent volume under a different
Linux device number.  The completed AIDS ComRecGC pair-store adoption records
the original `st_dev`, so a byte-identical source can otherwise fail terminal
publication after a host restart.

The exception is intentionally narrow.  Ordinary pair-store adoption and
reopen remain strict across every recorded stat field.  Only
`validate_reconciled_final_science()` explicitly requests the terminal-
reconciliation policy, which permits a changed `device` field while requiring
the same absolute paths, inode, mode, size, mtime, ctime, pair-store schema and
all three SHA-256 values.  Writable-FD/mapping scans run both before and after
the full hash reopen, the source-owner guard must still pass, and current stats
must remain unchanged throughout validation.

The drift is never silent.  The reopened AIDS terminal evidence contains
`pair_store_reopen_evidence` with the recorded and observed stats for every
source file, whether a device change was detected, the exact allowed field,
the verified hashes, writer-scan counts and same-session stability result.  The
fast16 append stores this object under `appended_cell.terminal_evidence` in
`append_authority.json`.

Use the existing publication entrypoint; no science stage is rerun:

```bash
sbatch scripts/slurm/reconcile_aids_comrecgc_terminal_publication.sh \
  publish \
  --controller-manifest "$CONTROLLER_MANIFEST" \
  --exact-receipt "$EXACT_RECEIPT" \
  --exact-adoption-gate "$EXACT_ADOPTION_GATE" \
  --reconciliation-root "$RECONCILIATION_ROOT" \
  --matrix-output-root "$MATRIX_OUTPUT_ROOT" \
  --authority-state-path "$MATRIX_AUTHORITY_STATE" \
  --authority-lock-path "$MATRIX_AUTHORITY_LOCK"
```

A device-only remount is accepted only by this reconciliation route.  An inode,
mode, size, mtime, ctime, content/hash or live-writer change remains a terminal
failure, and the immutable adoption manifest is never rewritten.
