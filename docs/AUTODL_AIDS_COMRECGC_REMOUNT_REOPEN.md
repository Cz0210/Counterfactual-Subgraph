# AIDS ComRecGC terminal reconciliation after an AutoDL remount

AutoDL maintenance can remount the same persistent volume under a different
Linux device number.  The completed AIDS ComRecGC pair-store adoption records
the original `st_dev`, so a byte-identical source can otherwise fail terminal
publication after a host restart.

The exception is intentionally narrow.  Ordinary pair-store, theta-close,
DBSCAN and component-summary reopen remain strict across every recorded stat
field.  Only
`validate_reconciled_final_science()` explicitly requests the terminal-
reconciliation policy, which permits a changed `device` field while requiring
the same absolute paths, inode, mode, size, mtime, ctime, pair-store schema and
all three SHA-256 values.  Writable-FD/mapping scans run both before and after
the full hash reopen, the source-owner guard must still pass, and current stats
must remain unchanged throughout validation.

The same maintenance event also changed `st_dev` on all three sources recorded
by the completed theta-close view: the 23.5-GB physical vector store, normalized
distance array, and pair-semantics contract.  Their inode, mode, size, mtime,
ctime and SHA-256 values remained exact.  The terminal-only reopen therefore
brackets all three with exact-inode writable-FD/mapping scans and a full replay.
The adopted DBSCAN terminal reuses that already-audited vector source, rehashes
it, and retains every non-device stat field.  The completed component summary
keeps its historical owner and writer-lock receipts byte-exact, permits only
their recorded device number to differ, and holds the existing writer lock
exclusively from admission through the complete terminal replay and final
same-session stat check.  No receipt is rewritten.

The drift is never silent.  The reopened AIDS terminal evidence contains
separate pair-store, close-view, DBSCAN-source and component-summary reopen
evidence with recorded and observed stats, whether a device change was
detected, the exact allowed field, verified hashes, writer/lock results and
same-session stability.  The fast16 append stores these objects under
`appended_cell.terminal_evidence` in `append_authority.json`.

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
