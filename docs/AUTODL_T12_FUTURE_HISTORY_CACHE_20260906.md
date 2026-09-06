# T12 future-only immutable local history cache

## Scope and deployment boundary

This change is storage/read routing only. It does not stop, signal, inject into,
or replace the files used by reference reader173495/owner162844. The current
reference continues at its original immutable execution commit. No current
production history has been copied by this development task.

The API may be used only by a **new** reader of a committed, closed source.
The producer must be absent, the existing no-live-writer check must pass, and
each source segment must have exactly its checkpoint-committed byte count.
An appended/uncommitted tail is not silently accepted. The snapshot identity
comes from its existing committed checkpoint/receipt; no source snapshot is
rewritten, and no diagnostic checkpoint is promoted to production.

`src/utils/t12_future_history_cache.py` copies original journal and first-seen
embedding files once to a fresh local directory. It computes the source hash
during that one copy and verifies the local copy once against the existing
segment digest. Files become read-only and are described by one small
no-replace cache manifest. Reopening an unchanged cache checks file identity;
it does not rescan large source hashes. Failed partial caches remain unsealed
diagnostic evidence and cannot be loaded.

The original `T12CompactHistoryJournal` and `T12FirstSeenEmbeddingStore`
decoders remain authoritative. Their optional `history_read_cache` argument
only changes reads to a bounded1MiB buffered local stream. Original sequence,
record, hash-chain, first-seen raw bytes, index counts, graph IDs and checkpoint
state are still checked by the same code. The original default path is
unchanged. Cache readers are explicitly read-only; they cannot start a writer.

## Existing CLI and resource contract

The thin entrypoint is `scripts/autodl/cache_t12_future_history.py` and supports
`--action prepare` and `--action verify-read`. Run its `--help` for complete
arguments. Prepare requires an exact snapshot SHA, dead producer PID/start
ticks, a fresh cache root and explicit existing local byte/inode reserves.
Verify-read requires the resulting small manifest SHA and a fresh disposable
index directory. Neither action obtains a GPU lease or starts scientific
generation. Both require `--config` and
`--set inference.fallback_to_heuristic=false`.

No live task spec is rebound in this change. A future owner must bind the
cache's snapshot and manifest before passing `history_read_cache=cache` to
the existing read-only journal constructor. A fresh-zero generation starts
from zero; this cache is not permission to reuse diagnostic walk state.

The actual future local destination must pass capacity admission before
copying. Preserve the caller's existing safety reserves. The earlier
approximately2.85GiB NVMe-free snapshot is not a new admission receipt; all
history and embedding files plus disposable index/checkpoint reservations
must fit at launch. The paired Slurm script intentionally refuses: Taste
payloads and this cache stay AutoDL-only, and allocating an HPC GPU for a
storage helper would be outside this task.

## Timing and bounded live read evidence

Only three FD/proc observations were taken on2026-09-06; all use reader173495,
start ticks20206493, same read-only FD628 and file size242191887bytes:

| CST | FD position | Bytes remaining |
|---|---:|---:|
|23:42:30.503|227679231|14512656|
|23:43:28.867|227725135|14466752|
|23:44:03.900|227760399|14431488|

The intervals were58.36s and35.03s. No backwards position/repeated read was
observed in this bounded window. The net FD advance was81168bytes/93.40s
(about869B/s); `/proc/io` read_bytes increased1286144bytes and includes other
I/O, so it must not be represented as this file's exact transfer progress.
All observations were D-state in `generic_file_read_iter`; no new scientific
step or full500/parity conclusion follows from these I/O observations.
The current history is not independently accepted as a closed no-writer
source for this cache, so it was not copied.

The cache records first staging-copy/verification timing separately from
subsequent local decoder/index timing. Kernel page-cache state is explicitly
uncontrolled: this tool never drops system caches or claims a cold-storage
benchmark merely because it is the first timed call. Tiny fixture speed is
not used to project full scientific completion time.

## Tests and evidence

Focused tests compare original and cached codecs on repeated graph records
and signed-zero first-seen embeddings; full checkpoint snapshots, records,
order and raw embedding bytes remain exact. Tests also reject live producers,
writable source FDs, uncommitted tails, bad resource admission, source/cache
mutation, writer use, changed snapshots and existing cache targets.

GNN corrected seed7 and L0 remain separate accepted ablation results. Their
local paper tables are present. This helper neither recomputes them nor
changes the main12/16 authority.
