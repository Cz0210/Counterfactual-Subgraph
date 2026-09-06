# Next-executable-stage resource overlay

This explicit 2026-09-07 authorization replaces the old fixed 100000-inode
engineering gate only for newly bound Mut/LLM stages. Platform statistics and
active T13/T14/T12 configurations remain unchanged. Admission is
`20000 + 2 * peak_new_files_until_next_safe_boundary`; unknown peaks block only
their corresponding stage, never count as zero. Serial exclusive work counts
the maximum only with explicit exclusion evidence. Existing files are excluded.

The original LLM config remains sealed. A resource-only dispatch adds a separate
policy reference; queue, owner evidence, child preflight and the existing owner
watchdog consume that policy. GPU generation code/spec/model/prompt/checkpoints
remain at their original scientific commit. The owner driver is a separate pin.
Current main-task phase guards are re-read; a future phase cannot silently reuse
the present next-checkpoint estimates. No reservation is cleared by this policy.

Future LLM CPU evaluation optionally stores lossless float32 MolCLR node arrays
and atom order in a new table in its own existing WNode database. This avoids a
file per graph. Original encoder, feature keys, weights, directional action keys,
exact OT, selectors and test boundary remain unchanged. Existing caches/results
are not converted. Original NPZ and compact cache are compared through the real
distance caller on a tiny fixture, including same-run reopen/cache reuse.

CPU evaluation checks real file slots at each parent boundary without rescanning
hashes; the existing owner additionally reads real memory/disk/statvfs at most
60 seconds apart. A resource pause retains the same queue owner and resumes the
same parent-checkpoint root after admission, preserving its control receipt.
Engineering failure still terminates instead of being retried indefinitely.
An owner crash requiring a different queue root needs an explicit recovery
binding; it cannot silently consume another owner's pause request.

Cold-directory inventory is separate from admission. No cleanup is required
when the new real gate passes. Without complete dependency evidence the outcome
is NO_OP, not a broad deletion or a platform-support prerequisite.

Focused tests exercise policy arithmetic, live phase scope, unknown stages,
unchanged legacy behavior, FD/UUID/exclusive locks, narrow dispatch binding,
lossless cache and the real WNode caller. They do not load a real LLM or run any
completed science. Deployment and actual GPU smoke remain distinct states.
