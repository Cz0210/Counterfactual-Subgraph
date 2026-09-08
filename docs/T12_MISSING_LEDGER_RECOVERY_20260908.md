# T12 bounded missing-ledger recovery (2026-09-08)

## Current result

This is an implementation increment, **not** diagnostic parity PASS and not a
submitted science job. The existing reference owner and its 500→510 reader must
finish naturally. Nothing in this change signals a process or edits an active
worktree. The existing natural checkpoint is not promoted to production.

The original architecture is unchanged: official GCFExplainer VRRW, the bound
frozen Taste GINE adapter and NeuroSED coverage, compact history/transition
checkpointing, the existing 15-stage fresh-zero production plan, and the original
canonical publisher. The new utility adds observational records and a finite
consumer of that existing plan, not another scheduler, GPU lock, or authority.

## Bounded recovery plan

`run_t12_shadow_recovery.py plan` creates an immutable 520-transition plan from
explicit input bindings. It adds ten same-process continuation transitions per
implementation only where a complete continuous ledger does not already exist.
The maximum is 540, with all transitions in 251–510; no 0–250 replay is planned.
The `status` action reads checkpoint metadata and process identity only and reports
the known implementation gaps honestly.

The observer reads actual selected-action frames, sampling weights, graph values,
candidate frequency/order, and RNG state. It captures raw classifier outputs from
the real forward hook and raw NeuroSED values only when actually computed. It
does not infer raw logits from probabilities or distances from coverage masks.
A missing cached value prevents an evidence-complete comparison. The independent
small observer fixture proves API-level RNG/candidate preservation only; it is
not the real production-path regression.

Ledger records are gzip-append segments. Before the original checkpoint writer,
the observer flushes and fsyncs the ledger. Only after the checkpoint exists can a
small joint manifest expose a complete boundary. Partial records never imply an
adoptable checkpoint. No scientific values are replaced by pickle object identity.

## Actual finite activation interface

`activate-inherited` executes the existing 15-stage plan synchronously, once,
after complete parity evidence and an existing canonical owner handoff. It checks
the inherited lock FD and actual parent PID/start ticks, registry binding, lock
competition from a separate process, full GPU UUID, and an existing resource
provider command per stage with ≤120-second measurements. It never creates a new
owner or acquires another GPU lock. Output roots must be fresh. It does not write
the matrix: final verification is followed by the existing publisher locator.

This interface is deliberately not dispatchable from a standalone shell. The
current owner still needs an explicit future stage-boundary binding that passes
its actually held FD and binds the resource commands; setting an integer in JSON
does not satisfy it. These bindings must not be guessed from old GPU1/NVMe plans.

## First unclosed science interfaces

1. `selected_raw_cache_evidence_binding`: old step-250 transition/canonical caches
   retain probabilities and binary coverage, not every raw query. The observer
   refuses to invent those values; a graph/query-key-bound historical raw source
   or a scientifically approved observational strategy remains necessary.
2. `live_state_continuous_501_510_tail`: current generation segment entrypoints
   exit and reload at each checkpoint. Invoking a second segment in the same shell
   is not the required same-process live-state control. No false continuous proof
   is emitted by this change.
3. The real train-only observational exact regression and reference/accelerated
   source-equivalence binding remain prerequisites before using the transition
   budget. No GPU diagnostic is launched while these gaps remain.
4. The original owner must bind the finite activation at its natural stage
   boundary, using the original registry CAS and publisher, once parity exists.

## Cache policy

Existing future-only history cache code is reused, not rewritten. Admission
counts tmpfs in cgroup memory, limits it to 1 GiB per task and 2 GiB aggregate,
and leaves persistent originals/checkpoints authoritative. This increment does
not move active readers or stage any cache onto the full NVMe disk.

## CLI

All commands require `--config configs/hpc.yaml` and
`--set inference.fallback_to_heuristic=false`; `--help` lists real options.
`scripts/slurm/run_t12_shadow_recovery.sh` is the synchronized repository Slurm
wrapper, not a new HPC science submission. Actual T12 science remains on the
original AutoDL owner/lease path.
