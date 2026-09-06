# Mut common completed-step repair and protected successor review

## Decision, scope and architecture (2026-09-06)

The existing immutable science checkout `66487c062c86d53ef2f762ce04d0fb965af5af08`
owns the official walker, classifier, trace and generation checkpoints. The
controller's common observer records identical semantic fields in both trace
modes. Their completion order, not the scientific algorithm, caused the loss:
the checkpoint/storage callback ran before the common observer was fsync'd.

This repair modifies that controller-only boundary. No source algorithm,
source cohort, oracle, RNG draw, selected action, candidate, budget, trace
treatment or scientific checkpoint schema is changed. No old output root is
modified; no task, GPU owner or Route B is started. Global roadmap/decision
documents are intentionally not edited in this isolated change, per integration
instructions; this dated document records motivation, impact and limitations.

## Actual failed attempt and numbering

Root (AutoDL runtime-relative):

`outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_same_contract_trace_ab_v1_20260904T052257Z/trace_on`

- Common JSONL is **1-based**: first `step=1,next_step=2`; last
  `step=249,next_step=250`. Its last history digest is
  `43b8970b090e3d418c63ceeeab0a8816e59d339cc3f081931e866b9f0e9a983b`.
- Primary and mirror contain only `step-000000000250`: manifest says
  `completed_step=250,next_step=251`, atomic complete. Its recorded digest is
  `6ba39f8a710f22ff0edaf93734193bad8919df4e440ea090d4dcfdde52d13152`.
- The optional debug recorder uses **0-based** `move_index`; its 249 means the
  250th move, not common step249. Debug events do not contain the common
  observer's random lead-head draw, complete classifier-call digest and
  population before/after decisions. A selected-action trace or final state
  therefore cannot be relabeled as the missing common step250.
- A one-pass frozen-ledger verification at10:24CST checked all249 actual
  step/cursor/scientific-digest/history rows and found no errors. No positive
  shared committed boundary exists. Do not reverse-engineer the
  missing action or skip it. Current justified restart point is the frozen
  deterministic initial seed/data/config state; replay **1..250** in fresh
  roots before continuing 251..500 and the independent501..510 reload.
- The previous 250-step arm took roughly36hours wall-clock; the last25steps
  took roughly9hours. This is a warning about replay cost, not a new ETA
  guarantee. The unstarted trace-off arm still requires its complete prefix.
- Small manifest validation is not payload reload verification. This review
  never loaded production torch checkpoints or queried SQLite/WAL.

## Repaired commit order

At each actual completed outer-loop boundary:

1. Capture the already-observed selected move, restart effects, candidates and
   current RNG digest with the unchanged scientific projection.
2. Append/fsync the complete common JSONL row, then advance the observer's
   history cursor. No fabricated event, duplicate or skipped step is accepted.
3. Call the original algorithm checkpoint/storage callback unchanged.
4. If an atomic algorithm checkpoint now exists, bind its manifest/completion
   marker SHA to the exact observer prefix SHA/bytes/step/history in a fresh
   `common_boundaries/step-<12digits>.json`. Publish via fsync+atomic rename.
5. A storage-stop exception is re-raised after the joint receipt is attempted.
   A failed checkpoint write cannot produce a joint commit or Route-B action.

The joint receipt lives beside the observer, not inside the immutable scientific
checkpoint. It expressly says payload reload has not yet been verified.
The normal pinned runtime still performs its full input, state and RNG reload
checks before a science continuation. Every new run-one performs the unchanged
100000-inode /50GiB /2%-free admission check before scientific modules load.

Segment continuation uses the existing command:

```text
run_mut_trace_mode_equivalence.py run-one ... --phase continuous --resume
```

It requires a genuinely joint committed boundary from this repaired fresh
attempt, an exact ledger end, and matching same-attempt scientific output/config
identity. It seeds history from that boundary and begins exactly `next_step`.
It does not append to the historical failed attempt, relocate its output-bound
checkpoint argv, truncate suffixes, or infer an event. Unexpected uncommitted
suffixes fail closed and require a separate fresh-prefix recovery plan.
The ordinary phase `reload` reopens joint500, retains continuous501..510, and
compares its independent501..510 rows. A partially existing reload cannot be
blindly appended again.

Read-only current-attempt planning (no CUDA/model/state load):

```bash
python scripts/autodl/run_mut_trace_mode_equivalence.py --config configs/hpc.yaml \
  plan-recovery --arm-root /absolute/failed/trace_on --trace-mode on
```

## Existing owner and executor binding

The existing `build_mut_same_contract_ab_task_spec_v1.py` ->
`run_mut_same_contract_ab_owner_v1.py --task-spec <fresh sealed spec>` remains
the bounded sequential A/B owner. Deploy the repaired **controller** in a new
immutable checkout; bind its new commit/runner hash and fresh UUID/control/run/
audit roots. Keep the original science, upstream, inputs and GPU0 lease
contracts. The new input manifest records both observer driver hashes. Do not
edit the dead owner's immutable spec or the healthy next-stage executor.

`run_mut_next_stage_executor_v1.py` currently only consumes ADOPTION or
ROUTE_B after the post-A/B action. `ENGINEERING_REPAIR` explicitly ends BLOCKED;
it is **not** an automatic A/B retry interface. The alive waiting executor must
not be fooled with a synthetic scientific-failure action. A fresh A/B attempt
requires matching post-A/B bindings before its result can enter the existing
adoption/conditional-Route-B pipeline. That runtime binding and resource
admission are not performed by this code commit.

## Inode and byte budget

At10:11CST the old stopped arm's bounded filesystem inventory contained
**37 descendant inodes and127436016bytes**, including primary+mirror each
57645251bytes and observer11559642bytes. Those files already exist and are
not additional recovery demand; all remain preserved.

The patch itself adds, per new arm, one `common_boundaries` directory, two
250/500 receipts and one atomic temporary slot: **four peak inode slots**,
eight for two arm layouts. Receipts are order-of-kilobytes, not pattern files.
For fresh sequential bounded A/B, the observed37-per-arm layout must also
account for retained250+500 checkpoints and mirrors, checkpoint atomics,
reload-specific graph-store files, driver/controller receipts and trace chunks.
A **160 additional inode planning allowance for known fixed layout** is
conservative relative to the observed74 two-arm base, but is not a measured
full peak; variable trace-chunk, temporary-file and checkpoint byte peaks remain
to be measured on the actual500/510 canary. There is no numerical PASS for
unknown peaks. Do not add Route-B's2569-inode future estimate to this as though
Route B has been selected; it remains a mutually conditional branch.

The path currently has about95104--95106 free inodes against an unchanged
100000 guard. It is BELOW_GUARD, not ENOSPC. At least4894--4896 plus new-file
reserve must be provided, and more than this fixed-layout allowance may be
needed. The ~1.68TB free-byte figure does not solve the inode deficit. The
small2.85GiB temporary filesystem is not validated alternative storage. No
deletion, quota change, cache move or threshold reduction is authorized by
this code change.

## T14/T12 protected successor review

At10:24CST T14 reference500 had sealed `checkpoint-000500.json` and ledger500.
The unchanged owner268102 automatically started the low-memory continuous510
child: launcher323640/startticks39530441, science323752/startticks39538869.
The reload child had not started. This is a legitimate handoff, not a loss of
the old reference PID. The owner plan contains actual commands for reference500,
low-memory continuous510, independent250->510 reload, parity, formal promotable
50/100/250/500,2500-step full checkpoints, then generation handoff to the
existing postprocess/publisher. These are real entrypoints, not a PASS claim
for stages that have not run yet.

One delayed engineering defect is fixed only in this isolated source:
`run_t14_route_c_owner.py` used `hashlib.sha256` in its >=10000 convergence
audit without importing hashlib. Its tiny test executes the exact source
expression. The active owner is not patched/restarted; application of this
fix needs the existing safe-boundary engineering successor procedure, never
an early interruption of its healthy reference/compact science.

T12 original PID173495 remains untouched. Its only actual sealed checkpoint
is250; the `segment-00000500.log` filename is a target, not completed500 proof.
Full required reference/accelerated251..500 plus501..510 parity still gates
the fresh-zero production plan. Endpoint-only diagnostics cannot promote it.

## Focused verification

Tests cover real checkpoint serialization/reload on tiny local fixtures,
500-step RNG/actions/candidate/frequency parity,250 guard-stop ordering,
251 continuation, continuous versus reloaded501..510, missing/duplicate/partial
rows, digest/history corruption, and no production payload read in planning.
They are not a production500-step claim. The paired Slurm gate remains a
non-submitting AutoDL-only refusal; no job is launched by these tests.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/autodl/test_mut_common_boundary_v1.py \
  tests/autodl/test_mut_trace_on_adoption_v1.py \
  tests/autodl/test_t14_route_c_convergence_hash_import.py
```
