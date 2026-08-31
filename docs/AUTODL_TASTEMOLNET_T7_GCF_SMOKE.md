# AutoDL TasteMolNet T7 GCFExplainer smoke

## Status

This revision is an implementation stage-freeze only. It is not an execution
release and cannot start science:

- `configs/autodl/tastemolnet_t7_gcf_smoke_release_v1.json` has the native JSON
  Boolean `release_enabled=false` and every deployment/predecessor pin is
  `null`;
- `scripts/autodl/run_tastemolnet_gcf_smoke.sh` exits with code 78 before GPU
  discovery, output creation, model loading, or controller launch;
- the paired Slurm script is an unconditional AutoDL-only refusal;
- environment variables and CLI paths are not release capabilities.

A later reviewed one-parent release commit may change only the release JSON and
the wrapper literal. It must additionally pin managed execution v2 PASS, the
managed stage root, the one Taste NeuroSED generic managed final (root, PASS,
gate, verification, checkpoint, feature schema, and checksum manifest), and the
exact absent final path. The worker cannot choose or publish that final path.

The official-semantics fixed-budget NeuroSED PASS is adopted without changing
its scientific bytes by
`scripts/autodl/adopt_tastemolnet_fixed_budget_neurosed_v2.py`. Copy mode runs
as a child of the existing generic managed-v2 worker and copies the complete
source tree into `MANAGED_ARTIFACT_ROOT`; a different process uses
`verify-and-publish` to reopen and rehash both trees and invoke the existing
atomic publisher. T7 accepts its fixed-budget consumer-v2 with truthfully named
sampler/label hashes. This adapter does not enable the disabled T7 release.

## Why the BACE adapter is not reused

The BACE GCF route is not a generic frozen-GINE adapter. It requires a BACE
model card, two classes, a BACE graph schema, and a binary projection. The
vendored official predicate then treats an importance score of at least 0.5 as
a counterfactual. That predicate is invalid for TasteMolNet: for example,
probabilities `[0.3, 0.4, 0.3]` produce `1-p(Sweet)=0.6`, but Sweet is still the
argmax and the graph is not a counterfactual.

The T7 implementation therefore reuses:

- official full-graph node/edge mutation enumeration;
- official VRRW transition, teleport, frequency, and bounded-walk machinery;
- official NeuroSED model loading and normalized threshold-coverage semantics;
- the existing lineage-aware molecule codec used to reconstruct a complete
  edited graph before GINE scoring.

It never loads a BACE dataset, BACE GINE, BACE-adapted NeuroSED, RF oracle, or
BACE output. If the official full-graph loop cannot run without such an
artifact, T7 fails closed.

`TasteFrozenGINENativeAdapter`, `TasteGCFGraphSchema`, `load_train_rows`,
`encode_taste_source_graph`, and `taste_record_to_pyg` are stable lazy
compatibility entries for successor Taste method smokes. They preserve the
same original three-class order, complete-graph decoder, origin lineage, and
embedding identity without copying classifier or codec logic. The shared
lineage codec's binary Mutagenicity labels are removed at this boundary:
`label`, `gnn_label`, PyG `y`, and `source_label` are the exact native Taste
label in `0/1/2`; no scalar `target_label` exists; `destination_labels` is the
ordered set of the other two native classes (therefore `[0,2]` for Sweet).

## Frozen smoke semantics

T7 uses the exact calibrated three-class GINE adopted by both T3 and T4:

```text
dataset=tastemolnet
backbone=gine
num_classes=3
source_label=1
label_map={0: Bitter, 1: Sweet, 2: Tasteless}
```

Every parseable connected native candidate is scored as a complete graph. The
two independent quantities are:

```text
importance = 1.0 - probabilities[1]
candidate  = argmax(probabilities) != 1
strict_flip = pred_before == 1 and pred_candidate != 1
```

The bounded smoke uses eight GINE-correct Sweet parents selected
deterministically from a 64-row train-only Sweet pool, 16 official VRRW steps,
sample size 128, capacity 512, teleport 0.1, seed 7, and `alpha=1`. The
individual-coverage term comes from the held Taste-specific NeuroSED model via
the official `neurosed_threshold_coverage_estimation` function. Candidate
ranking therefore preserves official normalized-distance coverage while the
classifier term remains exactly `1-p(Sweet)`.

The 16 steps are one checkpoint/resume smoke, not one uninterrupted call. The
first real official loop executes steps 1--8 and records the exact next graph
cursor. T7 creates a dedicated mode-0700 `runtime` child under a fresh private
temporary envelope, then creates a fresh
`checkpoints/<checkpoint_uuid>/` hierarchy exclusively with descriptor-
relative operations. The UUID is canonical v4 and never reused; a 256-bit
generation token is stored in and hash-bound to the checkpoint evidence. It
writes and fsyncs one mode-0600 checkpoint, fsyncs every held directory, and binds the
checkpoint SHA-256 plus device/inode/mode/link/owner/size/time identity. The
checkpoint contains the complete official mutable VRRW state, current cursor,
bridge records and counters, adapter/batch-scorer counters, action counts, and
Python/NumPy/Torch/available-CUDA RNG states. It is private runtime state, not
terminal output.

After the durable write, T7 deliberately raises and catches one private
planned-interruption exception. It drops the in-memory checkpoint payload,
resets official VRRW state, bridge state, adapter/scorer progress, action
counts, and every RNG, and proves the reset digests differ. It then reloads
only through the still-held exact checkpoint inode, rechecks its pathname,
physical identity, bytes, and SHA. Every write/load, the end of continuation,
and the start of terminal-evidence construction also reclose the complete
runtime ancestor chain, the dedicated temporary parent and `runtime` name, the
`checkpoints` container, and the UUID checkpoint-directory name/inode.
Whole-runtime, container, or checkpoint-directory rename/equal-copy/restore
attacks therefore fail even if
the bytes are restored. T7 restores the complete state and requires
the live progress/RNG digests to equal the saved digests. Steps 9--16 run
through the real official loop. Only its initial resume entry returns the
restored current graph without a fresh restart; later native moves, teleports,
restarts, sampling, frequency reinforcement, and candidate ordering remain
official. This prevents an 8-step deterministic restart from seed from being
misreported as resume.

Aggregate raw evidence requires native JSON Booleans for
`checkpoint_written`, `planned_interruption_observed`,
`checkpoint_reloaded`, and `resumed`; the exact split `8 + 8 = 16`; matching
saved/restored progress and RNG digests; a different reset digest; identical
checkpoint/final prefix commitments; equality between the saved resume cursor
and the first post-resume trace identity; the physical checkpoint binding;
and a recomputable aggregate continuity commitment. The independent verifier
rejects Boolean/integer aliases, numeric strings, fake counts, drifted restore
digests, or a broken resume boundary.

The following remain `NOT_EVALUATED`:

- global-summary selection and ordering;
- calibration threshold selection;
- full GCFExplainer readiness or paper-result eligibility.

The smoke passes only if the real official loop runs across that physical
checkpoint/reload boundary, invokes a native edit, emits the exact 16-step
result, and contains at least one valid complete graph whose calibrated GINE
argmax is Bitter or Tasteless. A scoring-only diagnostic, deletion-only
conversion, synthetic graph list, deterministic restart, or score-threshold
candidate shortcut cannot pass.

## Input authority

Runtime opens inputs in this order and retains them through worker close, just
before managed-v2 sealing:

1. a clean one-parent release checkout and exact critical blobs;
2. the SHA-pinned external release authority;
3. typed live controller and exclusive GPU-0 lease receipts;
4. the exact five-file T2 adoption, opened once with
   `hold_t2_gine_pass_adoption` using the release-pinned root, gate, receipt,
   and embedded-source SHA-256 values; no historical controller,
   training-state, scientific-output, execution, or identity-fix root is
   reopened;
5. retained T3 and T4 PASS outputs, both required to carry the canonical SHA
   of that complete T2 binding;
6. the one checkpoint identity common to T3 and T4, held with
   `hold_taste_checkpoint_bundle`;
7. managed execution v2 PASS and one fresh managed stage root;
8. one independently verified Taste NeuroSED managed-v2 final, held as a
   single root with its generic `PASS`, `gate.json`, `verification.json`, exact
   `best.pt`, feature schema, and checksum manifest;
9. only the train CSV path/hash/counts named by the frozen GINE checkpoint's
   split manifest.

The T3 and T4 evidence must match each other on every checkpoint identity and
must match the held T2 formal-bundle root, model digest, and hash-inventory
digest. The final pre-seal callback repeats the clean immutable-checkout test,
rejects Git `skip-worktree`/`assume-unchanged` concealment, rehashes every
critical blob, and repeats the controller PID-generation/full-cmdline and GPU
identity checks. The wrapper's literal release refusal runs before even
`common.sh` is sourced.

Validation, calibration, and test CSV payloads are not opened by T7. The
NeuroSED managed final independently proves train-only fitting,
validation-only selection, and `calibration_loaded=false` /
`test_loaded=false`; see `docs/TASTE_GCF_NEUROSED_PROTOCOL.md`. T3/T4 are
predecessor authorities; their payloads are not repurposed as T7 generation
data. The calibrated temperature JSON is read from the held checkpoint solely
as part of frozen-GINE inference.

## Managed output and no-redistribution boundary

The worker never creates a terminal root. It creates one managed-v2 UUID
attempt and UUID staging directory. Its only control evidence is:

```text
raw_evidence.json
worker_exit.json
SEALED.json
```

`raw_evidence.json` contains aggregate-only input commitments, scientific
summary, and opaque candidate trace; it contains no gate, terminal status, or
adoption authority. After all scientific inputs are revalidated and closed,
the worker writes `worker_exit.json`, hashes the closed staging inventory, and
writes `SEALED.json`. It then stops with
`SEALED_PENDING_INDEPENDENT_VERIFICATION`.

A separate verifier process opens SEALED with the frozen managed-v2 API,
descriptor-holds every file/directory, rejects symlink/inode/ABA or
modified-after-seal drift, and runs `verify_t7_worker_raw_evidence`. That
method verifier cross-binds attempt/generation, expected final path, managed-v2
PASS, the Taste NeuroSED generic final/checkpoint, frozen GINE/T2/T3/T4 inputs, checkpoint
resume proof, official full-graph actions, and exact three-class
score/candidate semantics. Only after that verification returns PASS may the
managed-v2 verifier write `verification.json`, `gate.json`, and `PASS`, then
atomically publish with no-replace rename (or cross-filesystem copy/fsync/
rehash followed by no-replace atomic rename). The worker imports neither the
sealed opener nor the verifier/publisher function, and no hardlink terminal
primitive remains in T7.

The official `counterfactuals.pt` and private VRRW checkpoint remain only under
the process-owned temporary runtime. The held checkpoint descriptor is kept
through reload, continuation, and evidence construction. T7 deliberately does
not perform a check-then-unlink pathname cleanup: a same-user replacement
between those operations could make it delete a foreign inode. Descriptor
close is non-destructive, and the enclosing private temporary-directory
lifecycle owns ordinary cleanup after science returns. The aggregate evidence
therefore states honestly that the checkpoint inode was held through resume
evidence, pathname cleanup was delegated to the temporary runtime, no unlink
was attempted by the T7 security boundary, and no checkpoint payload was
persisted to worker evidence. The eventual verified root contains no SMILES, molecule
ID, native graph tensor, source CSV row, checkpoint payload, or reconstructable
dataset artifact. Opaque SHA-256 graph/embedding identities and aggregate counts are
retained for smoke integrity only. `data_redistribution_allowed` remains
false.

## CLI parity

The reviewed AutoDL wrapper will eventually invoke:

```bash
python -B scripts/run_tastemolnet_gcf_smoke.py \
  --config configs/hpc.yaml \
  --stage T7_GCF_SMOKE \
  --output-dir /absolute/fresh/private/t7-gcf-smoke \
  --set inference.fallback_to_heuristic=false
```

The worker CLI prints only its SEALED receipt and never prints a success marker.
`--validate-only` delegates to the generic managed-v2 verified-gate consumer
and is read-only. The independent verifier owns terminal reporting. The Slurm
file documents the same CLI but exits before it; TasteMolNet policy-v2 science
is AutoDL-only.

The structured T7 method-verification marker is exactly
`[TASTE_T7_GCF_SMOKE_PASS]` (including brackets). It is domain evidence inside
the generic managed-v2 verification; it is not a second terminal PASS file.

## Work intentionally not performed here

This code-only task does not push, deploy, SSH, allocate a GPU, alter a
controller, touch a GPU lock, write a scientific output root, or run a model.
Fresh release evidence and an independent review are required before any of
those actions.
