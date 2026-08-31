# AutoDL TasteMolNet T6 Ours PPO contract

## Status and release boundary

The authoritative switch is
`configs/autodl/tastemolnet_t6_execution_release_v1.json`. A disabled file has
null deployment pins and makes the AutoDL wrapper stop before GPU discovery.
A release is valid only as the immediate child of the pinned implementation
commit and may change exactly that file plus the AutoDL wrapper. The paired
Slurm script remains a static refusal because Taste policy-v2 science is
AutoDL-only.

## Scientific contract

T6 is the bounded smoke for the project’s Ours method. It:

- calls the shared `run_stable_decoded_chem_ppo_loop` rather than a new PPO
  implementation;
- performs 5--10 real optimizer updates;
- selects true-Sweet, frozen-GINE-predicted-Sweet parents from the frozen train
  split only;
- uses the three-class label map Bitter=0, Sweet=1, Tasteless=2;
- uses source label 1 and strict flips `pred_before == 1 and pred_after != 1`;
- records destination labels only in `{0, 2}`;
- forbids RF, validation, calibration-payload, test, CPU fallback, and GNN
  ablation;
- keeps the frozen reference parameters unchanged while requiring the policy
  adapter to change.

The final and last periodic checkpoints must contain the same finite LoRA
tensor identity and the same finite value-head identity as the trained models.
The PEFT configuration must equal the in-memory reviewed configuration and
must name the lexical frozen source model. The tokenizer is not copied into
T6; it remains part of the separately retained T5 source-model authority.

## Input authority

The runner holds and revalidates all of the following through the terminal
boundary:

- exact T3 calibrated-adoption output and gate;
- exact T4 GINE oracle-smoke output and gate;
- exact independently verified managed-v2 T5 clean generic-base adoption and
  its complete source-model inventory;
- one deterministic runtime zero-step LoRA materialization for policy and
  reference, using the reviewed rank/alpha/dropout/target-module contract;
- exact frozen GINE checkpoint bundle;
- exact train CSV path, bytes, hash, and record contract;
- downstream and base policy-v2 documents;
- external release document and controller receipt;
- immutable execution commit/tree and release-bound physical GPU identity.

No validation, calibration split payload, or test payload is opened.

T5 does not claim to contain a trained adapter. T6 loads the adopted generic
base through a retained source descriptor, resets seed 7 before each of the
policy/reference LoRA constructions, and requires their initial tensor hashes
to match. The reference is frozen, its adapter and complete model hash must
remain unchanged, while the policy adapter must change after at least five
real optimizer steps.

## Terminal output

The fresh private output has a fixed root plus one `checkpoint-N` directory and
one log directory. Unknown files, unknown directories, pickle adapter weights,
CSV files, `FAIL`, duplicate terminal names, and tokenizer artifacts are
rejected. Candidate JSONL bytes must exactly equal the post-processing rows
captured in memory by the PPO observer; parsed equality is insufficient. The
shared loop performs I/O through the retained descriptor path, while a logger
filter rewrites that path to the reviewed lexical output root. The terminal
scan rejects `/proc/self/fd/` in every durable file, including logs.

Before the terminal marker is made visible, the runner:

1. closes and flushes its logger;
2. captures every output leaf and directory by descriptor;
3. fsyncs leaves and directories bottom-up;
4. writes and retains `output_hashes.json` and `.PASS.prepared`;
5. repeats GPU/Git/input/policy/checkpoint/candidate/document closure;
6. repeats the retained output-tree closure;
7. links the exact held prepared-marker inode to `PASS` without replacement;
8. removes `.PASS.prepared`.

A crash between steps 7 and 8 leaves two terminal names, which the strict
consumer rejects. No fallible validation or fsync occurs after the successful
commit.

Downstream code must call `hold_taste_ppo_output()` or
`validate_taste_ppo_output()`. Merely checking for `PASS`, required files, or a
log marker is not authoritative.

The managed GPU worker therefore runs a second Python process after the
trainer closes all writers. That process invokes the strict consumer and
atomically publishes a separate fresh verification receipt. The managed task
is successful only when the receipt contains
`[TASTE_T6_OURS_PPO_INDEPENDENT_VERIFIER_PASS]`; it never writes into the
scientific root.

The T0--T16 marker contract for this stage is exactly
`[TASTE_T6_OURS_PPO_SMOKE_PASS]`. The same already-bracketed string is stored
in every structured `marker` field, printed as the log marker, and written to
the `PASS` leaf followed by exactly one newline. Callers must not add or remove
a second pair of brackets.

T6 independently reopens the receipt-only T2 authority inherited through T5.
It accepts only the fresh five-file adoption root plus reviewed gate, receipt,
and embedded-source SHA-256 pins; the consumer checks the canonical hash DAG,
physical binding, fixed source identities, and formal 19-file GINE inventory
without reopening historical controller/training/execution roots. The complete
T2 downstream binding and all three hashes are written into T6 input and gate
evidence and must match T3, T4, and T5 exactly. This is provenance validation,
not an execution release; the tracked release gate remains false.

## Execution-release gates

Before any T6 science, a successor must:

1. pin the exact T2 adoption plus T3/T4 roots, gates, root inventories, common
   GINE checkpoint, feature schema, validation-fitted temperature, and
   downstream policy inside the external frozen-oracle authority;
2. pin the managed T5 gate/published inventory and complete generic-base
   inventory without inventing an adapter or Taste training step;
3. bind one fresh science root, one fresh verification root, execution
   commit/tree, controller receipt, physical GPU UUID, exclusive project lock,
   and storage reservation;
4. run the trainer and independent strict terminal consumer as one managed
   command; only the verifier receipt is the manager's expected terminal root;
5. preserve the two-commit release delta and use a fresh UUID for every
   attempt.
