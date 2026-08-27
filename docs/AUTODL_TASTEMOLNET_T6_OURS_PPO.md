# AutoDL TasteMolNet T6 Ours PPO contract

## Status

The checked-in T6 implementation is deliberately **release disabled**. It is
code and test evidence, not permission to start science. The authoritative
release file is
`configs/autodl/tastemolnet_t6_execution_release_v1.json`; its release bit is
false and all deployment/evidence pins are null. The AutoDL wrapper exits 78
before storage or GPU discovery, and the paired Slurm script exits 64 because
Taste policy-v2 science is AutoDL-only.

Do not edit those bits by hand. A reviewed controller-issued receipt is a
prerequisite for any later activation.

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
- exact T5 clean-policy output, source model, adapter, and loaded policy/
  reference tensor identity;
- exact frozen GINE checkpoint bundle;
- exact train CSV path, bytes, hash, and record contract;
- downstream and base policy-v2 documents;
- external release document and controller receipt;
- immutable execution commit/tree and GPU-1 runtime identity.

No validation, calibration split payload, or test payload is opened.

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

T6 independently reopens the receipt-only T2 authority inherited through T5.
It accepts only the fresh five-file adoption root plus reviewed gate, receipt,
and embedded-source SHA-256 pins; the consumer checks the canonical hash DAG,
physical binding, fixed source identities, and formal 19-file GINE inventory
without reopening historical controller/training/execution roots. The complete
T2 downstream binding and all three hashes are written into T6 input and gate
evidence and must match T3, T4, and T5 exactly. This is provenance validation,
not an execution release; the tracked release gate remains false.

## Remaining release gates

Before any T6 science, a successor must:

1. create a typed, controller-owned external receipt after the final immutable
   integration commit exists;
2. bind controller/task/run ID, live PID/start generation, GPU-1 UUID and held
   exclusive lease, execution commit/tree, release/config/wrapper blobs,
   T3--T5 gates, storage authority, and one fresh output root;
3. make the controller run the strict terminal consumer before adopting PASS;
4. verify on the real `/autodl-fs` output parent, without starting science,
   that held-inode `linkat`, unlink, directory fsync, and strict reopen work;
5. independently review the exact integration tree and production receipt.

Until all five pass, the truthful state is
`TASTE_T6_EXECUTION_NOT_RELEASED`; no launch command is provided here.
