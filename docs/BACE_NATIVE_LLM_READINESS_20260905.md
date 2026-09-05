# BACE native LLM successor readiness — 2026-09-05

Main-table owners and GPU1's Taste T13 reservation remain untouched. This is
preparation plus an isolated CPU model probe, not a completed LLM ablation.

## Truthful treatments and current implementation

The four rows are BRICS_FIXED, CHEMLLM_7B_OFF_THE_SHELF,
CHEMLLM_7B_PPO_LORA_MAIN, and CHEMLLM_2B_OFF_THE_SHELF. There is no independent
matched project SFT checkpoint. The BACE main model is base + fresh LoRA + 300
PPO optimizer updates. Existing PPO checkpoint weights are reused, never retrained.

The old main candidate pool used a plain prompt. The new comparison uses each
pinned model's native `build_inputs` with identical counterfactual task text.
Consequently L2 is **MATCHED_REGEN_REQUIRED**, not adopted candidate science.
The 386 train parents, four deterministic shards, two regimes (seed7/temp0.3 and
seed13/temp0.7), four sequences per parent-call, top-p0.9 and maximum96 new
tokens are fixed. Each shard/regime seeds once after model loading; there is no
per-parent reseeding and no claim of parity with the historical main pool.

`run_bace_native_llm.py prepare` writes four byte-bound task specs without a
model load. `generate` uses the actual existing NF4/BF16 main loader and optional
exact PPO adapter, native prompt rendering, and RNG-bound committed parent-call
checkpoints. A tiny real Torch/Python/NumPy regression verifies byte-identical
continuous versus resumed candidates. This is not yet a real 7B GPU resume test.

BRICS already has 472 vocabulary fragments, 386 parents, 3088 attempts and zero
shortfall. Its old proposal manifest omits a direct reference field but SHA-binds
the vocabulary and shortfall manifests, both of which bind the same frozen BACE
reference. The adopted transitive binding is verified; original files are not
rewritten. Common downstream verification must still execute on this pool.

## Actual isolated 2B evidence

Model: AI4Chem/CHEMLLM-2b-1_5 at
`215c0dbc89417a06bbc3bae43a3ad61e58f0a56e`.

Receipt:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm/isolated-2b-cpu-25d0c562-20260905/isolated_load_receipt.json`

Receipt byte SHA256:
`dc1b1b93f3304b63aae6eb6c68b6dbf655ad4398f105fc357592c0a0bce442af`.

The actual CPU load, isolated remote-code import, finite forward (1×25×92544),
and native greedy generation (4 of maximum4 tokens) passed on the first attempt.
CUDA device count was zero; no main GPU lock or main output root was touched.
Actual loaded counts: total1,889,110,016; embeddings379,060,224;
non-embeddings1,510,049,792; frozen inference trainable0 and LoRA0;
BF16 tensor bytes3,778,220,032. Model names were not used to derive these counts.
The tokenizer's unused vocabulary-export method is exempted only for its exact
reviewed source SHA and disabled at runtime; unreviewed writes still fail closed.

## Remaining conditions before formal LLM science

1. Independent GNN seed7 scientific verification and verified package must PASS;
   a producer marker alone is insufficient. Matrix13/16 is not a prerequisite.
2. An existing resource owner must pass its genuinely held lock FD and fresh
   resource evidence to the native entrypoint. The entrypoint creates no new
   lease or borrowing platform. If that owner is absent, state is WAITING_RESOURCE.
3. GPU1/T13 reservation and all ready main evaluator/worker requests take priority.
   Only a normal idle exclusive lease is supported. No <=120s safe-release bound
   has been measured, so revocable borrowing is not permitted by this runner.
4. Formal 7B/2B GPU quantized inference and their real model-specific resume
   health remain to be observed. CPU2B evidence does not claim GPU inference PASS.
5. The single frozen-GINE common verification/selector/test adapter now passes
   focused tests, but still must run on real candidates after the core gate.
   Readiness is `EXECUTABLE_ENTRYPOINT_READY_WAITING_GNN_CORE`, not result PASS.
   No final ablation metrics exist yet.

The CPU probe entrypoint is
`scripts/ablations/llm/audit_chemllm_2b_isolated_load.py --mode cpu-load --tiny-forward`.
The native task entrypoint is `scripts/ablations/llm/run_bace_native_llm.py`;
run `--help` and the `prepare`/`generate` subcommand help for exact required pins.
At preparation, pass the actual 2B receipt path and byte SHA through
`--two-b-isolated-receipt` and `--two-b-isolated-receipt-sha256`; the proof then
enters the immutable task hash. Generation never injects proof into a sealed
task, so the downstream verifier receives the exact same task digest.
Paired Slurm wrappers remain synchronized but have not been submitted for LLM science.
