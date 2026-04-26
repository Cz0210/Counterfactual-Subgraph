# Decisions Log

This file records major design decisions for the counterfactual subgraph v3 project.

It should be updated whenever a meaningful implementation, algorithmic, or interface decision is made.

---

## Template

```md
## [YYYY-MM-DD] Decision title

### Background
Why was this decision needed?

### Decision
What was decided?

### Alternatives considered
What other options were considered?

### Consequences
What changes because of this decision?

### Status
Proposed / Accepted / Deprecated / Superseded
```

---

## [2026-04-26] Tighten decoded PPO parse/connect diagnostics and tiny-fragment guard

### Background
The `decoded_chem_diag50_parsefix_connectfix_v2` diagnose run showed that
minimal syntax repair was attempted but never surfaced successful repaired
candidates, disconnected fragments were not reliably entering component
salvage, and the policy was beginning to exploit very small fragments such as
`O`, `S`, and `N=O`.

### Decision
Keep projection-v1 and the existing decoded PPO Slurm parameter chain intact,
and make a narrow reward-path patch:

- expand minimal syntax repair diagnostics with candidate counts, acceptance,
  and fine-grained failure reasons;
- detect raw and core disconnected components before strict substructure checks
  and run component salvage on the disconnected representation;
- add a diagnose-configurable tiny-fragment hard fail after strict/projected
  fragment resolution, so projection success cannot bypass the minimum atom
  constraint;
- add a v3 50-step Slurm wrapper that enables the new guard with
  `MIN_FRAGMENT_ATOMS=3` and `TINY_FRAGMENT_HARD_FAIL_PENALTY=-6.0`.

### Alternatives considered
1. Rewrite parse-failed outputs into nearest valid molecules.
2. Move tiny-fragment suppression into the prompt only.
3. Rebuild projection-v1 instead of preserving the existing retrieval path.

### Consequences
- Repaired parseable candidates now continue into the same strict parent
  subgraph / retrieval projection path rather than being treated as a separate
  unconstrained repair objective.
- Diagnose logs distinguish raw/core component counts and salvage failure
  stages.
- Very small strict or projected fragments receive a fixed hard-fail reward in
  the new diagnose script, preventing positive reward terms from masking tiny
  fragment collapse.

### Status
Accepted

---

## [2026-04-25] Add parent-constrained retrieval projection for decoded PPO non-substructure fragments

### Background
Decoded-chem PPO now surfaces parse failures separately from parseable fragments
that are not valid parent substructures. The latter failure bucket is actionable:
the raw fragment already has a chemically parseable shape, but the exact graph
does not occur in the parent molecule. For the current instance-level candidate
generator, these cases should be repaired by projecting onto strict
parent-derived candidates before reward and oracle scoring.

### Decision
Add an optional parent-constrained candidate retrieval projection path for
decoded PPO:

- keep parse-failed raw fragments on the existing failure path;
- mark parseable strict parent substructures as `projection_method=identity`;
- when a parseable connected core is not a parent substructure, build a
  parent-derived candidate pool from ring systems, SMARTS functional-group
  neighborhoods, atom-centered k-hop neighborhoods, bond-centered k-hop
  neighborhoods, and stable parent-index BRICS components;
- filter candidates to connected, RDKit-parseable, non-full-parent subgraphs
  whose deletion leaves a non-empty residual;
- score candidates with Morgan Tanimoto, MCS atom coverage, functional-group
  overlap, atom-count difference, and a large-fragment penalty;
- if the best score passes the configured threshold, continue reward and
  deletion-oracle scoring on the projected fragment and subtract a projection
  penalty from `reward_total`;
- expose projection controls through `scripts/train_ppo.py`, decoded PPO logs,
  and Slurm wrappers.

### Alternatives considered
1. Reuse the existing parent-aware repair path without adding k-hop candidates,
   strict deletion filtering, or projection-specific logs.
2. Penalize all parseable non-substructure outputs without attempting a
   parent-constrained projection.
3. Rewrite raw fragments directly with string heuristics instead of retrieving
   from parent atom-index subgraphs.

### Consequences
- The dominant `parse_ok_but_not_substructure` failure bucket can now produce
  rewardable parent-derived fragments without changing parse-failed behavior.
- Logs now record projection attempt/success, retrieval score/source, projected
  fragment, atom statistics, candidate count, and applied penalty.
- HPC diagnosis can run fixed 50-step and 200-step projection jobs through
  dedicated Slurm scripts without hand-written `sbatch --export` arguments.

### Status
Accepted

---

## [2026-04-25] Wire parent-aware repair controls through decoded PPO CLI and Slurm wrappers

### Background
The decoded PPO rewarder already supported one optional parent-aware repair
attempt for broken decoded fragments, but the training entrypoint and the HPC
Slurm wrappers did not expose those controls. As a result, users could set
environment variables such as `ENABLE_PARENT_AWARE_REPAIR=true` in `sbatch`
commands without the settings ever reaching `ChemRLRewarder`.

### Decision
Expose the existing repair controls end-to-end without changing the reward
objective:

- add `--enable-parent-aware-repair`,
  `--repair-min-similarity`, and `--repair-max-candidates` to
  `scripts/train_ppo.py`;
- pass the parsed values into `ChemRLRewarder`;
- log the resolved repair configuration at startup;
- forward the corresponding environment variables through
  `scripts/slurm/train_decoded_chem_ppo_full.sh` and
  `scripts/slurm/train_ppo.sh`.

### Alternatives considered
1. Leave repair available only as a code-level option in `reward_wrapper`.
2. Ask users to edit the Slurm shell scripts manually for each experiment.
3. Rework repair behavior inside the rewarder without exposing it through the
   CLI.

### Consequences
- `sbatch --export=ALL,ENABLE_PARENT_AWARE_REPAIR=...` style launches now
  actually affect decoded PPO runs.
- HPC diagnose jobs can sweep repair settings in the same way they already
  sweep decoded generation settings.
- The reward behavior itself is unchanged unless the user explicitly enables
  repair.

### Status
Accepted

---

## [2026-04-25] Preserve raw dummy-atom evidence in decoded PPO parse-failure logs

### Background
The decoded-chem PPO reward path already used dummy-aware normalization for
successful capped fragments such as `*CC(=O)O`, but parse-failed fragments were
still difficult to diagnose from logs alone. In particular, once raw parsing
failed, the existing trace fields could lose the evidence that the original
fragment string actually contained `*`, which made it hard to tell whether
failures mostly came from uncapped raw fragments or from the dummy-atom path
itself.

### Decision
Keep the decoded PPO chemistry objective unchanged and apply a minimal logging /
normalization refinement only in `reward_wrapper` and `train_ppo`:

- preserve the raw fragment string as the source of truth for dummy presence and
  dummy count before any RDKit parsing happens;
- continue to parse the raw fragment with `*` intact and never do string-level
  `replace("*", "")` before `MolFromSmiles`;
- keep `core_fragment` as a derived post-parse view used for teacher scoring and
  deletion checks only;
- surface explicit parse metadata in decoded PPO logs, including
  `raw_has_dummy`, `raw_dummy_count`, `parse_stage`,
  `parsed_raw_with_dummy`, `parsed_core`, `dummy_removed_before_parse`, and
  `parse_failed_reason`;
- split parse-failure buckets into
  `parse_failed_raw_with_dummy`, `parse_failed_raw_without_dummy`,
  `parse_failed_after_dummy_removal`, plus the existing obvious closure buckets
  such as `parse_failed_unclosed_ring` and
  `parse_failed_unbalanced_parentheses`;
- add per-batch counters for parse failures with and without raw dummy atoms.

### Alternatives considered
1. Keep the existing reward path unchanged and infer dummy-related failures only
   from ad hoc grep patterns.
2. Strip dummy atoms from strings before parsing so that all failures collapse
   onto core-fragment syntax.
3. Attempt automatic repair of ring digits or parentheses during reward-time
   normalization.

### Consequences
- Diagnose logs can now answer whether parse failures mostly come from raw
  fragments without `*` or from dummy-aware normalization paths.
- Successful capped fragments still use the same raw-then-core workflow, so the
  counterfactual objective and deletion logic do not change.
- The codebase now makes it explicit that dummy removal happens after raw
  parsing, not before it.

### Status
Accepted

---

## [2026-04-25] Harden decoded-chem PPO against overlong invalid fragments and surface failure buckets explicitly

### Background
After the second SFT round, decoded-chem PPO could initialize from
`checkpoint-300`, keep policy/reference aligned, and run a full 200-step
diagnose loop. The dominant failure mode, however, was no longer empty
responses. Instead, many generations failed as invalid or non-substructure
fragments, often because the decoded fragment was too long, truncated, or left
rings / brackets / parentheses unclosed. At the same time, full-parent and
empty-residual failures were still present, but their penalties were not strong
enough to clearly dominate those degenerate behaviors.

### Decision
Keep candidate-pool, selector, and SFT logic unchanged, and apply a minimal
decoded-chem PPO hardening pass only on generation / reward constraints:

- add decoded-generation-specific CLI knobs
  (`--gen-max-new-tokens`, `--gen-temperature`, `--gen-top-p`,
  `--gen-do-sample`) while keeping the legacy PPO generation flags usable;
- when decoded-chem PPO is launched without an explicit generation-length
  override, tighten its default `max_new_tokens` to `48`;
- preprocess decoded fragments before chemistry reward by stripping whitespace,
  keeping only the first line, and rejecting overlong fragments as
  `invalid_generation_too_long`;
- keep parse failures on the normal invalid path, but add an explicit
  `invalid_detail` field for obvious closure issues such as unbalanced
  parentheses, brackets, or ring digits;
- increase the default full-parent / empty-residual penalties to `-6.0` and
  `-4.0`, respectively;
- log `failure_tag`, `invalid_detail`, and generated fragment length alongside
  the existing `CHEM_REWARD_COMPONENTS` fields so bad cases can be grepped
  directly from decoded PPO diagnose runs;
- forward the new decoded-generation controls through the main HPC Slurm
  wrappers using `sbatch --export=ALL,...`.

### Alternatives considered
1. Leave generation settings untouched and only adjust the reward penalties.
2. Solve the issue in the candidate pool or selector instead of in decoded PPO.
3. Add aggressive chemistry-aware truncation that rewrites generated fragments
   rather than rejecting obviously bad strings early.

### Consequences
- Decoded PPO now pushes back earlier on the observed overlong / truncated
  invalid fragments without changing the project objective.
- Logs can distinguish `invalid_generation_too_long`,
  `invalid_or_not_substructure`, `full_parent_fragment`, and `empty_response`
  directly.
- HPC diagnose runs can sweep decoded generation settings through Slurm exports
  instead of editing shell scripts by hand.

### Status
Accepted

---

## [2026-04-25] Let decoded-chem PPO initialize both policy and reference from one explicit SFT LoRA checkpoint

### Background
The decoded chemistry PPO path is meant to start from an SFT policy rather than
from the raw base model. That becomes especially important once we want to run
PPO from a chosen SFT v2 checkpoint such as `checkpoint-300`. If the trainable
policy starts from that checkpoint but the KL reference remains the bare base
model, the KL term is misaligned and can exaggerate drift. At the same time,
collapsed generations such as empty responses, full-parent fragments, and
empty-residual deletions need to be surfaced explicitly in logs rather than
appearing as opaque chemistry failures.

### Decision
Keep the decoded-chem PPO objective unchanged, but make initialization and
anti-collapse diagnostics explicit:

- add `--sft-lora-path` as the preferred CLI name for the SFT initialization
  checkpoint while keeping the old `--sft-adapter-path` path for backward
  compatibility;
- resolve one effective SFT LoRA path and use it for both the trainable PPO
  policy and the frozen KL reference model;
- log the resolved policy/reference initialization path so HPC runs can verify
  that both models start from the same checkpoint;
- treat empty decoded responses as empty fragments instead of letting prompt
  echo accidentally fall back to the parent molecule;
- add explicit `full_parent` and `empty_residual` handling with configurable
  penalties and stable log fields (`empty_response`, `full_parent`,
  `empty_residual`, `oracle_ok`, `cf_drop`, `cf_flip`, `reward_total`).

### Alternatives considered
1. Keep using only `--sft-adapter-path` and rely on users to infer whether the
   reference model matches the policy.
2. Let the KL reference remain the raw base model even when PPO starts from an
   SFT adapter.
3. Leave empty-response and full-parent cases to be inferred indirectly from
   teacher-oracle failure reasons.

### Consequences
- PPO runs can now be launched from an explicit SFT v2 checkpoint such as
  `checkpoint-300` with policy/reference alignment preserved.
- Decoded-chem logs are easier to grep for collapse-related cases without
  changing the underlying deletion-based counterfactual objective.
- The full-training Slurm wrapper can forward the chosen SFT LoRA checkpoint
  and the new explicit penalties through `sbatch --export=ALL,...`.

### Status
Accepted

---

## [2026-04-25] Make SFT fragment-distribution audits chunkable and existence-first for HPC runs

### Background
The new SFT audit scripts are intended to characterize whether weak labels and
SFT generations collapse toward near-parent fragments or tiny trivial pieces.
On larger files, however, a few symmetric molecules caused audit runs to stall
for a long time inside RDKit substructure enumeration, which made whole-file
audits unreliable on both local machines and HPC nodes.

### Decision
Keep the audit objective unchanged, but make the audit path operationally safe:

- add chunk/window controls and progress logging to
  `scripts/analyze_sft_fragment_distribution.py`;
- isolate per-sample audit exceptions so one bad molecule does not abort the
  entire batch unless `--fail-fast` is explicitly requested;
- prefer existence-first substructure checks (`HasSubstructMatch` or capped
  queries limited to `maxMatches=1`) when the audit only needs to know whether
  a match exists;
- add cheap pruning before expensive chemistry work, including parse failures,
  fragment-larger-than-parent checks, and a full-parent shortcut based on
  canonical core equality;
- emit slow-sample records so long-running parent/fragment pairs can be
  inspected and re-run in isolated chunks on HPC.

### Alternatives considered
1. Keep the original all-in-one audit and rely on manual interruption when a
   job stalls.
2. Disable substructure and deletion checks globally, even for manageable
   molecules.
3. Rewrite the chemistry layer around a separate matching backend.

### Consequences
- SFT audits can now be submitted in bounded chunks through Slurm.
- Large runs produce explicit progress, warning, and slow-sample artifacts
  instead of appearing silently hung.
- The chemistry layer still enforces the same counterfactual-fragment
  definition, but audit-time existence checks avoid enumerating every symmetric
  match when only a yes/no answer is needed.

### Status
Accepted

---

## [2026-04-19] Add explicit teacher-semantic scoring on core fragments in the decoded chemistry PPO path

### Background
The decoded chemistry PPO loop now proves that generated text is decoded,
normalized, scored by chemistry utilities, and then used in a PPO update.
However, the logs still showed:

- `[CHEM_REWARD_COMPONENTS_MISSING] missing=teacher_sem`

That made it impossible to tell whether any auxiliary fragment-level semantic
signal was actually being applied after dummy-atom normalization.

### Decision
Add a dedicated `TeacherSemanticScorer` and wire it into the decoded chemistry
reward path only.

Key rules:

- the teacher always receives `core_fragment_smiles`, never the raw capped
  fragment with `*`;
- invalid or non-substructure fragments do not call the teacher and instead log
  a skip reason;
- when the teacher backend is unavailable, the code uses an explicit fallback
  penalty and logs the unavailability rather than pretending a real score
  exists;
- the repository's existing residual-molecule counterfactual term remains in
  place, so the teacher-semantic term is an auxiliary signal rather than a
  replacement for the deletion-based objective.

Because the repository currently ships one concrete classifier artifact at
`outputs/hpc/oracle/aids_rf_model.pkl`, the scorer first supports that
scikit-learn style bundle format (`predict_proba` plus fingerprint metadata).
Torch checkpoints are only accepted when they carry equally explicit
fingerprint configuration.

### Alternatives considered
1. Continue logging `teacher_sem` as missing.
2. Treat the residual-molecule oracle as if it already satisfied the teacher
   role and hide the distinction.
3. Require a new `teacher/teacher.pt` artifact before any teacher-semantic work
   could proceed.

### Consequences
- decoded PPO logs now expose `[TEACHER_SEM_CALLED]`,
  `[TEACHER_SEM_RESULT]`, `[TEACHER_SEM_SKIPPED]`, and
  `[TEACHER_SEM_UNAVAILABLE]`;
- `CHEM_REWARD_COMPONENTS` now shows both `teacher_sem` and the residual
  counterfactual term separately;
- the decoded chemistry smoke-test Slurm script now checks for a teacher file
  before submitting training.

### Status
Accepted

---

## [2026-04-19] Treat dummy-atom attachment points as valid decoded-fragment syntax in PPO chemistry rewards

### Background
The decoded chemistry PPO path now makes reward computation explicit, but the
rewarder was still too harsh on fragment strings containing `*`, for example
`*CC(=O)O`. In this project those stars are not arbitrary garbage characters;
they encode attachment points created by fragment cutting. If the rewarder
treated them as invalid text, PPO would incorrectly learn that many chemically
meaningful capped fragments were malformed.

### Decision
Keep two fragment views inside `src/rewards/reward_wrapper.py`:

- `raw_fragment_smiles`: the exact decoded fragment candidate, which may contain
  dummy atoms such as `*`;
- `core_fragment_smiles`: a dummy-free core used for substructure checks,
  compactness statistics, and any future fragment-level teacher signal.

The rewarder now:

- parses capped fragments with dummy-aware RDKit sanitization;
- removes dummy atoms through molecule editing instead of string replacement;
- checks parent substructure on the core fragment rather than on the raw capped
  string;
- counts fragment size on non-dummy atoms only;
- exposes raw/core parse flags and dummy counts in reward traces and decoded PPO
  logs.

### Alternatives considered
1. Keep treating all `*` tokens as invalid output.
2. Strip `*` with naive string replacement before every reward computation.
3. Move the dummy-aware normalization into TRL adapters instead of the chemistry
   rewarder.

### Consequences
- Decoded PPO logs can now distinguish raw capped fragments from their
  dummy-free core.
- Validity and substructure rewards no longer collapse to zero solely because a
  fragment uses attachment-point notation.
- The chemistry reward path remains honest about what is still missing, such as
  any dedicated teacher-semantic term.

### Status
Accepted

---

## [2026-04-18] Add a decoded-SMILES chemistry reward PPO loop alongside the TRL compatibility baseline

### Background
The repository's TRL experimental PPO smoke test now runs end to end, but the
successful path still relies on a hidden-state reward adapter that only
validates trainer interface compatibility. The logs already make this
limitation explicit:

- `ChemRewardModelWrapper remains the chemistry reward component and is not equivalent to TRL hidden-state reward head`

That means the baseline does not prove that decoded fragment strings are being
scored by the chemistry reward and then used for PPO updates.

### Decision
Keep the TRL experimental path as the engineering baseline, but add a second
training mode in `scripts/train_ppo.py`:

- `--ppo-loop decoded_chem`

This mode performs the reward flow explicitly:

- prompt batch
- `policy_model.generate()`
- decode generated response text
- extract one fragment candidate
- call `ChemRLRewarder.compute_rewards_from_decoded(...)`
- run a local PPO-style update with policy logprobs, reference logprobs,
  token-aligned value predictions, KL-shaped rewards, clipped policy loss, and
  clipped value loss

The CLI also now supports:

- `--require-chemistry-reward-path`
- `--decoded-chem-smoke-test`

and the repository includes a dedicated Slurm smoke-test script:

- `scripts/slurm/debug_decoded_chem_ppo_smoketest.sh`

### Alternatives considered
1. Keep using the TRL hidden-state reward adapter and treat it as good enough.
2. Patch TRL site-packages until they can consume the chemistry wrapper
   directly.
3. Block all further work until a legacy `PPOTrainer.step(...)` API is proven
   available in every environment.

### Consequences
- The repository now has one baseline path for TRL interface compatibility and
  one separate path that makes decoded-SMILES chemistry rewards enter PPO
  updates explicitly.
- Smoke-test logs can now distinguish “trainer compatibility succeeded” from
  “decoded chemistry reward was called and used in an update”.
- The local PPO loop stays inside repository code, so no external environment
  patching is required.

### Status
Accepted

---

## [2026-04-18] Skip trainer-managed completion previews when experimental PPO has no usable eval dataloader

### Background
After the ChemLLM cache fix, the value-model `.score` adapter, and the
reward-model compatibility adapter were all in place, the PPO smoke test
finally entered the trainer loop and emitted PPO metrics. The next failure came
from `trl.experimental.ppo.PPOTrainer.generate_completions()`, which tried to
iterate `self.eval_dataloader` even though the smoke-test path had no usable
evaluation data source. That eventually surfaced as:

- `TypeError: object of type 'NoneType' has no len()`

### Decision
Keep the main PPO training loop intact, but add a local repository-side guard
in `scripts/train_ppo.py` for the trainer-managed completion-preview stage:

- add `--skip-generate-completions` as an explicit CLI escape hatch;
- add `--diagnose-reward-flow` as a smoke-test-friendly debug flag;
- detect unusable completion-preview loaders such as missing
  `eval_dataloader`, missing `dataset`, or `sampler.data_source is None`;
- replace `ppo_trainer.generate_completions()` with a no-op logger only when
  the skip flag is enabled or the evaluation loader is clearly unusable.

The HPC smoke-test script now always passes both diagnostic flags so it can
focus on initialization and the core PPO loop without failing in the preview
generation branch.

### Alternatives considered
1. Patch `trl.experimental` directly in site-packages to special-case missing
   eval loaders.
2. Fabricate a fake evaluation dataset just to satisfy completion previews.
3. Catch and suppress broad exceptions around `ppo_trainer.train()`.

### Consequences
- The smoke test can keep exercising PPO initialization and training steps even
  when the trainer's optional preview-generation path has no evaluation data.
- Main training errors are still surfaced normally because only
  `generate_completions()` is replaced, not the full trainer loop.
- Slurm logs now include explicit `[PPO_GENERATE_COMPLETIONS_SKIPPED]` markers
  so it is obvious when this guard was applied.

### Status
Accepted

---

## [2026-04-18] Separate chemistry reward logic from TRL experimental reward-model interface compatibility

### Background
After the local `.score` adapter fixed the PPO critic-side value-model crash,
the smoke test progressed further and then failed inside
`trl.experimental.utils.get_reward()` with:

- `AttributeError: 'ChemRewardModelWrapper' object has no attribute 'base_model_prefix'`

The repository's `ChemRewardModelWrapper` computes chemistry-aware rewards by
decoding generated text back into parent / fragment pairs and calling
`ChemRLRewarder`. That is not the same interface as the Hugging Face-style
reward model expected by this experimental TRL path, which assumes:

- `base_model_prefix`
- a forwardable LM backbone accessible through that prefix
- `score(hidden_states)`

### Decision
Keep `ChemRewardModelWrapper` as the repository's chemistry reward component,
but stop passing it directly to experimental PPO when the runtime expects a
hidden-state reward model.

Instead, `scripts/train_ppo.py` now adds
`ensure_reward_model_for_experimental_ppo(...)`, which:

- reuses an existing reward-side backbone if one is already exposed;
- otherwise builds a lightweight TRL-compatible reward adapter around a
  fallback LM backbone such as `value_model.pretrained_model`;
- adds `base_model_prefix` and a token-level `score` head for interface
  compatibility;
- logs explicitly that the fallback adapter is only for smoke-test /
  interface-validation purposes and is not equivalent to the repository's
  deletion-based chemistry reward objective.

### Alternatives considered
1. Patch TRL directly in site-packages so it can accept the chemistry wrapper.
2. Pretend the chemistry wrapper is a native hidden-state reward model by
   adding only one missing attribute at a time.
3. Remove the chemistry reward component entirely and silently replace it with
   a generic reward head.

### Consequences
- The smoke test can progress past TRL's stricter `reward_model` interface
  checks without mutating the external conda environment.
- Repository code now makes the mismatch between chemistry rewards and TRL's
  hidden-state reward-model contract explicit instead of hiding it behind
  brittle monkey patches.
- Future work can reconnect true chemistry rewards to trainer-managed PPO more
  cleanly because the current limitation is now documented in code and docs.

### Status
Accepted

---

## [2026-04-18] Attach a local `.score` adapter for TRL value-head critics in experimental PPO

### Background
The ChemLLM PPO smoke test moved past the earlier InternLM2 cache failure, but
then failed inside `trl.experimental.utils.get_reward()` when the trainer tried
to evaluate the critic with:

- `model.score(output.hidden_states[-1])`

Our repository-side `value_model` was still
`AutoModelForCausalLMWithValueHead`, which exposes `v_head` rather than a
top-level `.score` method. Patching TRL in site-packages was explicitly out of
scope for the VS Code plus Git plus HPC workflow.

### Decision
Add a repository-local compatibility helper in `scripts/train_ppo.py`:

- `ensure_score_head_for_experimental_ppo(model, name=...)`

The helper now:

- leaves models that already expose `.score` unchanged;
- searches the top-level object and common wrapper layers such as
  `pretrained_model`, `base_model`, and `model` for a reachable `v_head`;
- attaches `model.score(hidden_states)` dynamically when only `v_head` exists;
- logs which wrapper layer supplied the adapted value head.

The adapter is applied to `value_model` before `PPOTrainer` construction. The
policy and reference models remain untouched so generation behavior stays
isolated from this critic-side interface patch.

### Alternatives considered
1. Patch `trl.experimental` directly inside the conda environment.
2. Replace the value wrapper again with another custom critic type.
3. Fork the PPO rollout path away from the trainer-managed `get_reward()`
   utility.

### Consequences
- Experimental PPO can keep using the existing TRL value-head wrapper while
  satisfying the newer `.score(hidden_states)` critic contract.
- The compatibility layer stays local to repository code and is therefore easy
  to review, sync to HPC, and remove later if upstream APIs converge.
- Smoke-test logs now make it explicit whether the value model has both
  `v_head` and the adapted `.score` interface before training begins.

### Status
Accepted

---

## [2026-04-09] Rebuild repository from scratch with documentation-first workflow

### Background
The previous project evolved through multiple experimental iterations. It likely accumulated script coupling, outdated assumptions, and mixed objectives inherited from earlier versions. To prevent repeated confusion during reconstruction, the repository needs a stable written definition before code implementation begins.

### Decision
Rebuild the repository from an empty root using a documentation-first workflow. The initial authoritative files are:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

These files define the objective, engineering rules, roadmap, and future design log.

### Alternatives considered
1. Start coding immediately and add documentation later.
2. Recover the old repository structure first and refactor afterward.
3. Keep all design notes only in chat history.

### Consequences
- The repository gets a clear source of truth from the start.
- Codex can be guided by repository-local instructions instead of relying on conversational memory.
- Early progress is slower in appearance but more stable in direction.

### Status
Accepted

---

## [2026-04-16] Make ChemLLM cache patch tool classify guarded vs unguarded accesses

### Background
The first repository-side ChemLLM cache patch tool could add the helper and
patch the forward-path cache-length logic, but its multiline matching for
`prepare_inputs_for_generation()` was too brittle. As soon as the target block
contained nested indentation, comments, or slightly different formatting, the
prepare-path patch silently failed. The checker output also counted every
`past_key_values[0][0].shape[2]` occurrence as equally dangerous even after it
had been moved under the new guard.

### Decision
Rework `tools/check_or_patch_chemllm_cache.py` to use indentation-aware,
line-based patching and reporting:

- patch the forward cache-length block and the
  `prepare_inputs_for_generation()` block independently;
- insert `else: past_key_values = None` / `past_length = 0` for the prepare
  path;
- detect already-patched blocks and skip them cleanly;
- classify dangerous accesses as either `guarded` or `unguarded`, and treat
  `unguarded_count=0` as the real success criterion.

### Alternatives considered
1. Keep the old regexes and only tweak them slightly.
2. Require manual patching for the prepare path on HPC.
3. Keep counting all `shape[2]` accesses as equally dangerous.

### Consequences
- The repository-side helper can now patch both critical cache branches before
  HPC smoke testing.
- Patch results are easier to interpret because protected accesses are no
  longer reported as unresolved failures.
- HPC documentation can now point users to `unguarded_count=0` instead of
  incorrectly suggesting that all `shape[2]` accesses must disappear.

### Status
Accepted

---

## [2026-04-16] Add PPO runtime import-path introspection for ChemLLM cache debugging

### Background
The ChemLLM / InternLM2 PPO path hit a cache-related generation crash inside
`modeling_internlm2.py`, but local repository edits alone were not enough to
prove which dynamically cached file the Slurm job was actually importing at
runtime. In a VS Code plus Git plus HPC workflow, the repository copy, the
Hugging Face dynamic module cache, and the job's working directory can diverge.

### Decision
Add lightweight runtime introspection to `scripts/train_ppo.py` so every PPO
run logs:

- the wrapped policy / reference / value model module names;
- the resolved module source files;
- the resolved `prepare_inputs_for_generation` source files;
- key environment variables such as `PYTHONPATH`, `HF_HOME`,
  `TRANSFORMERS_CACHE`, and `HUGGINGFACE_HUB_CACHE`.

Also add a dedicated Slurm smoke-test entrypoint:

- `scripts/slurm/debug_check_chemllm_runtime_path.sh`

that reuses the normal HPC environment bootstrap, prints repository and Python
runtime information, and runs a tiny PPO smoke test through
`scripts/train_rl.py`.

### Alternatives considered
1. Keep reasoning about cache behavior from static local files only.
2. Patch more InternLM2 code blindly without first proving the runtime import
   path.
3. Ask users to manually add debug prints on the HPC side.

### Consequences
- Future Slurm logs can show exactly which `modeling_internlm2.py` was imported
  during PPO generation.
- Runtime path mismatches between repository code and Hugging Face dynamic cache
  are easier to detect before chasing deeper trainer bugs.
- The repository now has a reusable HPC-first smoke test for ChemLLM runtime
  path debugging.

### Status
Accepted

---

## [2026-04-15] Implement first PPO training path with residual-based reward wrapper

### Background
The repository already had a trained SFT adapter, a lightweight AIDS oracle,
and a chemistry layer for capped fragment validation and deletion. What was
still missing was a real PPO training path that could optimize ChemLLM outputs
toward the deletion-based counterfactual objective on HPC hardware.

### Decision
Add a concrete PPO training entrypoint in `scripts/train_ppo.py` together with a
unified reward wrapper in `src/rewards/reward_wrapper.py`.

The reward wrapper uses a three-stage early-stop flow:

- parseability and connectedness checks;
- parent-subgraph validation for capped fragments;
- residual-molecule scoring after deleting the fragment from the parent.

The semantic reward term is computed on the residual graph rather than on the
fragment alone, because the v3 objective is label flipping after deletion.

The PPO script loads:

- ChemLLM-7B-Chat in 4-bit mode;
- the SFT LoRA checkpoint as the initial policy;
- a frozen LoRA-backed reference model for KL control;
- prompts from either the raw HIV CSV or a JSONL prompt file.

The training loop logs reward statistics, structural pass rates, simple
collapse diagnostics, and representative generations into the run directory.

### Alternatives considered
1. Keep RL as a runtime-preparation placeholder only.
2. Score the fragment alone with the oracle instead of the residual molecule.
3. Hardcode one dataset schema and refuse either CSV or JSONL prompts.

### Consequences
- The repository now has a runnable PPO-stage backbone aligned with the
  counterfactual v3 objective.
- KL control remains anchored to the SFT adapter rather than drifting from the
  base model directly.
- PPO runs surface chemistry and collapse signals explicitly instead of hiding
  them inside trainer internals.

### Status
Accepted

---

## [2026-04-15] Switch PPO policy loading back to native causal LM for experimental TRL

### Background
After the first PPO entrypoint landed, newer `trl.experimental.ppo` builds
showed constructor-time incompatibilities with explicit value-head wrappers,
including failures around missing wrapper attributes such as
`base_model_prefix`. This indicated that the trainer now expects native causal
LM policies and prefers to manage policy/value wrapping internally.

### Decision
Update `scripts/train_ppo.py` so that PPO loads native
`AutoModelForCausalLM`-style PEFT models for both the trainable policy and the
frozen reference policy, without constructing
`AutoModelForCausalLMWithValueHead`.

The PPO trainer initialization is now version-adaptive in three ways:

- `PPOConfig` kwargs are filtered against the runtime signature;
- trainer kwargs map across `args/config`, `model/policy`, and
  `ref_model/ref_policy` variants;
- external `value_model` is omitted or forced to `None` so TRL can own value
  wrapping internally.

The script also provides a lightweight reward adapter that exposes
`ChemRLRewarder` as either `reward_model` or `reward_funcs` when those newer
experimental hooks are present.

### Alternatives considered
1. Keep the explicit value-head wrapper and patch missing attributes one by one.
2. Pin the repository to one older TRL version instead of adapting the code.
3. Move the whole PPO path back to a handwritten optimizer loop.

### Consequences
- The PPO script is better aligned with current experimental TRL architecture.
- Fewer wrapper-specific attribute mismatches should appear when the trainer is
  upgraded.
- The script still keeps the repository's deletion-based counterfactual reward
  logic outside trainer internals.

### Status
Accepted

---

## [2026-04-15] Use explicit value and reward models for experimental PPOTrainer

### Background
Follow-up PPO integration work revealed another compatibility shift in newer
`trl.experimental.ppo.PPOTrainer` builds: the trainer now expects a real
transformers-style `value_model` object and also routes internal scoring
through a PyTorch `reward_model(input_ids=..., attention_mask=..., ...)`
interface.

### Decision
Keep the policy and reference networks as native causal LMs, but explicitly add:

- a 4-bit `AutoModelForSequenceClassification` value model with `num_labels=1`;
- a torch `reward_model` wrapper that decodes generated sequences back to text,
  reconstructs the parent / fragment pair, and calls `ChemRLRewarder`.

The trainer initialization in `scripts/train_ppo.py` now wires:

- `model`: native causal LM policy with the SFT LoRA adapter;
- `ref_model`: frozen native causal LM reference policy;
- `value_model`: explicit scalar sequence-classification model;
- `reward_model`: deletion-based chemistry reward wrapper.

### Alternatives considered
1. Continue passing `None` for the value model and rely on implicit trainer behavior.
2. Keep reward scoring entirely outside the trainer and ignore new internal hooks.
3. Reintroduce custom non-transformers wrapper classes around the value head.

### Consequences
- PPO initialization is better aligned with stricter experimental TRL releases.
- Internal trainer scoring can now call a PyTorch-compatible reward interface
  without losing the repository's residual-graph counterfactual objective.
- The value network contract is explicit instead of version-dependent.

### Status
Accepted

---

## [2026-04-15] Fall back to TRL value-head wrapper for InternLM2 PPO value model

### Background
The explicit sequence-classification critic introduced for experimental PPO
alignment turned out to be incompatible with the ChemLLM / InternLM2 base
checkpoint in environments where the InternLM2 config is not registered under
Hugging Face's `AutoModelForSequenceClassification` auto-mapping.

### Decision
Replace the PPO `value_model` in `scripts/train_ppo.py` with
`trl.AutoModelForCausalLMWithValueHead` loaded on top of the same quantized
ChemLLM base weights, and monkey-patch:

- `value_model.base_model_prefix = "pretrained_model"`

so that stricter `trl.experimental.ppo.PPOTrainer` builds can still treat the
wrapper like a native model during critic setup.

Only the wrapper's value head remains trainable; the wrapped causal LM backbone
stays frozen.

### Alternatives considered
1. Keep using `AutoModelForSequenceClassification` and require a newer
   InternLM2 registration patch from transformers.
2. Drop experimental PPOTrainer and fully hand-roll critic/value optimization.
3. Try to emulate a sequence-classification head with another custom wrapper.

### Consequences
- PPO critic initialization no longer depends on InternLM2 being registered in
  the sequence-classification auto-model registry.
- The project keeps the reward path aligned to the deletion-based
  counterfactual objective while staying closer to TRL's expected interfaces.
- The monkey patch is intentionally local to the PPO script and should be
  revisited if upstream TRL or transformers support improves.

### Status
Accepted

---

## [2026-04-15] Let experimental PPOTrainer own rollout and optimization

### Background
Once the experimental PPO trainer, explicit value model, and reward model were
all wired successfully, the remaining failures came from the old handwritten
training loop that still tried to call trainer-side generation and manual PPO
updates directly.

### Decision
Remove the legacy step-by-step PPO loop from `scripts/train_ppo.py` and switch
the entrypoint to the trainer-managed flow:

- initialize policy, reference, value, and reward models;
- construct `PPOTrainer`;
- call `ppo_trainer.train()`;
- save the final checkpoint via the trainer's own save path.

The PPO config assembly now also forwards `max_steps` and generation-related
kwargs when the runtime `PPOConfig` signature supports them.

### Alternatives considered
1. Keep maintaining a hybrid script that mixes experimental trainer internals
   with a manual rollout loop.
2. Revert fully to a classic step-based TRL API.
3. Move rollout back outside the trainer and bypass `reward_model`.

### Consequences
- The PPO entrypoint now matches the ownership model of newer
  `trl.experimental.ppo` releases more closely.
- Fewer incompatibilities should appear around missing `generate()` or `step()`
  methods on experimental trainer objects.
- Reward evaluation stays aligned with the repository's residual-graph
  counterfactual objective, but execution control is delegated to TRL.

### Status
Accepted

---

## [2026-04-15] Run PPO WandB logging in offline mode for HPC nodes

### Background
The PPO training entrypoint is designed for HPC execution, but compute nodes in
the target environment do not have outbound internet access. Direct online
Weights & Biases logging would therefore stall or fail during trainer startup.

### Decision
Configure `scripts/train_ppo.py` to force WandB offline mode via environment
variables:

- `WANDB_MODE=offline`
- `WANDB_SILENT=true`

At the same time, keep the trainer-side reporting target aligned to WandB when
supported by the runtime PPO config, so metrics are still written locally into
the standard `wandb/` directory for later sync.

The PPO config builder now forwards:

- `report_to="wandb"` when supported;
- `log_with="wandb"` for older compatible signatures;
- `run_name="ppo_aids_rl_v1"` as the semantic experiment label.

### Alternatives considered
1. Disable WandB entirely on HPC and rely only on stdout or custom JSON logs.
2. Require a separate shell wrapper to export WandB offline variables.
3. Keep online WandB enabled and tolerate repeated timeout failures.

### Consequences
- PPO runs on air-gapped or internal-only HPC nodes no longer depend on live
  WandB connectivity.
- Local WandB artifacts remain available for later `wandb sync`.
- Experiment naming is more stable across local logs and offline WandB runs.

### Status
Accepted

---

## [2026-04-16] Force PPO datasets to emit tensors before entering experimental trainer

### Background
After the PPO entrypoint successfully crossed model and trainer initialization,
the first runtime failure inside `trl.experimental.ppo.PPOTrainer.train()`
occurred when the trainer tried to move `data["input_ids"]` onto the device.
The immediate cause was that the Hugging Face dataset / collator path was still
yielding Python lists instead of PyTorch tensors.

### Decision
Update `scripts/train_ppo.py` so that the PPO training dataset explicitly calls:

- `dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)`

after tokenization, and replace the custom collator with a tensor-aware version
that:

- pads `input_ids` through `tokenizer.pad(..., return_tensors="pt")`;
- emits `input_ids` and `attention_mask` as tensors;
- preserves text metadata fields such as `query` and `parent_smiles` as Python
  lists for downstream reward reconstruction.

The reward wrapper's decoding path also now detaches tensor inputs to CPU
before `batch_decode`, while keeping the returned reward tensor on the same
device as `input_ids`.

### Alternatives considered
1. Rely only on `set_format("torch")` and keep the old collator unchanged.
2. Drop the custom collator entirely and hope the trainer default handles mixed
   text-and-tensor batches correctly.
3. Convert data inside the trainer loop instead of fixing the dataset contract.

### Consequences
- Experimental PPOTrainer can now consume `input_ids` with a valid `.to(device)`
  path.
- The batch contract is more explicit and robust against future TRL internal
  assumptions.
- Reward evaluation remains device-safe even when the trainer feeds tensor
  batches directly from GPU-backed training steps.

### Status
Accepted

---

## [2026-04-16] Use DataCollatorWithPadding-backed PPO collator for tensor-safe batches

### Background
After forcing the PPO dataset into torch format, the next trainer failure still
occurred at batch materialization time because the custom collator path was not
reliably returning a dictionary of tensors. In practice, the trainer ended up
receiving `None` instead of a batch payload.

### Decision
Replace the fragile handcrafted padding path in `scripts/train_ppo.py` with a
wrapper around Hugging Face's standard `DataCollatorWithPadding`, configured
with `return_tensors="pt"`.

The PPO collator now:

- delegates token padding to `DataCollatorWithPadding`;
- guarantees a returned batch dictionary;
- validates that `input_ids` exists and is a tensor;
- preserves non-model metadata fields such as `query`, `parent_smiles`, and
  `original_label` as Python lists for downstream reward reconstruction.

### Alternatives considered
1. Keep the custom collator and only add a missing `return batch`.
2. Drop metadata preservation and use a raw Hugging Face collator directly.
3. Remove the collator override and rely fully on trainer defaults.

### Consequences
- PPO batches now have a much stronger contract before they enter
  `trl.experimental.ppo.PPOTrainer`.
- Standard padding behavior is delegated to a well-tested transformers utility.
- Reward logic can still reconstruct prompt context without coupling that logic
  to the token padding implementation.

### Status
Accepted

---

## [2026-04-16] Monkey-patch experimental PPO wrapper to expose gradient checkpointing hooks

### Background
After the PPO data path was repaired, the next runtime failure came from
`trl.experimental.ppo` itself: the trainer's internal `PolicyAndValueWrapper`
was missing `gradient_checkpointing_disable` and
`gradient_checkpointing_enable`, even though the wrapped policy model already
implemented them.

### Decision
Patch the trainer-managed wrapper immediately after PPO trainer construction in
`scripts/train_ppo.py`:

- if `ppo_trainer.model.policy_model.gradient_checkpointing_disable` exists,
  bind it onto `ppo_trainer.model.gradient_checkpointing_disable`;
- do the same for `gradient_checkpointing_enable`.

In parallel, the PPO config builder now explicitly forwards
`gradient_checkpointing=False` when the runtime config signature supports that
field, so trainer-side generation does not try to rely on checkpoint toggling
more than necessary.

### Alternatives considered
1. Wait for an upstream TRL patch and leave the local training script broken.
2. Disable wrapper-managed generation entirely and fall back to a handwritten
   PPO rollout loop again.
3. Monkey-patch TRL package files directly in the environment.

### Consequences
- The repository no longer depends on an immediate upstream TRL fix for this
  wrapper attribute bug.
- The workaround stays local to the PPO entrypoint instead of mutating
  site-packages.
- Generation-time checkpoint toggling is less likely to derail short HPC PPO
  smoke tests.

### Status
Accepted

---

## [2026-04-16] Escalate PPO wrapper gradient-checkpointing fix from instance patch to class patch

### Background
The first local workaround for the experimental TRL wrapper bug patched the
trainer instance after construction. That turned out to be insufficient because
the runtime path could still recreate or access a wrapper object that had not
received the instance-local method bindings.

### Decision
Move the gradient-checkpointing workaround into the TRL import phase inside
`scripts/train_ppo.py` by patching the wrapper class itself:

- import `trl.experimental.ppo.ppo_trainer` when available;
- if `PolicyAndValueWrapper` exists, inject no-op
  `gradient_checkpointing_disable` and `gradient_checkpointing_enable` methods
  onto the class when they are missing.

This class-level patch supersedes the earlier instance-level workaround. The
config-side guard `gradient_checkpointing=False` remains in place as a second
line of defense.

### Alternatives considered
1. Keep stacking more instance-local patches after trainer construction.
2. Patch TRL directly inside site-packages on the target machine.
3. Revert to non-experimental PPO code paths entirely.

### Consequences
- Any `PolicyAndValueWrapper` instantiated after the patch inherits the missing
  methods automatically.
- The PPO script becomes less sensitive to internal wrapper recreation inside
  experimental TRL.
- The workaround remains local to the repository instead of mutating external
  package files.

### Status
Accepted

---

## [2026-04-16] Disable InternLM2 KV cache across PPO trainer wrappers before rollout

### Background
After the PPO entrypoint finally entered the trainer-managed batch generation
loop, ChemLLM's InternLM2 generation stack still failed inside
`modeling_internlm2.py` when the runtime attempted to consume incompatible
`past_key_values`. This failure occurred during PPO rollout rather than during
model initialization.

### Decision
Add a trainer-side runtime patch in `scripts/train_ppo.py` that recursively
walks the trainer model and common wrapper attributes such as:

- `policy_model`
- `pretrained_model`
- `model`
- `base_model`

For each discovered layer, the patch:

- forces `config.use_cache = False` when available;
- forces `generation_config.use_cache = False` when available;
- synchronizes `pad_token_id` and `eos_token_id` with the tokenizer.

This patch is applied immediately after PPO trainer construction and before
`ppo_trainer.train()`.

### Alternatives considered
1. Rely only on the earlier model-loading-time `use_cache=False` settings.
2. Try to sanitize or rewrite `past_key_values` formats for InternLM2.
3. Revert back to a completely manual rollout loop outside experimental TRL.

### Consequences
- PPO rollout is less likely to trigger InternLM2 KV-cache compatibility bugs
  through deeply wrapped trainer models.
- Token id settings stay aligned between tokenizer and generation config at the
  exact point where trainer-managed generation begins.
- The workaround remains local to the repository and does not require patching
  upstream model files.

### Status
Accepted

---

## [2026-04-16] Replace static InternLM2 cache patch with generate-method hijacking

### Background
Even after synchronizing `use_cache=False` into wrapped model configs, the PPO
runtime could still reintroduce cache-related generation kwargs dynamically
through experimental TRL batch generation. As a result, InternLM2 continued to
receive incompatible cache inputs during rollout.

### Decision
Change the PPO runtime workaround in `scripts/train_ppo.py` from static
generation-config edits to method hijacking:

- keep tokenizer-aligned `pad_token_id` / `eos_token_id` synchronization;
- wrap `generate()` on the trainer model, policy model, and base model when
  available;
- force `kwargs["use_cache"] = False` on every generation call;
- drop `past_key_values` from generation kwargs before delegating to the
  original method.

### Alternatives considered
1. Keep stacking more static `config.use_cache=False` assignments.
2. Patch InternLM2 model code directly.
3. Revert back to a custom manual rollout loop outside experimental TRL.

### Consequences
- Cache disabling now applies at the exact generation call boundary where TRL
  injects kwargs.
- The workaround is less sensitive to internal trainer overrides of static
  generation config.
- The script keeps a narrow, local compatibility layer without mutating
  external package files.

### Status
Accepted

---

## [2026-04-16] Override InternLM2 prepare_inputs_for_generation at the deepest wrapped class

### Background
The generate-method hijack still proved insufficient once TRL, PEFT, and the
current transformers cache stack interacted through multiple wrapper layers.
InternLM2 continued to fail inside its own `prepare_inputs_for_generation`
implementation because the expected tuple-style cache structure no longer
matched what the runtime was trying to pass.

### Decision
Replace the generate-method interception in `scripts/train_ppo.py` with a
deeper class-level patch that:

- unwraps the trainer model through common wrapper attributes such as
  `policy_model`, `base_model`, and `model`;
- identifies the actual underlying causal LM class;
- overrides `prepare_inputs_for_generation` on that class;
- forces `past_key_values=None` and `use_cache=False` on every call.

Tokenizer-aligned `pad_token_id` / `eos_token_id` synchronization remains in
place before the patch is applied.

### Alternatives considered
1. Keep layering more `generate()` wrappers on top of TRL and PEFT.
2. Patch InternLM2 source files directly in the runtime environment.
3. Abandon trainer-managed generation and return to a manual PPO loop.

### Consequences
- The cache-breaking branch is cut off at the exact InternLM2 method where the
  incompatibility manifests.
- The fix is more resilient to outer wrapper churn because it targets the
  effective model class rather than only the outermost object.
- The workaround stays local to repository code and can be removed later if
  upstream InternLM2 / transformers compatibility improves.

### Status
Accepted

---

## [2026-04-09] Treat counterfactual fragment generation as the sole primary objective

### Background
Earlier project stages and surrounding discussions may involve concept extraction, class-indicative subgraphs, or rationale-like objectives. Those formulations are related but not identical to the present v3 goal.

### Decision
Define the project strictly as counterfactual fragment generation: the generated fragment should be useful because deleting it is likely to flip the label.

### Alternatives considered
1. Optimize for fragment-only label predictiveness.
2. Optimize for class-shared concept extraction.
3. Optimize for rationale sufficiency rather than deletion-based effect.

### Consequences
- Reward design must include deletion-based semantics.
- Evaluation must include flip-related metrics.
- Old code paths aligned to non-counterfactual targets should be treated as legacy behavior.

### Status
Accepted

---

## [2026-04-09] Use modular repository structure rather than monolithic training script

### Background
The project involves chemistry checks, prompt generation, SFT, RL, evaluation, and logging. A single large script would make the system difficult to debug and easy to break.

### Decision
Adopt a modular structure with separate folders for data, models, rewards, training, evaluation, chemistry utilities, and general utilities.

### Alternatives considered
1. Keep everything in one file for convenience.
2. Split only by training stage.
3. Organize by experiments rather than functionality.

### Consequences
- Initial setup requires more files.
- The system becomes easier to test, replace, and extend.
- Reward and chemistry behavior can be validated independently.

### Status
Accepted

---

## [2026-04-09] Prioritize chemistry and reward layers before large-scale training code

### Background
The most brittle parts of this project are likely to be chemistry correctness and reward semantics. If these layers are unstable, training results will be misleading.

### Decision
Implement chemistry utilities and reward computation before building full SFT/RL training pipelines.

### Alternatives considered
1. Start with full training script and fill utility functions later.
2. Start with model wrapper only.
3. Start with evaluation only.

### Consequences
- Early iteration focuses on correctness rather than speed.
- The train loop can remain thinner and easier to debug.
- Reward failures and chemistry mismatches can be unit-tested.

### Status
Accepted

---

## [2026-04-09] Bootstrap the repository with typed module interfaces before any training implementation

### Background
After the documentation-first reset, the repository still had no source tree, no config folders, and no stable interface boundaries. If training code were introduced at that point, chemistry logic, reward semantics, and CLI behavior would likely become coupled again.

### Decision
Create the repository skeleton first and define minimum typed interfaces for:

- `src/data/`
- `src/chem/`
- `src/rewards/`
- `src/models/`
- `src/train/`
- `src/eval/`
- `src/utils/`

Also add thin placeholder CLI files under `scripts/` and minimal smoke tests for prompt and reward contracts.

The bootstrap intentionally implements only safe low-level helpers such as JSONL IO, prompt construction, reward-term aggregation, and text-level collapse diagnostics. RDKit-backed chemistry logic and all training loops remain deferred.

### Alternatives considered
1. Start with one large training script and refactor later.
2. Implement RDKit parsing and training code immediately without freezing interfaces first.
3. Leave the directory layout undocumented and decide file boundaries ad hoc.

### Consequences
- Future chemistry, reward, inference, and training work now has a stable import surface.
- The repository stays aligned to the deletion-based counterfactual objective instead of drifting toward concept extraction.
- Scripts remain thin by design, because domain logic now has clear module ownership.
- The next implementation phase can focus on chemistry correctness rather than file organization.

### Status
Accepted

---

## [2026-04-09] Implement chemistry utilities with RDKit-first behavior and explicit failure types

### Background
The v3 objective depends on structural correctness before any reward or training logic can be trusted. The repository therefore needs a chemistry layer that can safely parse SMILES, check connectedness and parent-substructure relations, and attempt fragment deletion without forcing train code to interpret raw RDKit exceptions.

### Decision
Implement the first chemistry utility layer in:

- `src/chem/smiles_utils.py`
- `src/chem/substructure.py`
- `src/chem/deletion.py`
- `src/chem/validation.py`

The layer is RDKit-first when RDKit is available, but it must also degrade safely when RDKit is missing by returning structured failure types instead of crashing. The shared result dataclasses now carry normalized failure categories, and `validate_fragment_candidate(...)` aggregates these into one interface for downstream reward and evaluation code.

### Alternatives considered
1. Raise raw exceptions from RDKit and let train or eval scripts handle them.
2. Hard-require RDKit at import time and fail the whole repository if it is absent.
3. Delay chemistry implementation until reward or inference code exists.

### Consequences
- Reward and evaluation modules can consume chemistry results without duplicating error handling.
- The repository remains usable in environments where RDKit has not been installed yet.
- Deletion behavior is deterministic at the interface level, with the first matched connected fragment removed and failures surfaced explicitly.

### Status
Accepted

---

## [2026-04-10] Add a config-driven local and HPC runtime layer without hardcoded paths

### Background
The rebuilt repository now has modular source folders and chemistry utilities, but it still needs a stable way to run from a local workstation and from an HPC cluster. Without a shared runtime layer, path handling, log directories, and environment-specific behavior would drift into ad hoc script logic.

### Decision
Add a runtime adaptation layer based on repository-relative config files and thin CLI entrypoints.

This layer includes:

- base, local, hpc, sft, rl, and eval config files under `configs/`
- environment detection and config merging in `src/utils/env.py`
- repository-relative path resolution in `src/utils/paths.py`
- file-backed logging helpers in `src/utils/logging_utils.py`
- run entrypoints in `scripts/run_sft.py`, `scripts/run_rl.py`, `scripts/run_eval.py`, and `scripts/run_infer.py`
- Slurm templates for single-node single-GPU execution under `scripts/slurm/`

Model and tokenizer handling must support local filesystem paths and avoid silent remote downloads by default.

### Alternatives considered
1. Hardcode local and HPC paths directly inside the stage scripts.
2. Depend on an external YAML package before adding any runtime config support.
3. Jump directly to distributed training setup before the single-node path is stable.

### Consequences
- Local development and HPC submission now share one config and path resolution story.
- The codebase remains portable because configs stay relative and resolved at runtime.
- The repository can prepare deterministic run manifests before full SFT, RL, and evaluation logic is implemented.
- Distributed training remains intentionally out of scope for this phase.

### Status
Accepted

---

## [2026-04-10] Implement a minimal single-sample inference loop with heuristic fragment generation

### Background
The repository already has chemistry utilities and runtime adaptation, but it still needs a runnable end-to-end path from one parent SMILES to one fragment candidate. This is necessary to validate the IO contract before wiring any full SFT or RL training logic.

### Decision
Implement a minimal inference closure centered on `scripts/run_infer.py` and `src/eval/inference.py`.

This path:

- accepts one parent SMILES from CLI or config
- produces one heuristic fragment candidate without using a trained policy
- runs chemistry utilities for parseability, connectedness, and parent-substructure checks
- prints a structured JSON result
- stays independent of the full training stack

The heuristic prefers small connected parent substructures when chemistry backends are available, and falls back to the parent SMILES when no smaller valid candidate can be established.

### Alternatives considered
1. Keep `run_infer.py` as config-only runtime preparation.
2. Wait for a trained model before implementing any inference path.
3. Print only free-form text instead of a machine-readable result.

### Consequences
- The project now has a minimal runnable contract for one-sample inference.
- Chemistry utilities can be exercised from a real CLI path without invoking training code.
- The current fragment proposal is intentionally heuristic and not yet counterfactual-optimal, but it keeps the repository moving toward the v3 objective with a reviewable baseline.

### Status
Accepted

---

## [2026-04-14] Add a lightweight RandomForest Oracle and residual-based PPO reward wrapper

### Background
The repository has completed the SFT stage and is moving toward PPO. At that point the project needs a reward path that is both chemically constrained and fast enough to run inside RL rollouts. The existing codebase already has RDKit-backed capped-subgraph validation and deletion, but it still lacked:

- a lightweight local Oracle for fast activity scoring;
- a PPO-facing reward wrapper that combines validity, subgraph, and counterfactual terms;
- a clear decision on whether the semantic reward should score the fragment itself or the residual molecule after deletion.

### Decision
Add:

- `scripts/train_aids_oracle.py` to train a RandomForest classifier on Morgan fingerprints from `data/aids_dataset.csv`;
- `src/rewards/chem_rules.py` as a small structural reward engine for validity and parent-subgraph checks;
- `src/rewards/reward_calculator.py` as the PPO-facing reward wrapper that loads the Oracle bundle and computes `R_valid + R_subgraph + R_counterfactual`.

The counterfactual term is explicitly defined on the residual molecule `x \ g` after deleting the generated fragment, not on the fragment alone. Dummy atoms (`*`) are cleaned through RDKit graph editing before Morgan fingerprint extraction.

### Alternatives considered
1. Score the generated fragment by itself and treat that as the counterfactual term.
2. Put Oracle logic directly inside the PPO training script.
3. Save a custom Python class inside the pickle bundle instead of a plain dictionary.

### Consequences
- PPO reward computation stays aligned with the v3 deletion-based counterfactual objective.
- Oracle scoring remains fast enough for rollout-time use because it relies on RandomForest plus Morgan fingerprints.
- Structural chemistry checks stay modular and reusable instead of being buried in the train loop.
- The saved Oracle bundle is easier to reload across scripts because it stores plain metadata alongside the fitted sklearn model.

### Status
Accepted

---

## [2026-04-13] Add base-model metric tooling and presentation visualization for the SFT stage

### Background
The SFT stage now has stable headline numbers for validity, capping behavior, and token accuracy, but the project still lacked two practical tools: a deterministic way to compute baseline metrics from JSONL inference logs, and a reusable plotting pipeline for lab-meeting summaries.

### Decision
Add a small evaluation utility layer for base-model JSONL logs and extend the SFT visualization module with presentation-ready outputs.

This change introduces:

- `src/eval/base_metrics.py` for RDKit-backed capping/validity statistics
- `src/eval/base_inference.py` for deterministic base-model batch inference over `sft_val.jsonl`
- `scripts/eval_base_metrics.py` as a thin CLI entrypoint for base-model log evaluation
- `scripts/run_infer_base.py` as a thin CLI entrypoint that saves base-model predictions to JSONL
- an upgraded `src/eval/sft_visualization.py` that renders:
  - base-vs-SFT comparison bars
  - a dual-axis simulated training dynamics figure
  - a high-resolution RDKit rendering of a capped fragment
- the updated `scripts/visualize_sft_summary.py` CLI for parameterized figure generation

### Alternatives considered
1. Compute baseline metrics manually in notebooks.
2. Keep visualization code in a single ad hoc script under `scripts/`.
3. Hardcode baseline numbers directly into plotting code without a reusable evaluation helper.

### Consequences
- The repository now has a reproducible path from base-model inference logs to group-meeting figures.
- Evaluation logic remains modular under `src/eval/`, while CLI scripts stay thin and HPC-friendly.
- Presentation plots can be regenerated with different baseline numbers without editing source code.
- The training dynamics figure now uses discrete epoch-level line charts instead of dense smooth curves, which makes the plot easier to explain during presentations.

### Status
Accepted

---

## [2026-04-11] Add balanced capped-fragment SFT data preparation for the HIV dataset

### Background
The rebuilt repository already has RDKit-backed capped-subgraph utilities and thin runtime entrypoints, but it still lacked a concrete path for constructing supervised fine-tuning data from the local HIV CSV. Because the HIV benchmark is heavily label-imbalanced, a naive sample would underexpose positive molecules during SFT.

### Decision
Add a new balanced SFT data-preparation path centered on `scripts/prepare_sft_data.py` and `src/data/sft_preparation.py`.

This path:

- loads `data/raw/AIDS/HIV.csv` with pandas
- filters out invalid parent SMILES using the existing chemistry layer
- keeps all valid positive molecules and fills the remaining target size with sampled negatives
- constructs capped fragment targets by cutting one or two acyclic single bonds with RDKit dummy-atom capping
- validates generated fragments with the shared capped-subgraph checks
- writes minimal `instruction` / `output` JSONL files for ChemLLM SFT

### Alternatives considered
1. Reuse the placeholder `scripts/prepare_data.py` without adding reusable source modules.
2. Sample the HIV dataset according to its natural class distribution.
3. Generate weak targets without validating capped-subgraph correctness against the parent molecule.

### Consequences
- The repository now has an end-to-end path for building balanced SFT supervision data aligned with the capped-fragment objective.
- Positive molecules are intentionally overrepresented relative to the raw dataset so ChemLLM sees enough active-molecule structure during SFT.
- Fragment generation can still fail for some molecules, so the script explicitly reports success rate and observed label ratios after construction.

### Status
Accepted

---

## [2026-04-10] Treat dummy atoms as attachment-point caps in the chemistry layer

### Background
The project now needs chemically valid graph cutting. Plain atom deletion breaks
valence and makes the fragment contract ambiguous whenever the generated
subgraph has open attachment points to the rest of the parent molecule.

### Decision
Adopt dummy atoms (`*`) as the canonical capping representation inside
`src/chem/`.

This means:

- capped fragment SMILES such as `*c1ccccc1` are parsed and sanitized through a
  dummy-aware RDKit path;
- dummy atoms are treated as attachment-point queries when checking whether a
  fragment is a valid parent subgraph;
- capped deletion uses RDKit core replacement semantics so the remainder graph
  is also capped with dummy atoms instead of leaving broken chemistry.

### Alternatives considered
1. Keep using uncapped raw atom deletion and repair broken valence later.
2. Represent cut points with text markers outside SMILES instead of dummy atoms.
3. Treat dummy atoms as ordinary fragment atoms that should also be deleted from
   the parent match.

### Consequences
- The chemistry layer now distinguishes between real fragment atoms and dummy
  attachment points.
- Validation and deletion stay aligned with the v3 counterfactual objective,
  because both fragment extraction and residual construction use the same capped
  semantics.
- Future model outputs can use capped fragment SMILES directly without adding an
  extra post-processing representation.

### Status
Accepted

---

## [2026-04-10] Bootstrap local-only ChemLLM inference with dataset-backed sampling

### Background
The chemistry layer can now validate capped fragment SMILES, but the repository
still needs a real local-model inference path before SFT or RL code is added.
At the same time, the user needs a safe way to confirm that the local AIDS/HIV
CSV file and local ChemLLM checkpoint are usable without any accidental network
requests.

### Decision
Add a local-only ChemLLM inference bootstrap consisting of:

- `scripts/test_assets.py` for local dataset and model asset validation;
- a ChemLLM-specific prompt builder with hard-coded capped-fragment few-shot
  examples;
- a lightweight Hugging Face `ChemLLMGenerator` that always loads with
  `local_files_only=True`;
- dataset-backed `run_infer.py` behavior that samples one real molecule from the
  local AIDS/HIV CSV file when no explicit SMILES is provided.

If the local transformers stack is unavailable at runtime, the CLI is allowed to
fall back to the existing heuristic inference path, but it must surface the
model-side failure explicitly in the structured result.

### Alternatives considered
1. Wait for training code before integrating any real LLM backend.
2. Hardcode one demo SMILES instead of sampling a real local dataset row.
3. Allow `from_pretrained` to use default remote resolution behavior.

### Consequences
- The project now has a true local model inference integration point without
  depending on vLLM or distributed serving.
- Asset validation becomes safer because the repository refuses to reach out to
  Hugging Face when local files are incomplete.
- The inference CLI stays usable in lightweight dev environments because it can
  degrade to heuristic generation while still reporting why the model path did
  not run.

### Status
Accepted

---

## [2026-04-19] Make decoded chemistry PPO use a deletion-based counterfactual teacher oracle

### Background
The decoded chemistry PPO smoke test already proved that a fragment-level
teacher score on `core_fragment_smiles` could be computed, but the semantic
term that actually entered `total` still came from a fixed
`counterfactual_sem=-5.0` missing penalty whenever the deletion-based branch
was unavailable. That meant the PPO loop still was not aligned tightly enough
with the v3 counterfactual objective.

### Decision
Introduce an explicit deletion-based counterfactual teacher scorer that:

- deletes exactly one matched instance of the core fragment from the parent
  molecule;
- scores both the original parent and the residual parent with the teacher
  classifier;
- computes `cf_drop = p_before - p_after` and adds a configurable flip bonus
  when the residual prediction no longer matches the original label;
- uses this deletion-based `counterfactual_sem` as the default semantic term
  that enters the decoded PPO reward and total score.

The earlier fragment-level teacher score is retained only as an auxiliary
diagnostic field (`fragment_teacher_sem`) so we can still inspect whether the
generated fragment itself looks label-associated.

### Alternatives considered
1. Keep using the fragment-level teacher score as the main semantic reward.
2. Continue using the old residual-oracle fallback plus a fixed missing penalty.
3. Collapse fragment-level and counterfactual teacher semantics into one field.

### Consequences
- The decoded PPO path is now explicitly aligned with the repository's
  counterfactual deletion objective instead of a fragment-only semantic proxy.
- Logs can distinguish fragment-level diagnostics from the real
  counterfactual reward that enters PPO.
- Deletion failures, unavailable teacher backends, and disabled counterfactual
  teacher scoring are surfaced explicitly instead of being hidden behind an
  unexplained `-5.0`.

### Status
Accepted
