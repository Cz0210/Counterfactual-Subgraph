# Label=1 SFT/PPO Ablation Design

This ablation isolates the label=1 contribution of SFT and PPO under the same parent set, oracle, generation count, projection logic, audit logic, and coverage-heavy selector settings.

## Methods

- A = Base ChemLLM, with no SFT adapter and no PPO adapter.
- B = SFT-only, using SFT v3 `checkpoint-500`.
- C = SFT+PPO, using the stable300 PPO checkpoint.

## Comparisons

- B vs A validates whether SFT improves valid, parseable, connected, parent-substructure fragment generation.
- C vs B validates whether PPO further improves counterfactuality, especially `cf_drop`, `cf_flip`, and selector-level class coverage / selected fragment strength.

## Scope

The main ablation intentionally excludes the high-temperature merged pool. The merged pool is useful for the final full system, but including it here would mix sampling-diversity effects into the SFT/PPO comparison.
