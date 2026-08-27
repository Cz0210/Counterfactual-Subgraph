# AutoDL TasteMolNet T5 Clean-Policy Initializer

## Scope

This route implements only `T5_CLEAN_POLICY_READY`. It converts a proven
generic ChemLLM base into a fresh zero-optimizer-step LoRA and publishes a
private, hash-closed initializer. It does not run SFT or PPO, open a Taste row,
score with an oracle, register a matrix cell, or start a controller.

The tracked configuration is intentionally release-disabled:

```text
tracked_release_enabled=false
tracked_release_state=RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT
```

The public builder currently stops on that tracked state before it opens a
release authority, source model, or output. A later reviewed release must add
a physical execution-receipt contract and final pins before changing this
state. The successor producer implementation is locally testable, but an
external JSON file cannot override the tracked release-disabled gate.

That later authority and its expected raw SHA-256 must bind all of the
following:

- policy-v2 raw and canonical identities plus its typed receipt;
- one clean immutable implementation commit/tree;
- one generic, dataset-independent ChemLLM base inventory;
- the common frozen three-class GINE checkpoint, its full/stat/SHA inventories,
  feature schema, label map, config, and validation-only temperature identity;
- the typed, descriptor-held T3 and T4 roots/gates, including T4's exact T3
  gate hash and their bidirectional common-GINE bindings;
- a declaration-only controller task and physical GPU-2 UUID/visibility
  identity; this is explicitly not GPU-lock ownership;
- a separately reviewed physical execution receipt before release is enabled;
- Sweet source label `1`, destination labels `{0, 2}`, and no RF;
- AutoDL-only execution and no data redistribution.

Missing or changed pins, the current declaration-only controller state, or the
tracked disabled state return release-disabled status before the final output
root is created. The historical BACE provenance schema, manifest, marker, and
dataset-specific adapter are not accepted.

## Split and oracle boundary

The implemented path is dataset-independent and therefore records both facts
without conflating them:

```text
initializer_data_split_used=none
taste_split_access_max=train_only
taste_splits_loaded=[]
optimizer_step_count=0
rf_reference_count=0
gnn_reward_used=false
validation_loaded=false
calibration_loaded=false
test_loaded=false
```

`taste_split_access_max=train_only` is a ceiling for future initializer work;
it is not evidence that the current zero-step route read the train split. A
train-only oracle-neutral SFT fallback is deliberately not implemented. It
requires a new explicit authority and review rather than a command-line switch.

## Private output contract

The output must be an absent UTC-timestamp child of:

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/clean-policy-initializer/
```

The parent and final root are physical owner-held mode-`0700` directories.
Files are mode `0600`, symlinks/special files/multiple hardlinks are rejected,
and final publication uses Linux `renameat2(RENAME_NOREPLACE)`. The exact
top-level inventory is:

```text
adapter/
policy_provenance.json
manifest.json
state.json
gate.json
input_hashes.json
output_hashes.json
PASS
```

The terminal producer marker is written last and is exactly:

```text
[TASTE_CLEAN_POLICY_INITIALIZER_PASS]
```

`policy_initializer_hash` binds the complete adapter inventory, source-model
inventory, `initializer_data_split_used=none`, and zero optimizer steps.
`reference_model_hash` and `source_model_inventory_sha256` identify the generic
base. `reference_policy_hash` is a distinct canonical hash over that base
inventory, the complete adapter inventory, and the actual serialized parameter
digest; it must not equal the bare-base hash. The adapter is produced through
the same ChemLLM loader and PEFT LoRA targets used by the stable PPO runtime
(`wqkv`, `wo`, `w1`, `w2`, `w3`; rank 8, alpha 16, dropout 0.05). The producer
parses the real safetensors payload, rejects non-LoRA/missing-pair/rank/shape/
dtype/non-finite states, proves every LoRA B matrix is zero, and reloads it
through a fresh base with `PeftModel.from_pretrained`. Downstream code must use
the Taste validator, never the BACE provenance validator.

## Read-only T6 consumer API

Use the stable import surface:

```python
from src.train.tastemolnet_clean_policy import (
    hold_clean_policy_load_authority,
    hold_clean_policy_output,
    hold_source_model_for_clean_policy,
    validate_clean_policy_output,
    validate_source_model_for_clean_policy,
)
```

`validate_clean_policy_output(root)` is a one-shot read-only validation.
`hold_clean_policy_output(root)` retains the physical root descriptor and
allows repeated `.revalidate()` calls while T6 holds its input authority. Its
evidence contains no mutable mtime field. The exact return keys are:

```text
schema_version
status
stage
output_root
adapter_dir
source_model_dir
source_model_path
policy_initializer_hash
reference_model_hash
reference_policy_hash
source_model_inventory_sha256
adapter_sha256
manifest_sha256
gate_sha256
t5_gate_sha256
pass_sha256
t5_pass_sha256
input_hashes_sha256
output_hashes_sha256
output_inventory_sha256
root_inventory_sha256
t5_output_inventory_sha256
frozen_oracle_identity
frozen_oracle_identity_sha256
gpu_identity
marker
```

The held validator reads every terminal leaf relative to the retained root,
revalidates the complete adapter and evidence closure, and checks the named
root inode before and after each pass. It also retains a full physical stat
inventory including inode and ctime, so a rename/copy/in-place mutation remains
detectable after an attacker restores the original bytes and names.

T6 must use the combined loading authority, not extract the lexical
`adapter_dir` or `source_model_dir` from one-shot evidence:

```python
with hold_clean_policy_load_authority(t5_root) as held:
    token = held.load_token()
    # token.source_model_load_path -> tokenizer/base/value loaders
    # token.adapter_load_path -> policy/reference PeftModel.from_pretrained
    held.revalidate_load_token(token)  # after tokenizer/base/value loads
    held.verify_loaded_policy(policy, token=token, role="policy")
    held.verify_loaded_policy(reference, token=token, role="reference")
```

On Linux the two load paths are retained `/proc/self/fd/N` directory paths and
remain valid only while the combined context is alive. The exact token keys
are `schema_version`, `output_root`, `source_model_load_path`,
`adapter_load_path`, `source_model_inventory_sha256`,
`adapter_inventory_sha256`, `adapter_parameter_sha256`,
`reference_policy_hash`, `t5_output_inventory_sha256`, and
`frozen_oracle_identity_sha256`. The adapter verifier hashes the actual loaded
PEFT state and checks exact key/rank/shape/dtype/finite/zero-step equality.

The lower-level `hold_source_model_for_clean_policy(...)` and one-shot
`validate_source_model_for_clean_policy(...)` helpers remain available for
bounded inspection. They are not a substitute for the combined T6 loader
authority.

The frozen oracle identity also contains the complete receipt-only T2 adoption
binding. T5 reopens the fresh exact-five-file root with the reviewed gate,
receipt, and source-evidence SHA-256 values, verifies its canonical hash DAG,
physical publication binding, fixed source identities, and formal 19-file GINE
inventory, and keeps that holder alive through terminal publication. It never
reopens the historical failed controller or training/execution roots. T3 and
T4 must already agree on this same binding; any mismatch blocks T5. This
provenance closure does not change the release-disabled state.

## AutoDL entrypoint (release-disabled)

The wrapper is foreground-only and does not acquire GPU locks or launch itself
with nohup/tmux. The command shape reserved for a future receipt-backed release
is:

```bash
TASTEMOLNET_T5_RELEASE_AUTHORITY=/absolute/release_authority.json \
TASTEMOLNET_T5_RELEASE_AUTHORITY_SHA256=<sha256> \
TASTEMOLNET_POLICY_RECEIPT=/absolute/policy_receipt.json \
TASTEMOLNET_CHEMLLM_BASE=/absolute/ChemLLM-7B-Chat \
CUDA_VISIBLE_DEVICES=2 \
OUTPUT_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/clean-policy-initializer/YYYYMMDDTHHMMSSZ \
scripts/autodl/run_tastemolnet_clean_policy_initializer.sh
```

The paired Slurm script is static CLI parity and exits `78` before invoking
the builder because this Taste route is AutoDL-only. The tracked AutoDL command
also exits release-disabled through the Python gate before any authority or
model load; a declaration-only controller record cannot release it.

No production release authority, AutoDL build, production model load, GPU
process, controller mutation, deployment, or scientific experiment was
performed while adding this implementation. A tiny local CPU model and real
PEFT/safetensors round trip were used only as a bounded format test.
