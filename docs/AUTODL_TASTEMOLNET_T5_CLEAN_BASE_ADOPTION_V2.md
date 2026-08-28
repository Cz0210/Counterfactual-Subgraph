# AutoDL TasteMolNet T5 Clean-Base Adoption v2

## Scope

This route closes the current campaign's T5 predecessor by adopting one
unchanged, generic `ChemLLM-7B-Chat` Hugging Face model tree. It is not SFT and
does not create a LoRA. The authoritative semantic fields are:

```text
semantic_state=ADOPTED_CLEAN_GENERIC_BASE
training_performed=false
optimizer_steps=0
taste_splits_loaded=[]
rf_reference_count=0
gnn_reward_used=false
matrix_method_cell=false
source_weights_copied=false
```

The verifier-only operational marker is `[TASTE_T5_CLEAN_SFT_PASS]`. That
historical stage label does not change the structured no-training semantics.

## Source gate

`inspect_clean_chemllm_base` streams and hashes every physical single-link file
without loading model tensors. It additionally proves:

- `config.json` is one InternLM/InternLM2 causal-LM configuration;
- `tokenizer_config.json` identifies an InternLM tokenizer and exactly one of
  `tokenizer.model` or `tokenizer.json` is present;
- `model.safetensors.index.json` references all and only the model shards;
- every shard has a bounded valid safetensors JSON header;
- the index tensor map equals the union of shard-header tensor maps;
- tensor byte intervals close every shard payload and their total equals the
  index `metadata.total_size`;
- there are no adapter, PEFT, or LoRA files/metadata;
- there are no TasteMolNet, BACE, AIDS/HIV, Mutagenicity, RF-oracle, or dataset
  payload paths.

The output evidence includes every relative filename, byte count, SHA-256, and
the canonical full-tree inventory SHA-256. The 15GB source tree is never copied.

## Managed-v2 boundary

The worker writes only these scientific candidate files:

```text
artifacts/source_inventory.json
artifacts/clean_base_adoption_candidate.json
```

The common managed worker adds `.generation_token.json`,
`raw_evidence.json`, `worker_exit.json`, and `SEALED.json`. It cannot write a
gate, verification, adoption receipt, or PASS.

A separate verifier process:

1. opens the exact SEALED UUID generation;
2. holds the external source root by descriptor;
3. independently rehashes and repeats all Hugging Face checks;
4. checks exact controller/attempt/config/Git/input hashes and EXITED process
   lineage;
5. requires byte-for-byte equality between worker and verifier source evidence;
6. revalidates source inode/ctime inventory and the complete SEALED tree;
7. writes verifier evidence, managed gate, and PASS, then publishes the whole
   receipt directory by atomic no-replace rename.

No signal is sent on any anomaly. Failures remain non-terminal evidence for
manual review, with `AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0` mandatory.

## AutoDL execution

Run only from the clean immutable execution checkout. Do not use HPC and do not
reserve a GPU for this CPU/read-only adoption.

Set the common environment:

```bash
export PROJECT_ROOT=/root/autodl-tmp/worktrees/<clean-t5-execution-tree>
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
export TASTE_T5_SOURCE_MODEL=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/pretrained_models/ChemLLM-7B-Chat
export TASTE_EXECUTION_COMMIT=<exact-clean-commit>
export TASTE_CONTROLLER_ID=<fresh-main-v2-controller-id>
```

First perform one read-only inventory and retain its JSON output:

```bash
TASTE_T5_ACTION=inspect \
scripts/autodl/run_tastemolnet_t5_clean_base_adoption_v2.sh
```

Set `TASTE_T5_SOURCE_INVENTORY_SHA256` to the printed
`source_model_inventory_sha256`. Then launch the managed worker:

```bash
export TASTE_T5_STAGE_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-main-v2/stages/T5_CLEAN_BASE_ADOPTION
export TASTE_T5_SOURCE_INVENTORY_SHA256=<reviewed-full-tree-sha256>
TASTE_T5_ACTION=worker \
scripts/autodl/run_tastemolnet_t5_clean_base_adoption_v2.sh
```

The last JSON line is only a SEALED receipt. Extract its `attempt_id`,
`generation_token`, and `staging_path`; do not call it PASS. Choose one absent
terminal path below the clean-policy output parent and invoke the independent
verifier:

```bash
export TASTE_T5_SEALED_ROOT=<exact-staging-path>
export TASTE_T5_ATTEMPT_ID=<attempt-uuid>
export TASTE_T5_GENERATION_TOKEN=<generation-uuid>
export TASTE_T5_FINAL_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/clean-policy/adopted-clean-base-<UTC>-<attempt-prefix>
TASTE_T5_ACTION=verify \
scripts/autodl/run_tastemolnet_t5_clean_base_adoption_v2.sh
```

Only the verifier may print `[TASTE_T5_CLEAN_SFT_PASS]`. Preserve the final
path, attempt UUID, generation UUID, source inventory SHA-256, verification
SHA-256, gate SHA-256, and publication mode in the campaign handoff.

## Slurm parity

The paired files
`scripts/slurm/tastemolnet_t5_clean_base_worker_v2.sh` and
`scripts/slurm/tastemolnet_t5_clean_base_verifier_v2.sh` include the repository
HPC environment and exact `--config configs/hpc.yaml` CLI parity but
intentionally exit with `REFUSING_HPC_EXECUTION`. TasteMolNet remains
AutoDL-only.
