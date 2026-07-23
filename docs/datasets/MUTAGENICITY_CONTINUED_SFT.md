# Mutagenicity Continued SFT

## 1. AIDS/HIV SFT-v3 audit

The existing AIDS/HIV SFT-v3 training entrypoint is
`scripts/train_sft.py`, invoked by `scripts/slurm/train_sft_v3.sh`.

Its training contract is:

- local base model: `pretrained_models/ChemLLM-7B-Chat`;
- 4-bit NF4 QLoRA, double quantization, bfloat16 compute;
- LoRA rank 8, alpha 16, dropout 0.05;
- target modules: `wqkv`, `wo`, `w1`, `w2`, `w3`;
- per-device train/eval batch size 4;
- gradient accumulation 4, effective single-GPU batch size 16;
- learning rate `2e-4`;
- maximum steps 500;
- cosine scheduler and warmup ratio 0.03;
- bf16 enabled, fp16 disabled;
- eval/save every 100 steps and at most three checkpoints;
- paged AdamW 8-bit and gradient checkpointing;
- seed 7.

The input loader accepts JSONL and normalizes `prompt/completion`,
`instruction/output`, or legacy `prompt/response` aliases through
`src/data/sft_column_compat.py`.

The historical trainer passes a prompt/completion dataset directly to TRL
`SFTTrainer`. Prompt masking and maximum sequence length are not explicitly
configured in repository code, so those details depend on the installed TRL
version. The `checkpoint-500` name comes from `max_steps=500` plus the
100-step checkpoint schedule; it is a PEFT LoRA adapter checkpoint, not a
standalone ChemLLM model.

The verified initialization adapter is:

```text
outputs/hpc/sft_checkpoints/
  sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/
  checkpoint-500
```

The existing A800 wrapper requests one `gpu:a800:1`, seven CPUs, and uses the
`A800` partition. Mutagenicity retains that allocation and uses 64 GB host
memory.

## 2. Continued-training semantics

Mutagenicity continued SFT loads the same quantized ChemLLM base model, then
attaches the AIDS `checkpoint-500` adapter with
`PeftModel.from_pretrained(..., is_trainable=True)`. It starts a new optimizer,
scheduler, and Trainer global step. It does not call
`resume_from_checkpoint`, because the objective is dataset continuation from
adapter weights rather than resuming the completed AIDS optimizer state.

The Mutagenicity entrypoint reuses `build_tokenizer()` and
`build_quantized_model()` from `scripts/train_sft.py`. It uses the standard
Hugging Face causal-LM Trainer with explicit labels so masking does not depend
on an implicit TRL default:

```text
input_ids = BOS + tokenize(prompt) + tokenize(completion) + EOS
labels    = -100 over BOS/prompt, token IDs over completion/EOS
```

Right truncation is audited at a fixed maximum length of 1024. A completion
that loses every content token is a hard pre-training error. Padding tokens are
masked with `-100`.

## 3. Data contract

Preferred data root:

```text
outputs/hpc/mutagenicity/final/sft_ppo_data_v1
```

Fallback root:

```text
outputs/hpc/mutagenicity/sft_ppo_data_v1
```

Only `mutagenicity_sft_train.csv` and `mutagenicity_sft_val.csv` are loaded.
The fixed full counts are 1,317 train and 250 validation rows. The loader
rejects calibration/test rows, source/teacher inconsistency, duplicate or
cross-split molecule IDs, canonical SMILES overlap, scaffold overlap, empty
prompts, and empty completions.

Smoke sampling validates the complete files first and then uses the existing
scaffold-plus-size round-robin sampler. It never takes the first N file rows.

## 4. Runs

Smoke:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
sbatch scripts/slurm/train_mutagenicity_sft_smoke.sh
```

The smoke run selects at most 64 train and 32 validation parents, trains for
three steps, and evaluates/saves every step.

Full:

```bash
sbatch scripts/slurm/train_mutagenicity_sft_full.sh
```

The full run uses all 1,317/250 examples and the AIDS-v3 500-step training
hyperparameters. Validation loss selects the best checkpoint. Calibration and
test are never loaded.

Standalone checkpoint evaluation:

```bash
CHECKPOINT=outputs/hpc/mutagenicity/sft_continued_v1/checkpoint-500 \
OUTPUT_DIR=outputs/hpc/mutagenicity/sft_continued_v1/eval_checkpoint_500 \
sbatch scripts/slurm/evaluate_mutagenicity_sft_checkpoint.sh
```

## 5. Artifacts

Training writes:

```text
resolved_config.json
dataset_manifest.json
tokenization_audit.json
train_metrics.json
eval_metrics.json
trainer_state.json
training_coverage.json
checkpoint-*/
best_checkpoint.json
checkpoint_manifest.json
generation_samples.csv
training_report.md
_RUN_COMPLETE.json
```

The coverage audit records examples seen, unique parents seen, effective batch
size, equivalent epochs, and global step. A completed full run requires at
least 99% unique-train-parent coverage.
