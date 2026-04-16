# Debug ChemLLM Cache On HPC

This note is for the VS Code -> Git -> HPC workflow used by this repository.
It assumes:

1. you modify the repository locally;
2. you push or sync the repo to the HPC checkout;
3. you then `git pull` on HPC and submit a Slurm job there.

The goal is to prove which `modeling_internlm2.py` the PPO job actually imports
at runtime, and to patch the correct Hugging Face cache file if needed.

## 1. Sync latest repository changes

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
git pull
git rev-parse HEAD
```

## 2. Find all candidate `modeling_internlm2.py` files

Search both the repository checkout and common Hugging Face cache roots.

```bash
find /share/home/u20526/czx/counterfactual-subgraph -name modeling_internlm2.py
find /share/home/u20526/.cache/huggingface -name modeling_internlm2.py
find "${HF_HOME:-/share/home/u20526/.cache/huggingface}" -name modeling_internlm2.py
```

If you also want to search the current conda environment:

```bash
python - <<'PY'
import os
import sys
from pathlib import Path

prefix = Path(sys.prefix)
print("sys.prefix:", prefix)
for path in prefix.rglob("modeling_internlm2.py"):
    print(path)
PY
```

## 3. Inspect the most suspicious cache file

Use the repository tool to print helper status, dangerous usage counts, and
nearby context.

```bash
python tools/check_or_patch_chemllm_cache.py \
  --path /share/home/u20526/.cache/huggingface/modules/transformers_modules/ChemLLM_hyphen_7B_hyphen_Chat/modeling_internlm2.py
```

You can also grep directly:

```bash
grep -n "_has_valid_past_key_values" /share/home/u20526/.cache/huggingface/modules/transformers_modules/ChemLLM_hyphen_7B_hyphen_Chat/modeling_internlm2.py
grep -n "past_key_values\\[0\\]\\[0\\]\\.shape\\[2\\]" /share/home/u20526/.cache/huggingface/modules/transformers_modules/ChemLLM_hyphen_7B_hyphen_Chat/modeling_internlm2.py
```

## 4. Patch the actual cache file if needed

Only patch the file that the runtime job is really importing.

```bash
python tools/check_or_patch_chemllm_cache.py \
  --path /share/home/u20526/.cache/huggingface/modules/transformers_modules/ChemLLM_hyphen_7B_hyphen_Chat/modeling_internlm2.py \
  --patch
```

The tool will:

- create `modeling_internlm2.py.bak` if it does not already exist;
- add `_has_valid_past_key_values`;
- patch the `prepare_inputs_for_generation()` cache branch;
- patch the `forward()` / `past_key_values_length` branch;
- print a unified diff and a post-patch summary.

Important:

- after a successful patch, `past_key_values[0][0].shape[2]` may still appear in
  the file;
- that is acceptable if the access is now inside
  `if _has_valid_past_key_values(past_key_values):`;
- the real success criterion is `unguarded_count=0`;
- `guarded_count>0` is allowed.

## 5. Submit the runtime-path smoke test

The repository now includes a debug Slurm template that prints:

- Python executable path;
- working directory;
- Git commit;
- `HF_HOME`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `PYTHONPATH`;
- the actual module path and source file of the loaded ChemLLM runtime class;
- the source file for `prepare_inputs_for_generation`.

Review and adjust the template if your cluster needs different queue settings:

```bash
sed -n '1,220p' scripts/slurm/debug_check_chemllm_runtime_path.sh
```

Then submit it:

```bash
sbatch scripts/slurm/debug_check_chemllm_runtime_path.sh
```

## 6. Track the job and inspect logs

After submission, record the job id and monitor it:

```bash
squeue -j <jobid>
sacct -j <jobid>
tail -n 100 logs/<jobid>.out
tail -n 100 logs/<jobid>.err
```

If your Slurm setup uses `%j.out` / `%j.err`, the log files usually live under
`logs/`.

## 7. What to confirm in the log

Look for the runtime debug lines emitted by `scripts/train_ppo.py`.

You want to confirm:

1. the actual `python_executable`;
2. the actual `cwd`;
3. the actual `git_commit`;
4. the actual `HF_HOME`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `PYTHONPATH`;
5. the actual `class_module`, `module_file`, and `prepare_inputs_source_file`
   for:
   - `policy_model`
   - `reference_model`
   - `value_model`
   - `ppo_trainer.model_before_patch`
   - `ppo_trainer.model_after_patch`

If the runtime points to an unexpected `modeling_internlm2.py`, patch that file,
not the one you assumed from static inspection.

When checking the patch-tool output, interpret it like this:

1. `has__has_valid_past_key_values=True` means the helper exists.
2. `unguarded_count=0` means the dangerous accesses are no longer exposed.
3. `guarded_count>0` means the accesses still exist, but only inside the new
   helper-protected branch.

## 8. If the cache file keeps getting regenerated

If the log shows that a new dynamic module cache file is recreated after every
pull or job launch, stabilize the cache path before retraining.

Examples:

```bash
export HF_HOME=/share/home/u20526/.cache/huggingface
export TRANSFORMERS_CACHE=/share/home/u20526/.cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/share/home/u20526/.cache/huggingface/hub
```

If needed, clear only the specific stale ChemLLM dynamic-module directory and
re-run the smoke test:

```bash
rm -rf /share/home/u20526/.cache/huggingface/modules/transformers_modules/ChemLLM_hyphen_7B_hyphen_Chat
sbatch scripts/slurm/debug_check_chemllm_runtime_path.sh
```

Use the targeted delete above carefully. Do not remove unrelated Hugging Face
cache directories unless you really intend to rebuild them.
