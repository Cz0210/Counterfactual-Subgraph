# BACE LLM resource-only dispatch binding

This is a narrow metadata helper for the existing AutoDL owner, not another
controller, lock, registry, preparation pass or GPU authorization. It supports
only the already sealed L1→L2→L3 science commit
`3b2605d681e2ef6116f3cacc662eead2f65dd28a`.

## Preserved scientific authority

The original dispatch, readiness, three task specs, model/adapter descriptors,
generation command and all input/output paths remain unchanged. Only small
JSON descriptors are reopened (2MiB maximum each); model weights, datasets,
candidate pools and accepted GNN archives are not scanned or loaded.

The real `bace_readiness.prepare` task descriptor has exactly four fields:
`path`, `sha256`, `generator_state`, `downstream_state`. Preserve all four
verbatim and bind the two known metadata values to the SHA-verified task file.
Only the verified path/SHA pair is passed to the small-file reader; additional
unknown fields, omitted metadata or metadata drift fail closed. A path/SHA-only
unit fixture is not a valid substitute for this actual3b2605d6 source schema.

`execution_commit` in a fresh dispatch denotes **owner_driver_commit**, not a
claim that old scientific specs were prepared with the new driver. The additive
`resource_only_overlay` binds the original dispatch bytes, old science commit,
new owner root/commit and fresh stage-file policy. Overlay-on-overlay is refused.

The fresh resource config must equal the original config plus exactly one
`stage_file_policy: {path, sha256}` descriptor. Existing memory, storage, READY
sources, registry, GPU locks and the historical100000-inode field stay unchanged.
The independently implemented stage-resource adapter decides how an authorized
stage-specific policy applies; this metadata helper does not grant admission.

For every downstream command the only allowed transform is:

1. Replace argv3 with the new driver's
   `scripts/ablations/llm/run_bace_common_downstream.py`.
2. Append exactly `--stage-file-policy PATH --stage-file-policy-sha256 SHA
   --compact-node-cache` (five argv elements after substitution).

Everything else, including Python executable, isolation flags, original task
descriptors, source files, CPU threads, device, selector, evaluation and output
arguments, must compare exactly. Compact cache is opt-in future storage only;
it cannot relabel changed encoder/OT/selector semantics as resource-only.
The new evaluator's lossless storage regression remains a separate prerequisite.

## Integration in the existing owner

The existing `scripts/autodl/gpu_lock.py` should replace its single-driver
commit/entrypoint assertion with:

```python
from src.ablations.llm.stage_dispatch_binding import validate_dispatch_runtime
binding = validate_dispatch_runtime(spec, current_commit, PROJECT_ROOT)
```

The caller must still verify the dispatch file descriptor, hold the original
exclusive GPU/project-slot FDs, honor primary reservations/READY tasks, sample
fresh resources and enforce the real GPU smoke/resume gate. Legacy same-driver
dispatch is supported without an overlay. This helper does not modify the
owner or gpu_lock modules, signal owner325928, or create a new owner itself.

## Actual CLI

`scripts/autodl/rebind_llm_stage_resource_policy.py --help` lists only `seal`
and `validate`; there is intentionally no activate/kill/prepare operation.

Seal requires `--original-dispatch`, `--original-dispatch-sha256`,
`--resource-config`, `--resource-config-sha256` and fresh `--output`.
Validate requires `--dispatch-spec` and `--dispatch-spec-sha256`.
Both use `--config configs/hpc.yaml` and optionally
`--set inference.fallback_to_heuristic=false`. The exact current owner commit
and root come from the executing immutable checkout, not user-provided text.

Sealing performs full structural validation before writing one temporary file,
fsyncs it, atomically publishes without replacing an existing destination, and
fsyncs the parent. It returns `SEALED_WAITING_RESOURCE`, `science_started=false`,
`owner_started=false`, `resource_admission_evaluated=false`.

No deployed output path or future PID is invented in this development note.
The integration agent supplies the actual new immutable checkout, policy/config
descriptors and destination after deployment; ordinary CLI help does not claim
that sealing or a GPU run has happened.

The paired `scripts/slurm/rebind_llm_stage_resource_policy.sh` is CPU-only by
this task's explicit metadata-only override of the repository GPU defaults.
It retains safe shell bootstrap, `smiles_pip118`, the documented HPC checkout,
PYTHONPATH and exact CLI options; it requests no GPU and runs no model probe.

## Focused validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/ablations/test_llm_stage_dispatch_binding.py
python -m compileall -q src/ablations/llm/stage_dispatch_binding.py \
  scripts/autodl/rebind_llm_stage_resource_policy.py
bash -n scripts/slurm/rebind_llm_stage_resource_policy.sh
git diff --check
```

Tiny CPU fixtures cover strict unchanged science fields, exact fixed downstream
tail, config threshold/registry preservation, old/source SHA drift, one-level
overlay, no overwrite, owner/science role split, legacy behavior and real CLI.
Passing these tests does not establish GPU admission or scientific completion.
