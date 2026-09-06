# T13 component-exact diagnosis (2026-09-06)

The previous bounded canary completed data/index/mask checks but failed after
the first lazy update. Its combined exception did not preserve loss, gradient,
optimizer, model or RNG differences. This diagnosis closes that observability
gap; it is not permission to relax equality or change formal numerical policy.

## Unchanged science

Use the same adopted train-only mining, 3823-parent source cohort, seed7,
targets0/2, original batch500/max5 batches per optimizer update, decoder,
Adam/StepLR and loss. No mining, model training for the main result, external
split, OT, existing-result write, or new full-start claim occurs here.

Each numeric profile performs at most six updates: eager reference epochs0/1,
two further independent eager epoch0 controls, then lazy epochs0/1 with a
checkpoint reopen between them. Every arm restores the identical initial
model, optimizer/scheduler configuration and Python/NumPy/CPU/CUDA RNG.
Three eager first updates complete even if the second differs. A fatal runtime
operator error stops the profile and is retained as a typed diagnostic failure.

## Evidence

Per target, `training_canary/component_diagnostics.json` contains exact component
differences and `first_difference`; `component_evidence.pt` stores the bounded
raw tensors needed to independently reproduce those differences. These two
files are replaced atomically, not multiplied by the 1.27-million sample count.
No full augmented batch/candidate universe is serialized. Records include:

- initial model/optimizer/scheduler/RNG/index bindings;
- rule tensors, batch SHA, RNG before/after each forward, and four raw losses;
- named gradients before `zero_grad`, model/optimizer/scheduler/RNG after update;
- exact changed leaves, non-finite counts and absolute deltas for diagnosis only.

There is no `allclose`, tolerance, warning-only acceptance or silent fallback.
The existing memory samples and standalone checkpoint reopen remain in force.

## Native first, deterministic second

The existing owner command is unchanged. The CLI now explicitly accepts
`--diagnostic-profile native|deterministic`, default `native`. Only after three
native eager controls demonstrate non-repeatability does the CLI run one fresh
child process in `matched-deterministic/`, still inside the existing owner lease
and process-tree memory guard. The child fixes `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before importing Torch and enables deterministic algorithms with `warn_only=False`.
TF32 and dtype are not changed. Unsupported deterministic operators are failures.

Native failure is preserved even if this matched diagnostic passes. The latter
uses state `T13_MATCHED_DETERMINISTIC_DIAGNOSTIC_PASS`, never the native PASS state
accepted by the one-shot full guard. Formal promotion remains blocked until the
original scientific contract is actually satisfied or a further explicit
numerical-policy decision is made. No second full start is introduced.

## Commands and resource boundary

Launch only through the existing `run_t13_from_hpc_owner_v1.py` owner with its
authorization and reserved GPU UUID. The underlying canary CLI remains:

```bash
python -I -B scripts/autodl/canary_t13_indexed_dataset.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --output-root <fresh-root> --official-root <pinned-root> \
  --train-csv <unchanged-train.csv> --gnn-checkpoint <unchanged-T3-root> \
  --gspan-adoption-proof <existing-import/adoption_proof.json> \
  --device cuda:0 --targets 0,2 --diagnostic-profile native
```

Do not submit the paired Slurm script: it deliberately refuses HPC execution.
The original 384GiB headroom/192GiB protected-main reserve/96GiB canary RSS gates
apply. Reserve compact diagnostic files within the T13 inode budget and account
for other outstanding reservations. Mut's separate 100000-inode admission is
not silently substituted for T13's own gate. Do not use the nearly-full local
AutoDL NVMe for a dataset/model duplicate.

Focused tests include an explicitly injected, CPU-only gradient mutation that
keeps the first forward loss identical. This fixture proves failure localization
and retention, **not** the cause of the real GPU failure. The real cause must be
read from the production component evidence after its short admitted diagnosis.
