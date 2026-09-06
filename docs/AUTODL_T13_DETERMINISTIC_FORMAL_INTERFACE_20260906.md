# T13 matched-deterministic formal interface

The user authorized adopting the existing target0/2 matched-deterministic
short-training and independent-reload evidence. The native eager self-control
failure is retained unchanged. This is an execution-contract adoption, not a
diagnostic checkpoint promotion or a claim of full100-epoch trajectory parity.

## Changed interface only

`seal_t13_deterministic_formal.py` seals a fresh spec set against the existing
verified import and publisher. It checks the two targets' three eager controls,
two optimizer updates, component equality, independent reload and original
input/index/mask bindings. Scientific source files must match the passed
ae676fbd checkout byte-for-byte; no model/data/optimizer implementation changes
are made. Large model/checkpoint contents rely on the already sealed receipts;
only bounded JSON and code identities are rechecked.

The existing T13 owner now accepts an explicit `MATCHED_DETERMINISTIC_CONTRACT`
route with paired contract path/SHA. It does not rerun the passed canary. The
original `t13-lazy-repair-20260906/full_start.json` ledger remains the only
formal-start authority, with max1. A new authorization file does not create a
new allowance. Any later recovery requires a legal checkpoint-specific plan.

T13 admission remains384GiB cgroup headroom,100GiB persistent bytes and8192 free
inodes including the4096 compact reservation, with the existing192GiB other-main
reserve and2× measured process peak. Mut's100000 guard is separate and unchanged.

The new child receives `CUBLAS_WORKSPACE_CONFIG=:4096:8` before creation. Before
baseline imports/CUDA initialization, the actual child sets and reads back:
Torch2.7.1+cu118; deterministic algorithms with warn_only=false; cudnn
deterministic=true, benchmark=false, TF32=true; matmulTF32=false.
It writes a PID/UUID/contract-bound runtime receipt. The original diagnostic
did not save numeric thread counts; this omission is disclosed. No new thread
override is introduced, and formal runtime thread counts are recorded.

The formal worker retains the existing train→rules→chemistry/GINE→calibration
freeze→held-out test→export→independent verifier→canonical locator chain. All
current T12/T14/Mut processes, old failed roots, scientific config and matrix
authority are untouched. Current science PID is not itself final cell PASS.

## Interface

The sealer requires explicit `--source-spec-root`, `--evidence-root`,
`--original-authorization`, `--fresh-root`, `--fresh-science-root`,
`--fresh-cache-root`, `--config configs/hpc.yaml` and
`--set inference.fallback_to_heuristic=false`. It emits only a bounded dispatch
receipt, not the full canary arrays. The existing launcher accepts
`T13_DETERMINISTIC_EXECUTION_CONTRACT` and
`T13_DETERMINISTIC_EXECUTION_SHA256` alongside the original lazy authorization.
Corresponding Slurm wrappers are kept AutoDL-only/static-refusal; no HPC GPU
submission is appropriate for this owner.

## Verification

Focused tests cover complete component evidence rather than marker-only PASS,
preservation of native failure, no checkpoint reloading during adoption, exact
TF32 flags, pre-CUDA environment/readback, native environment noninterference,
the original one-start ledger, and T13-specific inode admission. Existing
T13 lazy-guard/import/owner tests are included. No long canary is repeated.
