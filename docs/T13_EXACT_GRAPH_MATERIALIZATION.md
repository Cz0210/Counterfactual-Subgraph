# T13 future-only graph materialization patch

Baseline: `c0eb892dd13ef05a5891c4acf1c5f4fef3966f67`.
This changes only the frozen-GINE bridge's hard CPU decode transfer and ordered
soft tensor gathering. It changes no model, source/target, candidate, chemistry
gate, sampler, budget, loss, optimizer, validation frequency, or scientific data.

- Hard-only class argmax remains on the original device; its finite index lists
  and adjacency are transferred to CPU in bulk. The node order, lower-triangle
  bond indexing, asymmetric-adjacency failure order, RDKit behavior and fallback
  features remain unchanged.
- Directed soft edges keep the old source-major/destination-major order. Gather
  does not detach either adjacency or edge scores; duplicate directed uses
  retain their contribution to the underlying decoder gradient.
- There is no chemistry cache in this patch. Repeated evolving graph outputs
  were not proven common enough to justify a content cache, and a parent-ID
  cache would be invalid across epochs.

The test/benchmark loads the old code directly from the pinned Git object; it
never reads campaign weights or rewrites historical source. Synthetic fixtures
include singleton, hole padding, ring and deliberately invalid chemistry. It
compares outputs/audits/loss/gradients/two Adam updates/scheduler/RNG, and a
checkpoint reload. Exact equality is required; no tolerance has been widened.
The benchmark synchronizes CUDA boundaries and records wall time, allocated
GPU peak and process high-water RSS. Process RSS is cumulative, not an isolated
per-arm RSS measurement. CPU results do not prove the CUDA contract.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I -B scripts/benchmarks/benchmark_t13_bridge_materialization.py --config configs/hpc.yaml --device cpu --output /tmp/t13-bridge-cpu-fresh.json
```

The paired Slurm wrapper is `scripts/slurm/benchmark_t13_bridge_materialization.sh`.
For AutoDL CUDA use an already available exclusive lease, full GPU UUID mapped
to `cuda:0`, and child environment `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The CLI
requires Torch 2.7.1+cu118, deterministic algorithms (not warn-only), cuDNN
deterministic true/benchmark false, matmul TF32 false and cuDNN TF32 true.

**Not a handover receipt:** this bounded bridge fixture does not establish the
real official generator/full-batch or epoch-boundary checkpoint-resume parity.
The active T13 immutable process and its one-fresh-start ledger are unchanged.
Do not activate an optimized resume until those remaining gates and actual
resource/owner handover are satisfied. Never repeat a fresh 100-epoch run.
