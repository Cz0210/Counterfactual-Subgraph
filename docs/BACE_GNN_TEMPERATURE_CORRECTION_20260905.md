# BACE seed7 first validation-temperature correction

The project owner authorized a **first** validation-only scalar-temperature fit
for GIN, GCN, GATv2 and GatedGCN+. The original GINE fit, five weight files,
original package, 66 candidate identities, split files and historical results
remain immutable. This repairs missing `fit_on_validation`, not model training.

The actual saved validation split alias is `val`, with 187 ordered examples.
The saved raw logits are tied to the published model and the training terminal's
`best_state`; their original probability serialization uses float32 torch
softmax. The fitter retains the historical float64 CE/log-T/LBFGS contract.
No calibration or test data can enter fitting. A numerically fitted T=1 is valid;
the original `not_fit` placeholders remain scientifically blocked.

Stages (all real CLI actions of
`scripts/hpc/gnn/repair_bace_seed7_temperature_contract.py`):

1. `plan`: seal authorization, original SHA inventory and repair scope.
2. `fit`: four scalar optimizations and additive classifier overlays.
3. `reconcile-calibration`: use saved parent logits and frozen inference for
   missing residual logits. Original per-match raw distances are independently
   rebound to graph, mapping, original checkpoint, MolCLR/schema/solver inputs.
4. `freeze`: replay all ten original global calibration selectors.
5. `reconcile-test`, then `finish`: only after the fresh freeze, correct saved
   test probabilities and native/common metrics. The test was previously
   evaluated; this is explicitly not a first-unseen-test claim.
6. `verify-package`: independent scientific replay and corrective archive.

There is no OT solver fallback in this correction. An unproven distance or a
new test rule missing from the old evaluated union fails `CACHE_PROVENANCE_GAP`.
Successful parent units are hash-bound and reusable after failed-job recovery.
The old package and blocked audit are never rewritten as PASS.

The matched-calibration launch gate accepts only the new independent
`GNN_CORE_SEED7_CORRECTED_PASS` archive. L0 then runs the existing 472-fragment,
386×8 BRICS input through the same main calibrated GINE on HPC CPU. L1, L2 and
L3 remain resource-gated, ordered 7B off-the-shelf → original 300-update PPO LoRA
matched generation → 2B off-the-shelf. Neither a main-matrix count nor secondary
seeds is a prerequisite. GPU borrowing is disabled; no primary T13 reservation
may be removed. These ablation outputs cannot write the 4×4 matrix.

Slurm scripts under `scripts/slurm/` intentionally use the authorized `intel`
CPU partition instead of the generic A800 training defaults. Bootstrap suspends
`nounset` only while sourcing the existing environment. No environment upgrade,
new scheduler platform or active-worktree modification is involved.
