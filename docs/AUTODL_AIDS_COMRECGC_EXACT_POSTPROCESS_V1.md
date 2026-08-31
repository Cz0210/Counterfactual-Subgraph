# AIDS ComRecGC exact-complete postprocessing v1

## Purpose

The 91,916,686-row AIDS exact DBSCAN partition is already complete. This
dataset-specific route consumes that result from a fresh postprocess root and
must never rerun DBSCAN or regenerate the pair store. It runs the remaining
streamed all-core component summary, strict centroid-radius coverage, stable
official greedy selection, chemistry/lineage validation, unified WNode
evaluation, Figure 3/4 and Table 2 CSV export, gate, and standardized freeze.

The existing controller manifest and typed exact-stage receipt remain the
authority. The runner reopens the controller-bound terminal and validates the
source DBSCAN manifest, scientific identity, vector SHA/stat identity, exact
component closure, all-core/no-noise state, and sklearn-float64 semantics.

## Contract

- Exact receipt, DBSCAN manifest, physical pair store, and theta-close view are
  adopted read-only outside the fresh output root.
- Terminal DBSCAN reopen validates the complete closure and returns before
  creating a lock, checkpoint, or output file.
- The fresh root contains
  `common_recourse/external_memory/dbscan_adoption/run_manifest.json`; it must
  not contain `common_recourse/external_memory/dbscan/`.
- The adoption records `dbscan_recomputed=false` and
  `pair_store_recomputed=false` with every source hash and contract.
- CUDA is hidden and postprocessing is capped at eight CPU workers.
- Resume is permitted only for the same root and frozen continuation contract.
  PASS requires the common-recourse closure and every downstream stage.

## AutoDL command

Run from a clean immutable checkout with a fresh output and an external
heartbeat path:

```bash
cd /root/autodl-tmp/worktrees/<immutable-fast16-commit>
CUDA_VISIBLE_DEVICES= DEVICE=cpu GPU_REQUIRED=0 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 AIDS_POSTPROCESS_MAX_WORKERS=8 \
/root/miniconda3/envs/smiles_pip118/bin/python \
  scripts/autodl/run_aids_comrecgc_exact_postprocess_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --controller-manifest /autodl-fs/data/counterfactual-subgraph-runtime/control/<exact-controller>.manifest.json \
  --exact-receipt /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/repairs/<exact-root>/science/common_recourse/external_memory/exact_recovery_receipt.json \
  --output-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/repairs/aids_comrecgc_postprocess_v1_<UTC> \
  --heartbeat-path /autodl-fs/data/counterfactual-subgraph-runtime/control/aids_comrecgc_postprocess_v1_<UTC>.heartbeat.json \
  --max-workers 8
```

Add `--resume` only after an interrupted attempt with unchanged inputs. Matrix
refresh remains a separate post-PASS operation; this science runner cannot
claim a matrix cell itself.

The paired Slurm entrypoint is
`scripts/slurm/run_aids_comrecgc_exact_postprocess_v1.sh`. It follows the
repository A800 allocation baseline but hides CUDA, so the science is CPU-only.

