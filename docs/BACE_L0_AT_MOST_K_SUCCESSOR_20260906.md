# BACE L0 at-most-K evaluation successor (2026-09-06)

The project owner authorizes **at most K** for all four matched LLM variants.
K_MAX remains20 and Table2 requests K10. The effective prefix is
min(requested K, available eligible unique rules,20); larger requested prefixes
are a plateau, not duplicated rules. The same main selector objective, prefix
weights, tie-breaks, insertion/swap kernels and threshold grid remain frozen.
Only the LLM adapter changes; GNN and main-table protocols do not.

L0 source Slurm2560839 remains FAILED and immutable. Its386×8 scored train
attempts,386 committed parent records and15 eligible canonical rules are
adopted through a fresh `protocol_overlay.json`. The472-fragment vocabulary
and original portable task/reference bytes are retained. The adoption reopens
small source manifests/checkpoints once; it never runs BRICS generation,
train-oracle scoring, GNN correction, or corrected-package replay.

The new CPU evaluation does calibration first, seals its ordered true rule
IDs, then opens held-out test. Same-root parent checkpoint/cache resume is
supported. Missing L0 distances are first-time exact WNode work; distance cache
hits and misses are reported separately, and no GNN OT is repeated. A valid
zero-coverage/empty-rule result remains zero with undefined conditional costs.

## CPU CLI and dependency chain

The current task explicitly overrides AGENTS' default A800/GPU template: all
three phases run on intel CPU and write only beneath `/share/home/u20526/czx`.
Use an immutable driver checkout and the paired
`scripts/slurm/run_bace_l0_evaluation_successor.sh`.

```
python -I -B scripts/hpc/llm/run_bace_l0_evaluation_successor.py --config configs/hpc.yaml prepare \
  --source-train-root SOURCE/science-verify2 \
  --corrected-package-receipt GNN/verified/result_package.json --output-root FRESH/protocol

sbatch scripts/slurm/run_bace_l0_evaluation_successor.sh evaluate \
  --protocol-overlay FRESH/protocol/protocol_overlay.json \
  --portable-input-bundle EXISTING_L0_PORTABLE --gnn-input-bundle EXISTING_GNN_INPUT \
  --registry-root LLM_ONLY_REGISTRY --output-root FRESH/science

sbatch --dependency=afterok:EVALUATION_JOB scripts/slurm/run_bace_l0_evaluation_successor.sh package \
  --science-root FRESH/science --gnn-input-bundle EXISTING_GNN_INPUT --output-root FRESH/package
```

Submit via `scripts/exp_sbatch.py` with experiment registry/log paths under
FRESH; no tracking output is written into the immutable source checkout.
The package job separately verifies saved pair matrices, calibration order,
numeric metrics and no-padding plateau, without inference or OT. Evaluation
publishes only the independent LLM registry, never the main matrix. The compact
result package contains scientific parent records and hashes, no weights/cache.
Transfer only the completed package and `result_package.json` through Mac to
a fresh AutoDL LLM result import; do not transport T8/GNN packages again.

Tests: `tests/test_bace_llm_common_downstream.py` and
`tests/test_bace_l0_evaluation_successor.py`. Tests use CPU tiny fixtures and
explicitly preserve old source bytes; no full model is loaded.
