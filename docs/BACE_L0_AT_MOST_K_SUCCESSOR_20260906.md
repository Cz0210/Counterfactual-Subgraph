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

All LLM common evaluation CLIs also accept paired `--gnn-acceptance` and
`--gnn-acceptance-sha256` flags. The existing owner acceptance verifies its
small receipt and unchanged archive stat identity without archive hashing or
GNN replay. The paired generic wrapper passes `GNN_ACCEPTANCE` and
`GNN_ACCEPTANCE_SHA256`; absent flags retain first-adoption verification.

## Actual completed evaluation and scoped import

HPC execution commit `c6d5f0b507d75eaa93363dae49edbce0a5568141`:
evaluation2567257 completed0:0 in2m20s; afterok package2567271 completed0:0
in10s. The old2560839 failure remains unchanged. New L0 calibration has66
true-source parents and test141; this is the original BACE main
`load_bace_parents(source_label=1)` cohort, not GNN native/common filtering.

AutoDL import is a separate CPU-only `import-result` action. It checks only
this new small result's transport/inner hashes and previously independent
package audit, extracts into a fresh LLM-only root, and calls the existing
independent LLM registry publisher. It does not repeat model inference,
selection, OT or GNN verification. Original run manifests retain HPC provenance;
`publication_receipt.json` supplies the additive local locator.

```
python -I -B scripts/hpc/llm/run_bace_l0_evaluation_successor.py --config configs/hpc.yaml import-result \
  --archive FRESH_INCOMING/bace_l0_at_most_k.tar.gz \
  --package-receipt FRESH_INCOMING/result_package.json \
  --output-root FRESH_LLM_IMPORT --registry-root INDEPENDENT_LLM_REGISTRY
```

The CLI accepts import output only under AutoDL's `outputs/autodl/ablations/llm`;
the paired HPC wrapper refuses `import-result`. Neither route writes main control.
