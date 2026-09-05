# BACE common LLM downstream

The executable adapter is `scripts/ablations/llm/run_bace_common_downstream.py`.
It changes only the proposal pool. It does not train a classifier or LLM and
cannot append the main matrix. A successfully validated independent GNN
scientific archive is mandatory before any oracle load or result-root creation.

The native task specification and reference contract fix 386 training parents,
8 attempts per parent, the frozen BACE GINE and validation-fitted temperature,
MolCLR checkpoint, exact node-Wasserstein configuration, and main A4 selector
parameters/thresholds. BRICS adopts the existing vocabulary and proposal pool;
the attempt ledger, including unfilled slots, is verified separately. No
vocabulary is rebuilt. The old BRICS manifest has no direct reference field;
its SHA-bound vocabulary and shortfall manifests provide the two reference
links. No historical artifact is rewritten.

Science implementation:

- Main `bace_frozen_gnn_pool._score_generated_candidates` parses, canonicalizes,
  enumerates connected hard deletions, predicts with GINE, and scores attempts.
  Projection is disabled as in main B8/B9; invalid attempts remain in metrics.
- A narrow extraction of main B10's canonical parent/fragment merge preserves
  the original deterministic duplicate tie-break. Train flips do not filter
  the structural universe. Fewer than 20 valid unique rules produces a typed
  scientific failure, never padding or fictitious performance.
- Main `bace_frozen_gnn_verification._evaluate_rows` and exact MolCLR-WNode
  calculate complete cross-parent/candidate rows. The LLM cohort is the main
  route's **all true source-label 1 parents**, not the GNN-ablation native or
  common correctly-predicted subset.
- Frozen main selector settings are applied once globally to calibration.
  The order and input hashes are sealed before the test CSV is parsed. Test
  evaluates only those 20 selected rules. A nonempty cohort with no strict
  flips reports actual zero coverage; undefined costs and empty test cohorts
  are N/A. Empty calibration is blocked.
- Parent bootstrap is fixed to seed 7, 1,000 resamples, with no selector refit;
  it is not an across-seed standard deviation.

Each train, calibration, and test parent has a scientific-input-bound atomic
checkpoint. `SIGTERM` or an output-local `pause.request` pauses at the next
parent boundary. Resume is explicit and refuses changed inputs or selections.
Remove a user-requested pause file only when the scheduler authorizes resuming.

Example (after independent GNN archive PASS; actual paths must come from the
launch receipt):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD \
python scripts/ablations/llm/run_bace_common_downstream.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --task-spec "$LLM_TASK_SPEC" --candidate-root "$LLM_CANDIDATE_ROOT" \
  --gnn-input-bundle "$GNN_INPUT_BUNDLE" \
  --gnn-verified-archive "$GNN_VERIFIED_ARCHIVE" \
  --gnn-verified-sha256 "$GNN_VERIFIED_SHA256" \
  --registry-root "$LLM_REGISTRY_ROOT" --output-root "$LLM_DOWNSTREAM_ROOT" \
  --device cpu --cpu-threads 2
```

The paired `scripts/slurm/run_bace_common_downstream.sh` uses the standard A800
Slurm profile if a GPU run has separately been authorized. CPU execution on
AutoDL does not obtain a GPU lease. The launcher remains responsible for
main-task priority and GPU authorization; this adapter never starts a model
generator or acquires a main-task lease.

Outputs include scored attempts, merged pool/universe, metrics, immutable
calibration selector, selected rules, held-out metrics, Figure 3/4 CSVs,
Table 2 K=10, parent-bootstrap CI, and a hash inventory. Only the independent
`llm_result_registry.json` receives a same-hash-idempotent append. Main authority
directories are rejected.
