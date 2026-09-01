# AutoDL TasteMolNet T14 ComRecGC paper-cell postprocess

## Scope

`scripts/run_tastemolnet_comrecgc_full.py` is intentionally train-only. Its
`GENERATION_PASS` proves the native 20k/25k walk and resource-cap decision, but
leaves calibration, held-out test, export, and matrix eligibility unevaluated.
This dataset-specific successor closes only those stages and never reruns or
changes the walk.

The implementation lives in:

- `src/baselines/tastemolnet_comrecgc_postprocess.py`
- `scripts/run_tastemolnet_comrecgc_postprocess.py`
- `scripts/autodl/run_tastemolnet_t14_comrecgc_postprocess.sh`
- `scripts/autodl/run_tastemolnet_t14_postprocess_relay_v1.sh`
- `scripts/autodl/launch_tastemolnet_t14_postprocess_relay_v1.sh`

The paired `scripts/slurm/run_tastemolnet_comrecgc_postprocess.sh` preserves
CLI documentation but refuses direct execution because this route consumes
retained AutoDL authorities.

## Scientific contract

The continuation validates the generation terminal and exact effective 20k or
25k checkpoint. It joins each official common-recourse representative hash to
the checkpointed bridge record and lineage multiset. No graph is synthesized,
repaired, backfilled, or reordered. The representative set must contain
10--20 unique connected frozen-GINE strict non-Sweet graphs.

The same T3 calibrated three-class GINE is required by SHA; RF is forbidden.
The shared Taste WNode threshold JSON must prove a method-shared,
test-independent calibration source. Configured/effective M, resource-cap use,
early-stop use, and stop reason are copied from generation into final output.

Calibration writes one-parent atomic chunks and freezes up to 20 candidates
using marginal theta coverage, strict-flip coverage, WNode distance, and stable
candidate identity. Held-out test is not hashed or opened until both the
selection manifest and `CALIBRATION_SELECTION_FROZEN` checkpoint are fsynced.
The test exports the same Figure 3, Figure 4, prefix, destination, and Table 2
K=10 artifacts used by the other Taste full methods. Strict flip remains
`pred_before == 1 and pred_after != 1`, with destinations 0 and 2.

## Resume and publication

`--mode postprocess --resume` reopens only complete parent chunks bound to the
same split, candidate order, GINE, MolCLR, threshold, and generation checkpoint.
A partial chunk/receipt pair or byte drift fails closed.

Science stops at `SEALED` and cannot create `PASS`. A separate `--mode verify`
invocation revalidates generation, rejoins lineage, reconstructs chunks,
replays selection and metrics, runs the four-by-four registry candidate gate,
and publishes a fresh directory with no-replace rename. The final file is:

```text
[TASTE_COMRECGC_PASS]
```

## AutoDL operation

```bash
export TASTEMOLNET_T14_GENERATION_ROOT=/absolute/generation-pass-root
export TASTEMOLNET_T14_POSTPROCESS_ROOT=/absolute/fresh-or-resumable-science-root
export TASTEMOLNET_T14_FINAL_ROOT=/absolute/fresh-final-root
export TASTEMOLNET_CALIBRATION_CSV=/absolute/calibration.csv
export TASTEMOLNET_TEST_CSV=/absolute/test.csv
export TASTEMOLNET_T3_OUTPUT_ROOT=/absolute/t3-root
export MOLCLR_ROOT=/absolute/molclr-source
export MOLCLR_CHECKPOINT=/absolute/molclr-checkpoint.pth
export TASTEMOLNET_WNODE_THRESHOLD_JSON=/absolute/tastemolnet.json
export WNODE_CACHE_DB=/absolute/cache.sqlite
export NODE_EMBEDDING_CACHE_DIR=/absolute/node-cache
export TASTEMOLNET_T14_POSTPROCESS_RUN_ID=taste-t14-postprocess-UUID
export TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX=2
export TASTEMOLNET_T14_POSTPROCESS_RESUME=0
scripts/autodl/run_tastemolnet_t14_comrecgc_postprocess.sh
```

Set resume to `1` only for the same science root after it contains
`postprocess_checkpoint.json`. The final root must always be absent.
`RUN_GNN_ABLATION` remains `0` until the matrix is 16/16.

## Durable generation-to-paper relay

The narrow relay is appropriate when generation is already running from an
immutable checkout. It does not launch, restart, resume, or signal generation.
It reopens `launcher.json`, binds manager and science process generations by
PID start ticks and exact output-root command tokens, and waits for the exact
`[TASTE_T14_COMRECGC_FULL_GENERATION_PASS]` bytes. Before postprocess it waits
for both bound processes to exit and requires a procfs writable-FD audit of the
generation root. The ordinary postprocess wrapper then supplies the stable
idle-GPU and project GPU-lock gates.

Launch with the retained scientific inputs and one clean deployed checkout:

```bash
export T14_RELAY_REPO_ROOT=/absolute/immutable/worktree
export T14_GENERATION_ROOT=/absolute/fresh/t14-generation-root
export T14_GENERATION_LAUNCHER_JSON=/absolute/controller/launcher.json
export T14_GENERATION_EXECUTION_COMMIT=40_character_generation_commit
export TASTEMOLNET_CALIBRATION_CSV=/absolute/calibration.csv
export TASTEMOLNET_TEST_CSV=/absolute/test.csv
export TASTEMOLNET_T3_OUTPUT_ROOT=/absolute/t3-root
export MOLCLR_ROOT=/absolute/molclr-source
export MOLCLR_CHECKPOINT=/absolute/molclr-checkpoint.pth
export TASTEMOLNET_WNODE_THRESHOLD_JSON=/absolute/tastemolnet.json
export RUN_GNN_ABLATION=0
bash scripts/autodl/launch_tastemolnet_t14_postprocess_relay_v1.sh
```

The launcher prints the fresh controller root. Its
`cell_root_locator.json` appears only after the independent final verifier has
published exact `[TASTE_COMRECGC_PASS]`; point the existing fast16 publisher
queue's TasteMolNet/ComRecGC entry at that locator. The relay itself never
edits the matrix authority.
