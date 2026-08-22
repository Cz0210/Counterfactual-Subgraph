# AutoDL AIDS/Mutagenicity legacy standardization

This route is deterministic post-processing for existing raw evidence. It does
not submit to HPC, regenerate candidates, fit a selector, tune a threshold, or
write into an old result root. `CLEAR` is intentionally outside the four-method
matrix and is rejected if it appears in the source specification.

## Scientific status encoded by the source specification

| Dataset | Method | Status before continuation | Reusable portion |
| --- | --- | --- | --- |
| Mutagenicity | Ours | `ADOPTABLE_PASS` after strict audit | generation, frozen calibration order, held-out evaluation |
| Mutagenicity | GCFExplainer | `INCOMPLETE` | raw generation only |
| Mutagenicity | GlobalGCE | `INCOMPLETE` | train-only raw generation only |
| AIDS | Ours | `INCOMPLETE` | candidate pool only |
| AIDS | GCFExplainer | `MISSING` | local native output was not transferred into the AutoDL Step0 payload |
| AIDS | GlobalGCE | `MISSING` | local raw evidence was not transferred; its native LHS/RHS attachment mapping is also missing |

The old combined v4 plots are presentation artifacts, not proof that any cell
meets the current oracle, split, WNode, and checksum contract. They are not
adoption inputs.

## Mutagenicity Ours adoption gate

The adopter reads the exact Step0 final root from
`configs/autodl/am_legacy_sources_v1.json` and requires all of the following:

- the `final_result_manifest.json` checksum closure matches every listed file;
- the source has no live writable file descriptor under Linux `/proc`;
- the RF, MolCLR, and held-out test identities recorded in `run_manifest.json`
  remap to physical immutable inputs with exact hashes;
- when the original selector/test-run directories are absent from Step0, the
  self-contained final bundle passes exact manual-audit and threshold-semantic
  evidence checks, and its pair/match/prefix results are independently replayed;
- the complete 217 × 20 Cartesian pair matrix reconstructs independently;
- strict flip remains Mutagenicity label 1 to label 0;
- the RF and MolCLR bytes match the frozen project hashes;
- Figure 3 covers K=1..20 monotonically, Figure 4 is nonempty, and Table 2 is
  exactly K=10;
- source size and modification-time snapshots are unchanged after copying.

Only then does it atomically publish a fresh standardized cell with
`final_artifact_audit_passed=true`. It copies final data; it does not rerun the
held-out evaluator. The output records `generation_adopted`,
`ordering_adopted`, and `evaluation_adopted` independently.

## Foreground commands

Run the one-time adopter from the immutable AutoDL execution worktree **before**
starting the controller. The adopter reopens the already-frozen held-out run to
reconstruct its audit, so it is deliberately not disguised as a controller
`manifest_only` task. The destination must be a fresh path under the exact
future controller artifact root.

```bash
export ARTIFACT_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1
mkdir -p "$ARTIFACT_ROOT/am_legacy/source_specs"
cp configs/autodl/am_legacy_sources_v1.json \
  "$ARTIFACT_ROOT/am_legacy/source_specs/am_legacy_sources_v1.json"
chmod a-w "$ARTIFACT_ROOT/am_legacy/source_specs/am_legacy_sources_v1.json"
export SOURCE_SPEC="$ARTIFACT_ROOT/am_legacy/source_specs/am_legacy_sources_v1.json"
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
export OUTPUT_ROOT="$ARTIFACT_ROOT/am_legacy/precontroller/mutagenicity_ours_v1"
export ACTION=adopt-mut-ours
bash scripts/autodl/run_am_legacy_standardization.sh
```

The persistent controller then performs a manifest-only verification. The
inventory follows that verification automatically; these commands show the
same contract for manual diagnosis:

```bash
export ADOPTED_MUT_OURS_ROOT="$OUTPUT_ROOT"
export OUTPUT_ROOT="$ARTIFACT_ROOT/am_legacy/adoption-verification/attempt-0"
export ACTION=verify-mut-ours-adoption
bash scripts/autodl/run_am_legacy_standardization.sh

export OUTPUT_ROOT="$ARTIFACT_ROOT/am_legacy/inventory/attempt-0"
export ACTION=audit-inventory
bash scripts/autodl/run_am_legacy_standardization.sh
```

Success markers are respectively:

```text
[MUT_OURS_LEGACY_ADOPTION_PASS]
[MUT_OURS_ADOPTION_VERIFY_PASS]
[AM_LEGACY_INVENTORY_PASS]
```

The paired Slurm wrapper exists only for repository CLI parity. This AutoDL
campaign does not submit it.

### Mutagenicity GCFExplainer continuation

The repository already has deterministic native full-graph calibration,
held-out WNode evaluation, export, and final audit kernels. Step0 contains the
completed native Top20 export but not a `frozen_candidate_manifest.json`.
First convert that exact, checksum-pinned export into a fresh frozen package;
this validates its exporter manifests/filter audit and cannot invoke generation:

```bash
export ACTION=freeze-mut-gcf-candidates
export SOURCE_SPEC="$ARTIFACT_ROOT/am_legacy/source_specs/am_legacy_sources_v1.json"
export OUTPUT_ROOT="$ARTIFACT_ROOT/am_legacy/mut_gcf_frozen_top20/attempt-0"
bash scripts/autodl/run_am_legacy_standardization.sh
```

Only after `[MUT_GCF_LEGACY_FREEZE_PASS]`, use that output for deterministic
calibration:

```bash
export ACTION=calibration
export FROZEN_ROOT="$ARTIFACT_ROOT/am_legacy/mut_gcf_frozen_top20/attempt-0"
export FULLGRAPH_CANDIDATES_PATH="$FROZEN_ROOT/export/selected_top20.csv"
export FROZEN_MANIFEST="$FROZEN_ROOT/frozen_candidate_manifest.json"
export CALIBRATION_CSV=/autodl-fs/data/.../calibration_source_label1_teacher_correct.csv
export TEACHER_PATH=/autodl-fs/data/.../mutagenicity_rf_model.pkl
export MOLCLR_ROOT=/autodl-fs/data/.../MolCLR
export MOLCLR_CKPT="$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth"
export THRESHOLDS_JSON=/autodl-fs/data/.../ours_wnode_a2_test_v1/thresholds.json
export WNODE_CACHE_DB=/autodl-fs/data/counterfactual-subgraph-runtime/cache/mut_gcf_wnode.sqlite
export NODE_EMB_CACHE_DIR=/autodl-fs/data/counterfactual-subgraph-runtime/cache/molclr_nodes
export OUTPUT_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/mut_gcf/calibration/attempt-0
bash scripts/autodl/run_mut_gcf_legacy_evaluation.sh
```

After `[MUT_GCF_LEGACY_CALIBRATION_PASS]`, set `ACTION=heldout`,
`HELDOUT_CSV`, `CALIBRATION_RUN_DIR`, `OURS_SCHEMA_ROOT`, and another fresh
`OUTPUT_ROOT`. Success requires the underlying final artifact audit and emits
`[MUT_GCF_LEGACY_HELDOUT_PASS]`. This held-out task must be placed behind its
dataset-specific frozen-selector dependency by the new main controller; the
older BACE-specialized recovery controller intentionally rejects it.

## Continuation-controller contract

`configs/autodl/am_legacy_standardization_v1.tasks.json` is a task fragment,
not a standalone controller manifest. First copy the tracked source spec and
complete the pre-controller adopter at the exact paths above. Then append the
fragment's two task objects verbatim to a fresh continuation manifest:

1. `mut_ours_legacy_adoption_verify` is a CPU, manifest-only verification of
   the already-adopted standardized bundle. It never resolves the raw test
   CSV or frozen test-run root.
2. `am_legacy_inventory` depends on the exact passing adoption output and
   publishes a six-cell matrix patch plus bounded filename inventory.

Both controller inputs use the authoritative production prefix
`{runtime_root}/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/am_legacy/`,
which expands to the same `$ARTIFACT_ROOT/am_legacy/` location used by the
foreground commands. They intentionally do not use `{artifact_root}` because
the production controller defines that placeholder as `$RUNTIME/outputs`.
The source-spec filename does not name a raw split, and the inventory never
opens molecular rows. The controller owns detachment, heartbeat, retry, and
registry state for these two tasks.

## Fail-closed outputs

The inventory never promotes a configured incomplete cell based only on a
similarly named CSV, PNG, or directory. A missing configured root becomes
`MISSING`. Multiple existing roots are ambiguous and fail the task. AIDS
GlobalGCE remains
`BLOCKED_GLOBALGCE_LHS_RHS_ATTACHMENT_MAPPING_UNAVAILABLE` until a tested
native transformation adapter exists; coercing its rule into deletion
semantics is forbidden.
