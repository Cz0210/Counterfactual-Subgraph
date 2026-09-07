# Offline partial-snapshot numerical replay

`scripts/paper/reduce_result_snapshot.py` packages the reusable reduction from
the 2026-09-07 artifact-only closeout. It consumes already collected JSON
snapshots, not live owners, remote source files, checkpoints or databases.
It never collects inputs, launches science, fits a threshold, changes candidate
order, recomputes distances, edits a matrix or writes a scientific PASS marker.

The existing final 16-cell exporter and its release gates are unchanged. This
entrypoint is **PARTIAL staging**, even if a later supplied snapshot happens to
contain all cells. Its audit is a numerical replay, not independent scientific
or source-authority verification. It must not be used as a matrix-adoption gate.

## Inputs and reproduction

Provide a read-only snapshot directory containing these three existing formats:

- `canonical/small_results_snapshot.json`: the collected authority pointer,
  accepted cell metadata, and embedded small result CSV/JSON texts.
- `lineage/raw_reduction.json`: precomputed legacy parent IDs and the complete
  K=1..20 strict-flip prefix-best rectangle, plus original pair/order bindings.
- `lineage/v2_combined_manifest.json`: the original legacy threshold and
  pair/order SHA declarations. It is not a replacement data/split identity.

The collector and legacy raw-file reducer remain separately audited artifact
steps; this command does not implement or replay those source reads. For a
snapshot without legacy cells, use `{"cells": []}` for `raw_reduction.json`
and `{}` for `v2_combined_manifest.json`. No input data ships in Git.

From the checked-out source commit:

```bash
python -I -B scripts/paper/reduce_result_snapshot.py \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --output-root "$FRESH_ARTIFACT_ROOT"

python -m pytest -q tests/test_paper_snapshot_reduction.py
```

`SNAPSHOT_ROOT` must be a local, already collected snapshot. The output must be
a fresh external artifact directory, not inside/above that snapshot or inside
the source checkout. Keep it separate from all scientific and user manuscript
roots. Each input JSON is limited to 64 MiB and checked for stat changes during
its read. Output is generated only after validation; an interrupted write has
no final `export_audit.json` and must not be adopted. Do not reuse that path.

The command writes the existing Figure3/Figure4/Table2-compatible source CSV
names, exact threshold keypoints, registered-versus-reduced differences,
cohort bindings, and all-K wins/ties/losses/unavailable comparisons. Missing
matrix cells are derived from the canonical four-dataset/four-method contract,
never hardcoded from a previous 13/16 snapshot. They have no numerical values.
`current_registered_cells.csv` deliberately omits a stale cell count in its
filename. Original embedded registered figures/tables remain untouched.

The audit binds the exact three local snapshot byte streams and each generated
artifact with SHA-256. Source paths are descriptive only and are never opened.
Inherited source byte hashes are read from summary/final-audit/run-manifest
receipt fields and checked for conflicting declarations; they are **not newly
verified source bytes**. The hash of embedded UTF-8 CSV text is a separate field
because text collection may normalize line endings. Missing original per-file
SHA remains `PER_FILE_HASH_UNAVAILABLE`, not repaired by a transport hash.
Legacy raw-prefix SHA declarations must match the supplied V2 manifest; this
does not independently establish that the precomputed prefix matches that raw
source. The old legacy identity waivers remain explicitly open.

## Numerical contracts

- Preserve native parent order and the complete, nonempty, unique cohort at
  every K. Reject increasing prefix minima, lost strict recourse, negative or
  nonfinite distances, malformed booleans, duplicated cells, and missing K.
- Coverage uses `best_distance <= frozen_threshold`, including equality.
- AIDS/Mutagenicity cost is the median over **all strict-flip parents**, not
  only threshold-covered parents. No strict flip means `N/A`, not zero.
- BACE/TasteMolNet cost is the frozen-cap mean over **all parents**; unavailable
  recourse contributes the cap. Zero coverage need not imply unavailable cost.
- Figure4 retains its actual sorted unique threshold grid. Exact off-grid
  keypoints, Figure3 K10 and Table2 all use the same K10 parent reduction; no
  interpolation, grid replacement, smoothing or selector fitting occurs.
- A source Figure4 K20 label can be projected to K10 only when the complete
  K20 and K10 prefix vectors are exactly equal. Other mismatches fail.
- Nonlegacy Figure3, Figure4 and Table2 must agree with reduction. Legacy
  display differences are preserved in a separate audit, never written back.
- The `1e-12` comparison tolerance applies only to CSV numeric comparison and
  displayed ranking ties; input distances/thresholds are never changed.
- Cross-method comparisons require matching snapshot parent-ID sets, frozen
  theta and the applicable cost contract/cap. Otherwise comparisons and ranks
  are unavailable; the CLI never invents a parent-ID mapping. Incomplete
  datasets have no numeric rank. Matching IDs still do not close historical
  split/source identity waivers or constitute cross-dataset comparability.
  Different ID namespaces are not proof of different molecules: this initial
  CLI conservatively suppresses comparisons until an independently bound,
  complete source-to-canonical identity mapping is explicitly supported. Equal
  parent counts never substitute for that mapping.

## CPU-only Slurm exception

The paired `scripts/slurm/reduce_result_snapshot.sh` is a thin, opt-in `intel`
CPU-only wrapper. Set `SNAPSHOT_ROOT` and a fresh `OUTPUT_ROOT` before `sbatch`.
It preserves the project environment/PYTHONPATH/config diagnostics but requests
no A800/GPU and hides CUDA. This explicitly authorized artifact-only exception
does not change the project's normal science submission template. No remote
deployment or Slurm submission is part of this source change.

## Renderer boundary

The complete closeout renderer is **not** migrated in this bounded commit. Its
style loader, GNN/LLM inputs, accepted-state captions and snapshot-specific
claims need separate parameterization and visual QA. This CLI emits numeric
inputs only and promises no PDF equivalence. The previously audited artifact
renderer and its outputs remain intact; no figures, PDFs, datasets or weights
are committed. A future renderer should reuse the existing GCF plotting style
and derive status/cohort/budget prose from explicit, verified export metadata.
