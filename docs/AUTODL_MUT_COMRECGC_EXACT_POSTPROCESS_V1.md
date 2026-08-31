# Mutagenicity ComRecGC exact read-only postprocess v1

## Scope

This route closes only the downstream work for the adopted Mutagenicity exact
science result.  The adopted root already contains the completed labels,
multi-component DBSCAN partition, centroids, radii, coverage, and official
greedy selection.  This runner cannot invoke pair-store construction, DBSCAN,
or common-recourse selection.

The ordered stages are:

1. validate the exact adoption receipt and all source hashes read-only;
2. require a real `mut_trace_on_off_parity_v1` PASS receipt;
3. run deterministic chemistry with exactly 100 selected common recourses;
4. run standardized WNode evaluation against the frozen test CSV, RF teacher,
   MolCLR checkpoint, and threshold contract;
5. run the full gate and freeze the standardized cell;
6. append only `Mutagenicity/ComRecGC` to a hash-closed matrix authority.

The continuation uses deletion-based `strict_flip` semantics.  It does not
reinterpret a selected fragment as a concept that predicts the original label.

## Exact adoption closure

Adoption is fail-closed on the production scientific identity.  In addition to
the receipt and artifact SHA-256 closure, the runner requires all of the
following without aliases or tolerances:

- common-recourse parameters `theta=0.1`, `delta=0.02`,
  `recourse_size=100`, `cf_size=100000`, `cluster_size=3`, and `seed=0`;
- exactly 100 rows in `selected_common_recourses.json`;
- DBSCAN `eps=0.02`, `min_samples=3`, Euclidean brute-neighbor search,
  sklearn float64 label semantics (including self-neighbor behavior), and
  sklearn `1.7.2`;
- `sklearn_float64_exact_multi_component_v1`, four exact workers, no
  single-component assumption, no failure-cap use, no approximation, and the
  frozen minimum-adjacent-core-label border assignment;
- a hash-valid terminal controller state for stage
  `MUT_EXACT_MULTICOMPONENT_FAST16`, with matching worker PID, exit code zero,
  empty failures, and both controller and worker absent from procfs.

The canonical trace parity validator is used directly.  Its
`traced_source_root` must be the same physical generation root adopted by the
exact result; a valid receipt for another generation is rejected.

## Current scientific blocker

The completed exact root and its adoption receipt are sufficient for the
common-recourse half of the continuation.  The frozen generation trace at

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/recovery/
mutagenicity_comrecgc_lineage_v3_20260822T025620Z/trace/trace_summary.json
```

is trace-only evidence and contains neither `trace_parity_passed=true` nor an
explicit accepted Mutagenicity no-RNG exception.  Mutagenicity intentionally
does not share the AIDS/BACE trace-integrity waiver.  Consequently the runner
requires a separate `mut_trace_on_off_parity_v1` receipt with `status=PASS`.
It validates that receipt before creating the fresh postprocess root.

Do not point `MUT_TRACE_PARITY` at `trace_summary.json`, and do not weaken
`validate_chemistry_trace_evidence`.  Either action would overstate the
available scientific evidence.

## Frozen inputs

The Slurm wrapper defaults the immutable science inputs to the known adopted
locations, including:

- exact full root:
  `.../mut_comrecgc_exact_multicomponent_v1_20260830T184359Z/full`;
- exact adoption receipt:
  `.../control/fast_16of16_v2/adoptions/mut_comrecgc_exact_multicomponent_adoption.json`;
- generation root:
  `.../mutagenicity_comrecgc_lineage_v3_20260822T025620Z`;
- teacher-consistent held-out test CSV and frozen RF teacher;
- frozen MolCLR checkpoint;
- the existing Mutagenicity ComRecGC threshold contract.

The operator must supply four run-specific authorities:

```bash
export MUT_TRACE_PARITY=/absolute/path/to/trace_parity.json
export MUT_POSTPROCESS_OUTPUT_ROOT=/absolute/fresh/postprocess/root
export PRIOR_MATRIX_ROOT=/absolute/path/to/current/closed/matrix
export MUT_MATRIX_OUTPUT_ROOT=/absolute/fresh/matrix/root
```

Then submit:

```bash
sbatch scripts/slurm/run_mut_comrecgc_exact_postprocess_v1.sh
```

For an interrupted downstream stage, keep every frozen input unchanged and set:

```bash
export MUT_POSTPROCESS_RESUME=1
sbatch scripts/slurm/run_mut_comrecgc_exact_postprocess_v1.sh
```

Resume reopens the frozen command/input contract, validates completed stage
markers and quiescent child process groups, and archives an incomplete
non-checkpointed downstream stage before retry.  It never resumes or rewrites
the adopted exact root.

The postprocess root and matrix successor must be absolute, non-symlink,
mutually disjoint paths.  Neither may equal, contain, or be contained by the
generation, exact, parity, controller, upstream, dataset, teacher, MolCLR,
threshold, or prior-matrix authorities.  The normalized isolation result is
part of the resume contract.

## Matrix publication

Matrix publication accepts only a closed prior authority whose
`Mutagenicity/ComRecGC` row is not already passing and whose `Mutagenicity/Ours`
row supplies the frozen dataset/split/oracle/MolCLR/distance/threshold identity.
The successor must add exactly one passing cell.  Every non-target row must be
byte-for-byte equal as structured JSON, the standardized source must have no
live writer, and the published authority must pass an independent reopen.

Publication is crash-recoverable: all registry files and the append receipt are
first written and verified in a unique same-parent staging directory.  Only a
no-replace atomic directory rename may expose the final matrix path.  An
interruption before that rename leaves the final path absent; an interruption
after it is adopted only after revalidating the prior hashes, all non-target
rows, the target/shared identity, source inventory, writer audit, append flags,
and complete combined-audit closure.

If the current matrix has a non-target transient row that cannot be reproduced
from its frozen passing-cell roots, the strict append fails.  Provide a closed
predecessor with stable non-target rows; do not edit or impute the matrix.

## Focused verification

```bash
python -m pytest -q tests/autodl/test_mut_comrecgc_exact_postprocess_v1.py
python -m compileall \
  src/utils/autodl_mut_comrecgc_exact_postprocess_v1.py \
  scripts/autodl/run_mut_comrecgc_exact_postprocess_v1.py
bash -n scripts/slurm/run_mut_comrecgc_exact_postprocess_v1.sh
git diff --check
```
