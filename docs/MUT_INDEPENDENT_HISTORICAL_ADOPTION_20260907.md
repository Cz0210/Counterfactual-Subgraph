# Mut independent historical adoption — 2026-09-07

## Chemistry-only CPU boundary

The existing standardizer accepts `--through-stage chemistry`, with an explicit
`--stage-resource-config` and `--persistent-root`. It uses its normal startup
barrier and child session, plus the existing Mut nonblocking invocation lease.
The optional five-second observer reuses the existing process-tree reader;
pressure exits through the runner's existing SIGTERM/quiescence cleanup, never
SIGKILL. Budget caps are not measured chemistry peaks: RSS96GiB, other-main
reserve192GiB, transient32GiB, start headroom320GiB, persistent free100GiB.
Only more conservative reserve/cap settings are accepted.

Stage `mut_chemistry` has a conservative128-new-inode code bound. The chemistry
implementation writes at most32 fixed containers/manifests (all100235 candidate
and action rows are consolidated), no per-candidate files. Allow16 wrapper,
preregistration, startup-barrier, resource/phase records;16 directory/lock/log
entries;32 transient/error-publication entries;32 unused safety entries.
Existing inputs are not counted again. New WNode per-graph NPZ caches belong to
the separate, not-yet-admitted `mut_unified_evaluation` stage.

Successful chemistry publishes `chemistry_stage_boundary.json` in state
`SEALED_CHEMISTRY_WAITING_EVALUATION_ADMISSION`, **not** final cell PASS.
Use the identical immutable execution tree and arguments, replace
`--through-stage chemistry` with `--resume-after-chemistry`, and bind a new
resource config whose `mut_unified_evaluation` stage is admitted. This checks
the saved command/input identities and small closure manifests before skipping
chemistry; it refuses to duplicate an already-created evaluation directory.
No source payload/model is rehashed at this boundary. The remainder uses the
unchanged RF/WNode evaluator, project-full gate, freeze and canonical publisher.
An interrupted chemistry with no completion marker is not promoted or resumed
as complete; retain its failure and diagnose the exact stage.

Only `mutagenicity_comrecgc_lineage_v3_20260822T025620Z` is eligible for this
new explicitly authorized route. No search over runs or performance-dependent
selection is allowed. The old A/B and resource-stop evidence remains unchanged.

The complete historical inventory already hashed the 3,191,869,449-byte payload.
Its closure independently reloaded all frozen graph records, and the selected
action receipt records 224,690 exact replayed transitions and all 100,235
candidate lineages. Fourteen selected-target parent mismatches and one
cross-parent canonical convergence are retained with their existing reviewed
interpretation; they are not silently rewritten to zero.

The original pair producer consumed this exact payload and froze the ordered
50,620 candidate subset, generation indices and 1,448 parents. Its complete
791-chunk manifest is linked to the read-only adoption and 813,595-row exact
DBSCAN input. The DBSCAN has a vector identity, not a native candidate-universe
field. The new route calls this producer-receipt/transitive evidence; it does
not claim a fresh full-payload reconstruction. Only device-id drift across a
remount is accepted without changing inode, size, times or mode.

Native generation uses the frozen project GNN and NeuroSED, as the pinned
historical implementation and recovery protocol require. Only the final
slot-preserving evaluator queries RF/MolCLR-WNode. RF does not retrospectively
rank or change native candidates. `RESUME_SAFE=false` on the old completed
artifact is disclosed separately from its successful frozen-payload reload.

`scripts/autodl/adopt_mut_historical_independent.py --action audit` is CPU-only,
read-only and prints the classification. `--action seal` additionally requires
the narrow authorization, fresh output root and the existing stage resource
policy descriptor. It writes only two result JSON files using the existing
atomic writer (20 new-file upper bound includes directories and simultaneous
temporary outputs); it starts no controller, GPU, algorithm or matrix writer.

The resulting `historical_adoption.json` enters the existing
`run_mut_comrecgc_parity_standardization.py --historical-adoption` interface.
The v3 branch alone removes the old 500-step prerequisite and honestly emits
`500_step_semantic_equivalence_passed=false`. The old v2 validator remains
unchanged. The original matrix publisher independently reopens the new audit.
Final chemistry and CPU-only RF/MolCLR evaluation still require their own
resource preflight and real scientific completion before matrix publication.

No paired Slurm GPU request is appropriate for this metadata operation: its
paired script uses the explicit current CPU-only task override. Real AutoDL
dispatch remains under the existing project owner and resource provider.
