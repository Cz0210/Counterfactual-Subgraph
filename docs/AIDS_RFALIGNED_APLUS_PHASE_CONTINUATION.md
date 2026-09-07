# AIDS RFAligned A+ completed-pair continuation (2026-09-08)

Architecture remains frozen pool → normalized GREED pairs → existing certified
exact DBSCAN → existing native centroid/radius/greedy recourse → original RF and
WNode evaluation/release. The new dataset-specific CLI starts **after** the
completed pair store; it never calls pool screen, graph restore, GREED embedding,
pair materialization, or RF-guided search. The old failed owner and receipts are
not edited. The source is the RF-aligned 37,342,977-row / 266-chunk universe, not
the historical 91,916,686 GNN pair universe.

## Scope and immutable mapping

- AIDS/HIV source 1283; frozen RF source=1 count1097; target=0.
- 34,041 screened candidate records retain original order/multiplicity.
- vectors float32 `(37342977,64)`, pair IDs int64 `(37342977,2)`;
  candidate-major / parent-minor order; original normalized GREED theta0.1.
- Existing DBSCAN epsilon0.02, min_samples3, inclusive radius/self-neighbor
  contract, core components and earliest-core border assignment are unchanged.
- Shortcuts still require full certificates, including multi-component recovery.
  Above100000 rows an unproven shortcut is a complexity blocker, not permission
  to perform an unreported all-pairs fallback.
- Source generation/evaluation overlap remains disclosed. This is not a new
  independent held-out AIDS experiment.

## Execution-only differences

One CPU worker, OMP/MKL/OpenBLAS/NumExpr threads1, no CUDA visibility. The
engine's resource block ceiling is1024 query rows, versus the old65536 default;
64 blocks are committed together, retaining the old maximum65536-row durable
boundary instead of introducing36,000 growing-ledger fsyncs per scan;
scientific values/order are not changed. Existing focused exact tests cover
resource-block independence, core/border/components and resume.

The separate phase budget remains14GiB, including full vector page residency,
17bytes/row full-length state pages, up to3GiB for the capped4099-anchor Python
graph/edge construction, 256MiB bounded-query workspace, 1GiB NumPy/sklearn
runtime and128MiB owner/sampler. Summary substitutes pair pages and2GiB runtime
for RF/Torch/selected records. The bound is below the taskbook64GiB experiment
ceiling; the ceiling is **not** claimed to be an installed child cgroup limit.

The old384GiB other-task reserve and100GiB persistent-space floor are unchanged.
At start, actual cgroup headroom must cover the phase's remaining peak plus that
reserve. Only conservative private anonymous pages already charged to this
process may be deducted; shared/file-cache pages are never deducted. During a
running phase actual cgroup usage already includes the task, so runtime checks
preserve the other-task reserve rather than double-reserving the whole task.

A five-second real process-tree/cgroup/RSS/cache sampler writes bounded compact
state. After the original engine atomically commits a checkpoint, a scoped
callback in the new process waits if live resource evidence no longer passes.
No SIGKILL, active-file replacement, CUDA borrowing or global environment change
is used. The exact engine source is not changed. Summary checkpoint callbacks
use the same post-commit rule. The final RF medoid check has a separate memory
receipt. The process-tree sum conservatively double-counts shared RSS; it is not
substituted for cgroup usage. No child-cgroup hard isolation is claimed.

## Owner and publication

`scripts/continue_aids_rf_pairs.py --action owner` acquires the **existing**
`recourse_root/writer.lock`. The actual FD is inherited by its science child,
which verifies identity and independent-open exclusion. Only cluster-existing
and summary-existing can be dispatched. Resource waiting samples every60s;
science is not claimed before a real child exists. Successful phases are not
repeated. Failed/unreconciled submissions require diagnosis, not silent retries.

The original writer FD is closed before the existing release-after-recourse
entrypoint runs. Release preserves original1283/RF/WNode/frozen thresholds and
the original same-cell matrix authority CAS. The owner has no parallel matrix
writer. A successful DBSCAN certificate alone is not a released scientific cell.

Paired CPU Slurm wrapper: `scripts/slurm/continue_aids_rf_pairs.sh`.
AutoDL source vectors remain in place; this task does not copy them to HPC.

## Focused validation

`tests/test_aids_rf_cluster_phase.py`, plus the existing external DBSCAN tests
for connected/disconnected components, border/noise, inclusive boundaries,
duplicate identity, resume, capped fallback and resource block independence;
`tests/test_aids_rf_incremental_memory.py`.

Implementation/tests/deployment/live science/final acceptance are reported
separately. Phase plans and CPU fixture evidence are not scientific PASS.
