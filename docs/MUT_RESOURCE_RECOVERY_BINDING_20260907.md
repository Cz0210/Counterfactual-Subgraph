# Mut resource-stop replay and existing successor binding

## Scope and causal boundary

The2026-09-06 user authorization requires replay1..250, not skippingevent250
or relabeling resource pressure as trace divergence. The legacy observer has
completed1..249, while its algorithm checkpoint has completed250/next251.
There is no joint positive-step checkpoint. Its inputs and both scientific
commits remain unchanged; the new driver only changes observation/completion
transport and the existing orchestration of exact diagnostics.

`run_mut_trace_mode_equivalence.py run-pair --recovery-contract` now runs:

1. freshA1..250 using the same algorithm/RNG/hashseed and real observer;
2. the existing atomic250 checkpoint plus common-observer joint receipt;
3. child exit, then a separate `compare-replayed-250` verifier;
4. original sealed250 versus replayed250 algorithm/RNG/candidate/registry and
   read-only sealed SQLite semantic comparison, plus original observer1..249;
5. only after exactPASS, real checkpoint restoration andA251..510;
6. independentA500→501..510 reload; freshB1..510; independentB reload;
7. originalA/B and reload equivalence gates, then existing postAB/adoption.

The250 transport stop does not reduceM_MAX50000, alter scientific parameters
or change checkpoint cadence: the original storage callback already seals
every250 steps. No diagnostic checkpoint is called a completed50k result.
The two trace modes remain sequential. The additionalA segmentation is
explicit in the manifest; fresh-versus-fresh and resume evidence are not
conflated. Both real RNG and scientific snapshots must agree, not pickle bytes.

The old checkpoint state is deserialized once per new comparison process;
source immutable manifests bind the payloads. Writable descriptors to either
checkpoint prohibit SQLite reads. Active graph databases/WAL are never opened.
Payload comparisons are required new recovery work, not repeated status scans.
Restoration mismatches stop as `BLOCKED_REPLAY_STATE_MISMATCH`, never RouteB.

## Sealed state, not fabricated deployment success

`build_mut_recovery_binding_v1.py seal` writes newA/B arm contracts, one existing
A/B owner spec, postAB command binding, existing next-stage executor spec and
exact existing downstream stage commands. It preserves the canonical owner
and executor lease namespace, publisher identity/locator and one matrix
authority. Source hashes are adopted from the unchanged sealed input spec;
the actual owner still reopens input bindings at scientific admission.

No new scheduler or lock is created. Preparation launches no science and
signals no PID. The old nonreloadable executor must not be fooled with a new
pointer. Before operational activation, the existing canonical registry must
be rebound at a safe boundary, and the exact old idle executor must have no
child/claim; only after fresh resource admission may it receiveSIGTERM and
the original executor command start under the same namespace.

The available RouteB entrypoint still has a typed
`BLOCKED_ADAPTER_MISSING_IF_TRUE_SCIENTIFIC_DIVERGENCE` terminal. This change
does not claim a causal producer or full pair-store rebuild exists. Successful
adoption remains the runnable science path; resource stops cannot selectRouteB.

## Resource contract

The100000 free-inode guard is unchanged. Add160 for the known new compact
recovery layout before acquisition of either owner orGPU lease. Already
existing files are not recharged. Dynamic trace/temp peaks remain explicitly
UNKNOWN, not zero.50GiB and2%-free guards remain. This preparation does not
delete files, change quota, start recovery, or claim complete admission.

## Focused validation

Tests use real tiny torch checkpoint serialization/deserialization and
sealed SQLite fixtures: changed payloads, candidate frequencies and old event
prefixes fail; correct250 snapshots compare;250 cannot be called500;
stop boundaries are only250/510; canonical publisher/lease survive rebind;
storage deficits include the known160 without lowering100000. Existing common
boundary tests exercise real1..500 replay/RNG/candidate and501..510 reload.
These are implementation tests, not production500-step scientificPASS.

The paired Slurm wrappers are intentional AutoDL-only refusal/help adapters,
not requests to run this bound AutoDL recovery onHPC.
