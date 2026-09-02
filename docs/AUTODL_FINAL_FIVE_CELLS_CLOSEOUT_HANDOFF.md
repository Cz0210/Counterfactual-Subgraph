# AutoDL final-cell closeout handoff (2026-09-02)

## Authority and completed work

- Matrix authority pointer:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`
- Current registered matrix: **12/16**.
- TasteMolNet/Ours (T11) was published without a science rerun through the
  content-identical policy relocation overlay.  Its row is now part of the
  hash-closed authority.
- BACE is 4/4.  Remaining rows are Mutagenicity/ComRecGC and TasteMolNet
  GCFExplainer, GlobalGCE, and ComRecGC.

## Mutagenicity / ComRecGC

- Runner commit: `f0e1d5ec9d38c451984356798a701ffd04d99f19`.
- Runner worktree:
  `/root/autodl-tmp/worktrees/mut-semantic-adoption-f0e1d5e-20260902T070000Z`.
- Resolver worktree:
  `/root/autodl-tmp/worktrees/final-five-closeout-582bc4b-20260902T040000Z`.
- Fresh one-shot PID: `67797` (nohup launcher/worker process at handoff).
- Fresh output:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_trace_on_adoption_semantic_finalizer_20260902T041411Z`.
- Log:
  `/autodl-fs/data/counterfactual-subgraph-runtime/logs/mut_trace_on_adoption/mut_semantic_finalizer_20260902T041411Z.log`.

The historical 7f/664 worktrees still own generation and random-walk
semantics.  Only post-walk lineage materialization is delegated, under a
private module namespace, to the exact reviewed 582 resolver.  Both 500-step
arms must agree on semantic-transition sequence digests.  The route does not
regenerate 50k or recompute the pair store/DBSCAN.  Expected end-to-end time is
about 3.5--6.25 hours.

## Taste T8 / T13

- Existing sealed branches remain read-only:
  - target 0: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t8-dual-branch-recovery/target-0-attempt-3af51e32-8429-4b3e-8bae-71ba16df1683/state/target-0`
  - target 2: `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t8-dual-branch-recovery/target-2-attempt-4a0651fa-5df4-43c0-a897-20f040f779d8/state/target-2`
- Typed-preflight commit: `ba8374d7f64fd923e668430112c041373c37834e`.
- Fresh typed-preflight controller:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t8-salvage-20260902T042219Z-da32d7e3`.
- Fresh typed-preflight attempt:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t8-salvage/attempt-9c7105bf-fe6c-46de-9ef8-da56e2b24c98`.
- Typed RHS evidence:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t8-salvage/attempt-9c7105bf-fe6c-46de-9ef8-da56e2b24c98/rhs-standalone-chemistry-preflight.json`.
- Recovery request:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t8-salvage/attempt-9c7105bf-fe6c-46de-9ef8-da56e2b24c98/single-branch-rerun-request.json`.

The seven exposed RHS rules are chemically unusable independently of the
parent, so the current contract cannot produce a real candidate or strict
flip.  Do not start T13 from these artifacts.  The recommended single recovery
decision, if explicitly authorized, is one fresh seed-7, 100-epoch,
T13-grade two-target train-only attempt, maximum one attempt, no decoder
changes and no test access.  Estimated sequential training is 34--38 hours
plus 0.5--1 hour validation, with only low-to-medium success likelihood.

## Taste T12 / GCFExplainer

- Failed root (read-only evidence):
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/gcfexplainer/t12-production/attempt-6616449a-6fa5-4502-8c8a-ae01b11366fb`.
- The failed partial journal contains 967,680 complete records but no 10k
  checkpoint, so it is not a legal resume point.
- Lock-fix commit: `104814451f80f07c09a23fc735d64d0d631a4bb5`.
- Fresh generation controller:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t12-gcf-release-20260902T035837Z-7e5b1f3f`.
- Fresh science PID at post-launch verification: `66459`, GPU3.
- Fresh root:
  `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/gcfexplainer/t12-production/attempt-f6c6df48-563e-4c49-9d83-c7136fc84152`.
- Paper relay PID `66548`:
  `/autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t12-paper-after-generation-20260902T035900Z-1048144`.

Never open the active `history_index/*.sqlite3`, even read-only.  Status must
use heartbeat, logs, non-SQLite receipts, and file growth.  The observed first
attempt throughput implies roughly 16--18 days to 10k and 32--36 days to 20k;
the pre-registered 10k convergence gate may shorten this.

## Taste T14 / ComRecGC

- Protected science PID: `7224`, GPU2.
- Last audited progress: 5,400/20,000; committed checkpoint 5,000.
- Conservative ETA: 40--48 hours plus postprocess, potentially shorter only
  if the existing pre-registered convergence gate passes.

Do not restart, reconfigure, or inspect its active SQLite state.

## Operational rules and status

- No LLM or GNN ablation process is authorized or running.
- Do not query active T12/T14 SQLite databases.
- Do not launch T13 until the T8 scientific protocol decision is explicit.
- Each completed cell must append through the unique matrix authority; final
  Figure 3, Figure 4, and Table 2 follow only after 16/16.

Useful commands:

```bash
cat /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json
cat /autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t12-gcf-release-20260902T035837Z-7e5b1f3f/{state,heartbeat.json}
cat /autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t12-paper-after-generation-20260902T035900Z-1048144/{state,heartbeat.json}
tail -n 80 /autodl-fs/data/counterfactual-subgraph-runtime/logs/mut_trace_on_adoption/mut_semantic_finalizer_20260902T041411Z.log
cat /autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-t8-salvage-20260902T042219Z-da32d7e3/{state,heartbeat.json}
```
