# AutoDL Main Recovery and Ablation Framework Handoff

Last live audit: 2026-09-02 01:27-01:31 CST.  This document records live
execution facts separately from the post-main, config-only ablation design.
It must not be used as a substitute for the matrix authority or terminal
artifact receipts.

## 1. Main matrix authority

- Authority pointer: `/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json`
- Live complete cells: `10 / 16`
- Applied cells: AIDS/Ours, AIDS/GCFExplainer, AIDS/GlobalGCE;
  Mutagenicity/Ours, Mutagenicity/GCFExplainer, Mutagenicity/GlobalGCE;
  BACE/Ours, BACE/GCFExplainer, BACE/GlobalGCE, BACE/ComRecGC.
- Missing cells: AIDS/ComRecGC, Mutagenicity/ComRecGC, and all four
  TasteMolNet methods.
- No second matrix authority was created or modified by the framework build.

## 2. Live main-table workers

| Line | PID | GPU | Live state / checkpoint | Root |
|---|---:|---:|---|---|
| AIDS ComRecGC publication-only reconciliation | 28985 | none | Existing science PASS; validating the large immutable artifact inventory before append | `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/publication_reconciliation/aids_comrecgc_matrix_append_76a27fb_20260902T011500Z` |
| Mut ComRecGC trace-on adoption | 22325 | waiting | Worker alive; code audit and protected-throughput baseline PASS; waits for a naturally free main-table GPU | `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_trace_on_adoption_20260901T154900Z` |
| Taste T11 Ours | 15930 | 0 | `GENERATION_HIGH_TEMP`, 3699/3823 parents | `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/ours/t11-full/science-attempt-b73f789d-3888-4ec8-8e07-0442e224df29` |
| Taste T8 target-0 recovery (feeds T13) | 12751 | 1 | training epoch 19/25, best checkpoint and heartbeat present | `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/globalgce/t8-dual-branch-recovery/target-0-attempt-3af51e32-8429-4b3e-8bae-71ba16df1683` |
| Taste T14 ComRecGC | 7224 | 2 | generation 2100/20000, live progress writes | `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/comrecgc/t14-full/attempt-f3b2e5f2-9f20-4c12-bd26-3d7cc8e0d9ab` |
| Taste T12 GCFExplainer | 21562 | 3 | science process alive; no early production checkpoint yet | `/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/tastemolnet/gcfexplainer/t12-production/attempt-6616449a-6fa5-4502-8c8a-ae01b11366fb` |

The T12/T13/T14 publisher and post-process relays had current heartbeats at
the audit time.  T11, T8, T12 and T14 were not restarted, reconfigured or
preempted.  All four GPUs were owned by these main-table jobs; no ablation PID
or ablation GPU lock existed.

## 3. Main-table critical path

T11 is close to the end of its current generation stage and T8 target-0 has
six configured training epochs remaining, but both still require downstream
verification/publication.  T14 had 17,900 configured generation steps left;
its measured protected baseline was about 0.330 steps/second, corresponding to
roughly 15 hours for generation alone if sustained.  T12 remains the largest
uncertainty because its current implementation does not yet expose an early
progress checkpoint; its PID, GPU memory and process I/O are active, so it must
not be restarted merely to add observability.  Consequently no defensible
single 16/16 completion timestamp is available yet.  Natural completions must
feed the existing post-process and publisher queues immediately.

## 4. BACE/Ours main-reference contract

The framework freezes a machine-readable reference at:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/contracts/bace_ours_main_reference_v1.json`

The builder reads the real, already-published BACE/Ours cell from the current
matrix authority plus its GINE, temperature, feature schema,
split, MolCLR, WNode, PPO, candidate-pool, verification and selector artifacts.
Missing or mismatched provenance fails closed; values are not filled from
memory.  Building this reference is allowed before 16/16 because it does not
run science; the independent launch gate still requires final 16/16.  The
reference records the real 386-parent, 4+4 proposal contract.

The verified main policy lineage is ChemLLM base plus a fresh LoRA initializer
plus PPO.  There is no independently matched BACE SFT checkpoint.  Therefore:

- `BRICS_FIXED`: config-only, no checkpoint;
- `CHEMLLM_PRETRAINED`: available after exact base/tokenizer binding;
- `CHEMLLM_SFT`: `BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT`;
- `CHEMLLM_SFT_PPO`: `BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT` under the named
  four-stage comparison (the existing PPO policy must not be relabelled as
  SFT+PPO).

## 5. LLM proposer ablation contract

Dataset and method are fixed to BACE/Ours.  The primary budget is proposal
attempt matched: the same 386 train parents and the same main sampling regimes
are bound for every available variant.  Valid-candidate matching is a secondary
diagnostic only.

The BRICS vocabulary is built only from the frozen BACE train cohort.  BRICS
attachment dummies are removed to form canonical connected cores.  Records
contain fragment SMILES, train frequency, atom count, source-parent count and
rank; ordering is descending train frequency with canonical-SMILES tie-break.
For each proposal parent, only vocabulary fragments that actually match that
parent may be emitted.  A shortfall is recorded and candidates are never
duplicated or oracle-ranked.

All variants share the frozen parser, canonicalizer, direct-substructure and
projection checks, hard deletion, GINE oracle, WNode matrix, calibration-only
selector and held-out-test evaluator.  Planned candidate, novelty, coverage,
cost, diversity, resource and parent-bootstrap metrics are schemas only until
an authorized science run writes PASS manifests.

## 6. GNN backbone ablation contract

The primary framework is BACE/Ours proposal-fixed with GINE (reference), GIN,
GCN and GATv2.  It freezes the main candidate identities and budget, then
recomputes classifier-dependent source cohorts, strict flips, WNode matrices,
calibration selection and final test evaluation per backbone.  Candidate
generation and ChemLLM are not rerun in proposal-fixed mode.

All models bind the same atom/bond feature schema and explicitly disclose how
edge features enter messages.  Model training uses train, checkpoint selection
and temperature fitting use validation, calibration is reserved for rule
selection, and test is opened only after all model and selector freezes.
Classifier metrics are ROC-AUC, balanced accuracy, macro-F1, NLL, ECE, Brier
and parameter count.  Explanation metrics are reported on both each model's
native correctly classified cohort and the common intersection; common-cohort
stability is primary.

The optional BACE/Ours end-to-end mode is config-only.  It may eventually rerun
PPO, generation and selection, but no end-to-end science runner is started by
this framework.

## 7. Launch gates and current state

A future run requires all of the following, bound to the same matrix-authority
root, matrix SHA, and combined-audit SHA:

1. the exact canonical 16-cell main matrix and final combined audit PASS;
2. self-hashed PASS receipts for the final audit and final four-dataset
   Figure 3, Figure 4, and Table 2 artifacts;
3. a self-hashed project-owner run-authorization receipt bound to the family,
   execution commit, run-contract SHA, authority, and all four receipt SHAs;
4. both the tracked family flag and the corresponding operator run request.

At handoff, `RUN_LLM_ABLATION=0`, `RUN_GNN_ABLATION=0`, and no valid
authorization receipt exists.  No ablation science has started.  A boolean or
environment variable is not authorization.  Checked-in Slurm wrappers exist
solely because repository policy requires a paired wrapper for Python
entrypoints; they were not submitted and must not be submitted while
main-table GPUs are occupied.

## 8. Paper staging

Claim-safe, number-free templates are written under:

`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_staging/ablations_v1/`

They contain TODO fields and disclose unavailable matched-SFT variants.  They
must not be copied into the paper as result claims until hash-bound PASS
aggregate manifests exist.

## 9. Suggested future run order

Only after the gates above and a new explicit authorization:

1. BACE LLM proposer ablation (available variants; blocked variants remain
   blocked);
2. BACE GNN proposal-fixed ablation;
3. BACE GNN end-to-end ablation;
4. optional TasteMolNet GNN proposal-fixed ablation;
5. optional TasteMolNet LLM ablation.

## 10. Status commands

From the deployed immutable framework worktree on AutoDL:

```bash
/root/miniconda3/envs/smiles_pip118/bin/python scripts/autodl/status_llm_ablation.py \
  --config configs/hpc.yaml \
  --common-config configs/ablations/common_v1.yaml \
  --family llm \
  --matrix-authority /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json

/root/miniconda3/envs/smiles_pip118/bin/python scripts/autodl/status_gnn_ablation.py \
  --config configs/hpc.yaml \
  --common-config configs/ablations/common_v1.yaml \
  --family gnn \
  --matrix-authority /autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json
```

For main status, read the matrix pointer and the exact controller heartbeats;
do not infer completion from PID exit alone.  Do not restart T12 merely because
it lacks an early progress file.
