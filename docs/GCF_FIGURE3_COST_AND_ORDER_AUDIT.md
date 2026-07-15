# GCF Figure 3 Cost and Order Audit

## 1. Executive conclusion

The flat GCFExplainer lower curve is primarily explained by a confirmed metric-definition
mismatch. The current default Figure 3 value is the median best distance among parents
covered at the current theta, whereas the paper-style recourse cost is the median prefix
minimum over all parents. The covered-parent population changes with K, so the former is
not required to decrease even though every individual parent's prefix minimum cannot
increase.

No prefix/order bug is visible in the report implementation: it loads the external rank,
uses candidates with `rank <= K`, and takes a cumulative minimum. Exact alignment of the
HPC metadata, converted SMILES, ad-hoc `top20_for_fgw.csv`, and `pair_details.csv` remains
an artifact-level check because those ignored HPC outputs are not present in the local
checkout. Run `scripts/slurm/audit_gcf_prefix_cost_and_order.sh` to complete that check.

The current HIVCSV search is a lightweight adapted implementation, not an official
GCFExplainer reproduction. In particular, `alpha` and `teleport` are parsed and recorded
but do not influence its search loop.

## 2. Paper cost vs current cost

For an externally ranked prefix \(C_{1:K}\), define the strict-valid prefix distance:

```text
best_K(G) = min d(G, C_j)
            over j <= K with teacher-strict flip
```

The paper-style all-parent statistic requested for this audit is:

```text
unconditional_all_parent_median_cost(K)
    = median over every parent G of best_K(G)
```

Unavailable parents contribute `+inf`. Since `C_1` through `C_K` are nested,
`best_(K+1)(G) <= best_K(G)` for every parent, and this median must be monotone
non-increasing.

The current default lower panel instead uses:

```text
theta_covered_conditional_median_cost(K, theta)
    = median { best_K(G) | best_K(G) <= theta }
```

The conditioning set can gain new parents as K increases. Those new parents may have
larger close distances than the previously covered population, so this statistic may be
flat, increase, or fluctuate.

Code evidence:

- `src/eval/gcf_style_recourse_report.py:630` computes prefix metrics. Lines 631-639
  separate all-parent, finite-applicable, and theta-covered populations; lines 651-658
  expose `median_cost`, `applicable_parent_median_cost`, and
  `theta_covered_conditional_median_cost`.
- `src/eval/gcf_style_recourse_report.py:67` makes the theta-covered statistic the
  default Figure 3/Table 2 cost metric.
- `src/eval/gcf_style_recourse_report.py:676` copies the configured field to
  `plotted_cost`; `src/eval/gcf_style_recourse_report.py:1017` plots that field.
- `src/eval/gcf_style_recourse_report.py:876` retains the full audit Table 2. Its legacy
  `Conditional Median Cost` is the median over all finite strict-recourse-applicable
  parents, not the theta-covered subset.
- `src/eval/gcf_style_recourse_report.py:1235` builds the compact final table from the
  configured cost metric. The CLI default at line 1603 is theta-covered conditional
  cost.

Thus the current Figure 3 and compact Table 2 do filter to strict-flip parents with
`distance <= theta` before their default cost median. The set of contributing parents may
change at every K. NaN/inf values are not converted to zero: unavailable parents remain
`inf` in `median_cost`, while the two conditional summaries intentionally exclude them.

Figure 4 uses the same strict-valid prefix distances but does not use either median-cost
field. `src/eval/gcf_style_recourse_report.py:1375` constructs one shared absolute
threshold grid and explicitly inserts `theta_star`; lines 1390-1400 obtain the fixed-K
prefix minimum once per method and bootstrap parents on that vector. Therefore Figure 4
is a coverage-vs-threshold view. Its monotonicity follows from thresholding fixed best
distances and is independent of the Figure 3 cost-statistic choice.

## 3. Prefix metric audit

The report implementation does not reselect candidates for each K:

- `src/eval/gcf_style_recourse_report.py:283` loads the external selector output.
- Lines 293-335 use `rank`, `selection_rank`, or `candidate_rank`, then validate exactly
  `1..20`; row order is only a documented fallback.
- `src/eval/gcf_style_recourse_report.py:598` selects IDs with `rank <= K`, initializes
  every parent to `+inf`, and updates by minimum distance.
- `src/eval/gcf_style_recourse_report.py:668` evaluates K=1 through K=20 and asserts that
  coverage is non-decreasing and all-parent median cost is non-increasing.

The independent audit repeats this from saved pair details without computing distances.
It also computes `raw_distance_best_K` without a strict-flip condition. This separates a
distance/order defect from effects introduced by the counterfactual validity filter.

The independent output `prefix_metrics_audit.csv` contains both:

- `unconditional_median_best_distance_all_parents`;
- `conditional_median_best_distance_theta_covered`.

It additionally records finite-applicable median, all-parent mean and quartiles, missing
distances, NaN rows, and the raw-distance diagnostic.

## 4. Candidate order audit

The committed code establishes this order up to the point where the current ad-hoc
Top20 artifact was created:

1. `scripts/gcf_hiv_csv_export_summary.py:62` runs deterministic native greedy selection
   with key `(marginal coverage gain, frequency, -min_distance_seen)`.
2. `scripts/gcf_hiv_csv_export_summary.py:137` stores `selected_graphs` and
   `selected_records` from that same list. Lines 155-175 write metadata in that list order.
3. `scripts/convert_gcf_hiv_csv_graphs_to_smiles.py:78` zips `selected_graphs` with
   `selected_records` by list index. Lines 118-137 convert and append without sorting.
4. No committed script references the filename
   `gcf_hiv_csv_alpha0.5_selected_smiles_top20_for_fgw.csv`. Its exact construction cannot
   be inferred safely from the repository.
5. The newer `scripts/gcf_hiv_csv_export_valid_greedy_topk.py:149` filters validity before
   greedy selection, and lines 212-249 assert graph/metadata/SMILES order alignment. Its
   official output names are `valid_greedy_top20_*`, so it is not automatically proven to
   be the producer of the older Top20 file.
6. `scripts/evaluate_ccrcov_with_molclr_node_fgw.py:839` loads the supplied fullgraph CSV
   and passes its candidate list directly to `_evaluate_gt_fullgraph`.
7. `src/eval/ccrcov_distance_eval.py:624` iterates parent then candidate, comparing each
   parent SMILES directly with each complete candidate SMILES. It does not use fragment
   matching or deletion for this path.

The audit output `candidate_order_audit.csv` aligns selected metadata, valid converted
SMILES, Top20 prefix, and first candidate occurrence in pair details by `candidate_id`,
then canonical SMILES as a fallback. Original selected ranks may contain gaps after
sanitize filtering; relative order must be preserved, while valid-filtered rank,
Top20 rank, and pair-details index must agree exactly.

It explicitly detects lexical-ID reordering (`1,10,...,2`), 0/1-based rank mistakes,
missing candidates, canonical-SMILES duplicates, graph-hash duplicates when available,
and incomplete parent-by-candidate matrices.

## 5. Selection/evaluation mismatch

The HIVCSV adapted search uses `graph_distance_proxy` in
`scripts/gcf_hiv_csv_run_vrrw.py:130`. It sums normalized node-count difference,
symmetric edge difference, and node-label-histogram difference. The training threshold
`theta=0.05` is applied to that proxy at lines 251-275.

The unified final evaluation uses MolCLR-Node-FGW with `lambda=0.5` and theta `0.0328`.
Therefore:

- native `covered_indices` are proxy-distance coverage sets;
- `min_distance_seen` is a native proxy distance;
- neither is an FGW distance;
- the native greedy rank is not optimized for unified FGW coverage or cost.

This metric mismatch is expected to produce plateaus or late gains under FGW and does not
by itself indicate a plotting error.

The earlier zero-value bug is already fixed:
`scripts/gcf_hiv_csv_export_summary.py:52` explicitly distinguishes missing/NaN values
from `0.0`, so a real zero is no longer replaced with `999.0`.

## 6. Adaptation fidelity

The current adapted loop in `scripts/gcf_hiv_csv_run_vrrw.py:239` independently samples a
target parent, samples simple mutations, keeps the nearest valid counterfactual for that
step, increments frequency, and updates only that parent's native coverage. It does not
implement a reinforced walk over a transition graph.

Confirmed omissions relative to official GCFExplainer:

- `alpha` is parsed at line 195 and only written to config at line 345;
- `teleport` is parsed at line 198 and only written to config at line 348;
- there is no `p_phi`, individual/cumulative coverage mixture in transition importance,
  visit-frequency transition reinforcement, or dynamic teleportation in the adapted loop.

By contrast, the untouched official source:

- combines individual and cumulative coverage using alpha in
  `baselines/gcfexplainer_official/vrrw.py:259`;
- reinforces transition probabilities by visit frequency at line 284;
- performs teleportation and neighbor transitions at line 303;
- performs dynamic restart selection at line 363;
- obtains coverage using a NeuRoSED threshold in
  `baselines/gcfexplainer_official/importance.py:70`.

The current path should therefore be reported as **GCFExplainer-HIVCSV lightweight
adapted implementation**, not as an official-native algorithm reproduction.

## 7. Confirmed bugs

1. **No current prefix implementation bug is confirmed by code.** Prefix rank and
   cumulative minimum logic are correct and guarded by monotonicity assertions.
2. **Metric-definition mismatch is confirmed.** The plotted default is theta-covered
   conditional median, not the paper-style all-parent median.
3. **Adaptation provenance bug is confirmed if alpha is described as active.** Alpha and
   teleport are configuration-only in the current HIVCSV search.
4. **The old Top20 artifact lacks a committed producer.** This is a reproducibility gap,
   not proof that its runtime order is wrong.

Whether the concrete HPC artifacts contain an order mismatch is intentionally left to the
read-only artifact audit rather than guessed from source code.

## 8. Non-bug explanations

- Theta-covered conditional cost may remain around 0.028-0.029 because newly covered
  parents enter the median population as K grows.
- Native greedy gain is based on proxy-distance coverage, not FGW coverage.
- Native frequency and proxy closeness need not rank candidates by FGW cost.
- Strict-flip filtering can leave a different method-specific parent subset at each K.
- Invalid native candidates may have influenced the old greedy covered set if validity was
  checked only after native selection.

The last point depends on artifact provenance. The old flow “greedy all candidates, then
sanitize, then first 20 valid” differs from “sanitize all candidates, rerun greedy on valid
candidates, then take Top20”. In the former, invalid candidates can update `covered` inside
`scripts/gcf_hiv_csv_export_summary.py:83`, changing later valid-candidate gains. The newer
valid-first exporter prevents this, but it must not be assumed to have produced an older
differently named file.

## 9. Recommended fixes

**P0**

- Run the independent audit on the actual HPC files and require
  `candidate_order_exact_match=true`, `prefix_is_nested=true`, and
  `all_parents_have_fullgraph_pairs=true` before interpreting the curve.
- Report both cost statistics. Use `unconditional_median_best_distance_all_parents` when
  making a paper Figure 3 semantic claim; retain theta-covered conditional median as a
  clearly named CCRCov companion metric.

**P1**

- Record the exact producer command for every Top20 artifact.
- For a validity-constrained experiment, use the existing valid-first greedy exporter and
  keep its output separate from the historical result.

**P2**

- Keep the lightweight HIVCSV adaptation clearly labeled.
- If official algorithm fidelity is required, implement or invoke the official transition,
  alpha importance, reinforcement, and teleportation semantics in a separate run line.

## 10. Commands to reproduce the audit

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs

sbatch scripts/slurm/audit_gcf_prefix_cost_and_order.sh
```

To compare directly with an existing Figure 3 CSV:

```bash
export FIGURE3_CSV=outputs/hpc/eval/paper/molclr_node_fgw_gcf_style_teacher_strict_ref1283_v4/figure3_fgw_coverage_cost_vs_k.csv
export OUTPUT_DIR=outputs/hpc/audits/gcf_prefix_cost_and_order_with_plot
sbatch --export=ALL scripts/slurm/audit_gcf_prefix_cost_and_order.sh
```

The audit writes only to its new output directory:

```text
prefix_metrics_audit.csv
candidate_order_audit.csv
audit_summary.json
audit_report.txt
```
