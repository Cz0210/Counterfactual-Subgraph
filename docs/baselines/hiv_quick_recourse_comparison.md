# HIV Quick Recourse And CAMC Comparison

This document describes the quick two-layer comparison between the current
HIV/SMILES selector output, a simple opposite-label full-graph greedy baseline,
and optional SMILES full-graph baselines such as a GCF-HIV adapter.

## Scope

This is a quick comparison under the current HIV/SMILES + RF teacher setting. It
is not the official GCFExplainer reproduction. The official AIDS graph baseline
remains documented separately in `docs/baselines/gcfexplainer_reproduction.md`.

The evaluator now reports two complementary views:

- recourse-level comparison: answers whether a method can provide a low-cost
  counterfactual recourse `G_i'` for each input molecule `G_i`;
- action-motif-level CAMC comparison: answers whether a method provides shared,
  effective, low-redundancy counterfactual action motifs.

The recourse-level view puts two very different candidate types onto one
per-input interface:

- `ours_selected_subgraph`: selected class-level fragments from our selector;
- `gt_fullgraph_greedy`: full molecules greedily selected from opposite-label
  HIV examples.

For every target-label input molecule `G_i`, each method attempts to produce a
recourse candidate `G_i'` whenever possible. The same RF teacher then scores
`p_before`, `p_after`, `cf_drop`, and `cf_flip`.

## Distance And Metrics

The comparison uses an RDKit MCS proxy distance:

```text
proxy_edit =
  (atoms1 - mcs_atoms)
+ (atoms2 - mcs_atoms)
+ (bonds1 - mcs_bonds)
+ (bonds2 - mcs_bonds)

normalized_proxy =
  proxy_edit / max(1, atoms1 + atoms2 + bonds1 + bonds2)
```

This is not the paper's exact or learned graph edit distance. Treat it as a
fast, approximate recourse-cost proxy for our SMILES/RDKit setting.

Primary recourse-level metrics:

- `coverage`: fraction of target inputs with a valid recourse that flips the RF
  teacher and has proxy distance at most `theta`;
- `coverage_theta`: explicit alias for the theta-constrained coverage above;
- `coverage_unconstrained_flip`: fraction of target inputs with any valid
  flipping recourse before applying the theta constraint;
- `median_cost` / `mean_cost`: proxy cost among covered recourses;
- `median_cost_covered_only`: covered-only median cost;
- `mean_cf_drop`: mean drop in target-label RF probability among valid
  recourses;
- `mean_cf_drop_covered_only`: covered-only mean probability drop;
- `flip_rate`: fraction of target inputs whose valid recourse flips the RF
  teacher;
- `valid_recourse_rate`: fraction of target inputs where the method produced a
  parseable/scorable recourse.

The evaluator also reports `ours_substructure_match_rate_by_k` in
`comparison_summary.json`. That is an internal ours-style diagnostic. It should
not be hard-compared directly against the full-graph baseline because the
candidate objects are different.

## Method Definitions

### Ours: Selected Subgraph

The evaluator loads selected fragments from `--ours-selected-dir`, looking for:

```text
selector_summary.json
selected_subgraphs.json
selected_subgraphs.csv
selector_report.txt
```

It tries compatible fragment fields such as `final_fragment`, `core_fragment`,
`fragment_smiles`, and `fragment`.

For each target-label input molecule, the evaluator first expands all selected
fragments up to the requested max `k` into `ours_action_candidates.csv`.
Each action records:

```text
input_idx, parent_smiles, fragment_rank, fragment, match_ok, delete_ok,
valid_after, distance_proxy, p_before, p_after, cf_drop, cf_flip,
failure_reason
```

For each `k` and `theta`, coverage is theta-aware existential coverage:

```text
covered_i(theta) =
  exists action a among top-k fragments such that
    a.match_ok
    and a.delete_ok
    and a.valid_after
    and a.cf_flip
    and a.distance_proxy <= theta
```

For covered inputs, cost is the minimum feasible action distance and cf-drop is
the maximum feasible action probability drop. This fixes the earlier failure
mode where choosing one max-drop action before applying theta could make
coverage decrease when `k` increased.

For each target-label input molecule and each `k`, the old per-input CSV is
still emitted, but it is now an aggregate view over the top-`k` action set.

### GT-Fullgraph Greedy

The candidate pool is made from opposite-label HIV molecules, capped by
`--max-gt-candidates` and sampled reproducibly with `--seed` when needed.

For each selected full molecule candidate `C`, target input `G` is covered when:

```text
teacher_flip(G -> C) and distance_proxy(G, C) <= theta
```

The evaluator uses one greedy order selected at the maximum requested `theta`
and evaluates all theta values against prefixes of that same order. This keeps
the reported theta-aware existential coverage monotone in both `k` and `theta`.

## CAMC: Action-Motif-Level Comparison

Counterfactual Action Motif Coverage (CAMC) compares methods after converting
their outputs into an action motif set `M`.

This is more favorable to our method because our selector naturally outputs
selected subgraphs, so the action motif is exactly the selected fragment.
However, CAMC is still applicable to GCFExplainer or another full-graph
baseline: a full-graph counterfactual implicitly defines an edit action from
`G` to `C`, and the deleted motif can be extracted with an MCS difference.

### Action Motifs

For `ours_selected_subgraph`, action motifs are the selected fragments in their
existing selector order. The evaluator writes:

```text
camc_ours_action_motifs.csv
```

For `gt_fullgraph_greedy`, the evaluator:

1. takes the selected full molecules;
2. finds the nearest selected fullgraph `C*(G_i)` for each input `G_i`;
3. computes `MCS(G_i, C*(G_i))`;
4. extracts the deleted motif `G_i - MCS(G_i, C*(G_i))`;
5. keeps the largest connected component when the deleted motif is disconnected;
6. canonicalizes and filters motifs smaller than `--camc-min-motif-atoms`;
7. deduplicates the motif pool;
8. greedily/MMR-selects top-k action motifs for CAMC evaluation.

The evaluator writes:

```text
camc_gt_fullgraph_motif_pool.csv
camc_gt_fullgraph_selected_motifs.csv
```

Extra fullgraph baselines can be supplied with:

```bash
--extra-fullgraph-selected-csv method_name:/path/to/selected_fullgraphs.csv
```

The CSV must contain one of:

```text
smiles
counterfactual_smiles
selected_smiles
fullgraph_smiles
```

Optional columns include `rank`, `score`, `coverage`, `cost`, and `method`.
Current HIV/SMILES CAMC evaluation supports only fullgraph SMILES. Official
AIDS graph benchmark GCF output must either be evaluated by a graph-level CAMC
evaluator or converted by a GCF-HIV adapter that emits selected fullgraph
SMILES. If a graph benchmark file is passed directly, the evaluator fails with
a clear error instead of treating graph IDs as SMILES.

### CAMC Metrics

For each method and top-k action motif set `M_k`, CAMC reports:

- `support_coverage`: fraction of target inputs where at least one action motif
  is a substructure;
- `camc_flip_coverage`: fraction of target inputs where at least one motif can
  be deleted, produces a valid residual, and flips the RF teacher;
- `camc_delta_{delta}`: fraction of target inputs where at least one motif
  produces a target-label probability drop greater than `delta`;
- `mean_cf_drop_all_matched`;
- `mean_cf_drop_covered`;
- `motif_atom_count_mean`;
- `motif_atom_ratio_mean`;
- pairwise Morgan fingerprint Tanimoto mean and max.

CAMC writes `camc_comparison_table.csv`, `camc_summary.json`, and
`camc_per_input.csv`. `camc_summary.json` and `diagnostic_counts.json` also
include `motif_overlap_diagnostics`, which compares ours and GT selected motifs
by exact overlap, max Tanimoto similarity, atom counts, aromatic motifs, and
hetero-atom motifs.

The evaluator uses RDKit's MorganGenerator API for motif fingerprints. It also
defaults to `--suppress-rdkit-warnings` as a log fallback; pass
`--no-suppress-rdkit-warnings` only when debugging RDKit warning output.

## Local Command

Example smoke run:

```bash
python scripts/eval/compare_hiv_recourse_baselines.py \
  --hiv-csv data/raw/AIDS/HIV.csv \
  --teacher-path outputs/hpc/oracle/aids_rf_model.pkl \
  --target-label 1 \
  --ours-selected-dir outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20 \
  --top-k-list 10 20 \
  --theta-list 0.05 0.10 0.15 0.20 \
  --max-inputs 200 \
  --max-gt-candidates 1000 \
  --out-dir outputs/local/hiv_quick_recourse_compare_label1 \
  --seed 13 \
  --progress-every 100 \
  --enable-camc \
  --camc-delta-list 0.1 0.2 0.3 0.5
```

Output files:

```text
comparison_summary.json
comparison_table.csv
ours_per_input.csv
gt_fullgraph_per_input.csv
gt_selected_fullgraphs.csv
ours_action_candidates.csv
diagnostic_counts.json
progress.log
run_config.json
camc_comparison_table.csv
camc_summary.json
camc_per_input.csv
camc_ours_action_motifs.csv
camc_gt_fullgraph_motif_pool.csv
camc_gt_fullgraph_selected_motifs.csv
```

## HPC Command

The label-1 Slurm wrapper uses `smiles_pip118` and writes to:

```text
outputs/hpc/comparison/hiv_quick/label1_${SLURM_JOB_ID}/
```

Submit with an explicit HIV CSV if auto-discovery is ambiguous:

```bash
cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs
HIV_CSV=/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv \
  sbatch scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh
```

Optional quick-run overrides:

```bash
MAX_INPUTS=300 \
MAX_GT_CANDIDATES=1000 \
PROGRESS_EVERY=100 \
ENABLE_CAMC=true \
TOP_K_LIST="10 20" \
THETA_LIST="0.05 0.10 0.15 0.20" \
HIV_CSV=/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv \
  sbatch scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh
```

Inspect results:

```bash
OUT=outputs/hpc/comparison/hiv_quick/label1_${JOB_ID}
cat "${OUT}/comparison_table.csv"
cat "${OUT}/camc_comparison_table.csv"
cat "${OUT}/comparison_summary.json"
cat "${OUT}/diagnostic_counts.json"
tail -f "${OUT}/progress.log"
grep -c "DEPRECATION WARNING: please use MorganGenerator" "${OUT}/progress.log" || true
export OUT
python - <<'PY'
import json
import os
from pathlib import Path
out = Path(os.environ["OUT"])
print(json.dumps(json.loads((out / "comparison_summary.json").read_text())["recourse_monotonicity_warnings"], indent=2))
print(json.dumps(json.loads((out / "camc_summary.json").read_text())["camc_monotonicity_warnings"], indent=2))
PY
```
