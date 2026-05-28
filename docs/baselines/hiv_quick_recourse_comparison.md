# HIV Quick Recourse Comparison

This document describes the quick recourse-level comparison between the current
HIV/SMILES selector output and a simple opposite-label full-graph greedy
baseline.

## Scope

This is a quick comparison under the current HIV/SMILES + RF teacher setting. It
is not the official GCFExplainer reproduction. The official AIDS graph baseline
remains documented separately in `docs/baselines/gcfexplainer_reproduction.md`.

The purpose here is to put two very different candidate types onto one
recourse-level interface:

- `ours_selected_subgraph`: selected class-level fragments from our selector;
- `gt_fullgraph_greedy`: full molecules greedily selected from opposite-label
  HIV examples.

For every target-label input molecule `G_i`, each method must produce a recourse
candidate `G_i'` whenever possible. The same RF teacher then scores
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
- `median_cost` / `mean_cost`: proxy cost among covered recourses;
- `mean_cf_drop`: mean drop in target-label RF probability among valid recourses;
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

For each target-label input molecule and each `k`:

1. Keep the top-`k` selected fragments.
2. Check whether each fragment is a substructure of the input.
3. Delete matched fragments to construct residual molecules.
4. Score residuals with the RF teacher.
5. Choose the valid residual with maximum `cf_drop` as `G_i'`.

### GT-Fullgraph Greedy

The candidate pool is made from opposite-label HIV molecules, capped by
`--max-gt-candidates` and sampled reproducibly with `--seed` when needed.

For each `theta`, each full molecule candidate `C` covers target input `G` when:

```text
teacher_flip(G -> C) and distance_proxy(G, C) <= theta
```

The evaluator greedily selects candidates by marginal coverage gain up to the
maximum requested `k`. For a particular top-`k` result, each input uses the
nearest selected full molecule as `G_i'`.

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
  --seed 13
```

Output files:

```text
comparison_summary.json
comparison_table.csv
ours_per_input.csv
gt_fullgraph_per_input.csv
gt_selected_fullgraphs.csv
run_config.json
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
TOP_K_LIST="10 20" \
THETA_LIST="0.05 0.10 0.15 0.20" \
HIV_CSV=/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv \
  sbatch scripts/slurm/gcfexplainer/run_hiv_quick_recourse_compare_label1.sh
```

Inspect results:

```bash
OUT=outputs/hpc/comparison/hiv_quick/label1_${JOB_ID}
cat "${OUT}/comparison_table.csv"
cat "${OUT}/comparison_summary.json"
```
