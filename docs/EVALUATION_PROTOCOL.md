# Evaluation Protocol: Native-action CCRCov for Global Counterfactual Subgraph Explanation

## 1. Motivation

The baseline space in this project is heterogeneous. Different methods may output:

- class-level counterfactual subgraphs;
- full counterfactual graphs;
- graph transformation rules;
- local edge deletion actions;
- local counterfactual graphs;
- molecule fragment masks;
- addition / replacement edits.

Because of this heterogeneity, cross-baseline evaluation should not force every method into subgraph matching coverage, and it should not force every method into a deletion-only intervention. The current protocol adopts a GCF-style close counterfactual coverage idea and evaluates every method by the low-cost counterfactual effect of its native action, rule, or counterfactual graph on the original input graph.

SuppCov is temporarily reserved for subgraph-specific auxiliary analysis. It is not the current main cross-baseline metric. In future work, SuppCov may still be useful for methods whose outputs are explicitly subgraphs, fragments, or rule left-hand sides, but it should not be used as the final fair comparison metric across all baseline families.

For AIDS/HIV experiments, every final CCRCOV / CFDrop / FlipRate / Cost / Redundancy table must follow `docs/DATASET_CONTRACT.md`. The canonical source is `data/raw/AIDS/HIV.csv` with `SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`, and `TARGET_LABEL=1`. Legacy `hiv` / `hiv_quick` names and graph-baseline `aids` keys are internal names for that same source; `ogbg_molhiv` is engineering validation only.

## 2. Main Metric: CCRCov

Given a selected set of global explanations or actions

$$A_K = \{a_1, \ldots, a_K\},$$

an input graph $G_i$ with class label $y$, and a frozen teacher classifier $\phi$, each action produces an intervened graph:

$$G_i^a = T_a(G_i).$$

Here $T_a$ is the native intervention defined by the method. For Ours this is hard deletion of a selected subgraph. For other baselines it may be a full counterfactual graph choice, a graph transformation rule, an addition edit, a replacement edit, an edge deletion, or another method-native action.

The main metric is Close Counterfactual Rule Coverage:

$$
\mathrm{CCRCov}_{\theta}@K =
\frac{
\left|\left\{G_i \in D_y :
\exists a \in A_K,\,
\mathrm{valid}(G_i^a),\,
d(G_i, G_i^a) \le \theta,\,
\phi(G_i^a) \ne y
\right\}\right|
}{
|D_y|
}.
$$

Definitions:

- $D_y$ is the evaluation subset with class label $y$.
- $T_a$ is the native intervention defined by the method.
- $d$ can be normalized GED or teacher embedding distance.
- $\theta$ is the cost threshold.
- $\phi$ is the frozen dataset-level teacher.
- $\mathrm{valid}(G_i^a)$ means the intervened graph is parseable and acceptable under the evaluation pipeline.

The relaxed drop-based variant is:

$$
\mathrm{CCRCovDrop}_{\theta,\delta}@K =
\frac{
\left|\left\{G_i \in D_y :
\exists a \in A_K,\,
\mathrm{valid}(G_i^a),\,
d(G_i, G_i^a) \le \theta,\,
p_\phi(y \mid G_i) - p_\phi(y \mid G_i^a) \ge \delta
\right\}\right|
}{
|D_y|
}.
$$

The main result table should prioritize flip-based CCRCov. Drop-based CCRCov is reported as a complementary stability and strength metric, especially when teacher probabilities provide useful signal but hard label flips are sparse.

## 3. Cost / Distance

Two cost functions are reported in parallel.

### A. Normalized GED

$$
\mathrm{cost}_{GED}(G, G^a) =
\frac{\mathrm{GED}(G, G^a)}
{|V_G| + |V_{G^a}| + |E_G| + |E_{G^a}|}.
$$

GED includes:

- node addition;
- node deletion;
- edge addition;
- edge deletion;
- node label change;
- edge label change.

For Ours, which is a hard-deletion method by design, the fast deletion edit cost can be used as a valid upper-bound style implementation: removed nodes plus removed incident edges, normalized by the same denominator. For addition / replacement baselines, the evaluation must use full edit cost or approximate GED; these methods must not be reduced to deletion ratio only.

### B. Embedding Distance

$$
\mathrm{cost}_{emb}(G, G^a) =
1 - \cos\left(h_\phi(G), h_\phi(G^a)\right).
$$

Here $h_\phi$ is the frozen teacher's graph-level embedding. Embedding distance is a parallel evaluation track and should be reported together with GED whenever the teacher exposes a stable graph embedding layer.

Default GED thresholds:

```text
0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20
```

Default embedding distance thresholds:

```text
0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
```

The key reporting thresholds are:

- $\mathrm{CCRCov}@0.10$;
- $\mathrm{CCRCov}@0.20$.

The $0.20$ threshold aligns with the earlier low-cost flip coverage@0.20 protocol. Under Ours, when the candidate counterfactual graph is the hard-deletion residual $G \setminus s$, distance is normalized GED or embedding distance, and a teacher flip is required, low-cost flip coverage@0.20 is a deletion-action instance of $\mathrm{CCRCov}@0.20$.

## 4. Auxiliary Metrics

Auxiliary metrics are reported to explain why a method succeeds or fails under CCRCov. They should not override the main native-action CCRCov table.

### CFDrop

$$
\mathrm{CFDrop}(A_K) =
\mathrm{mean}_{G_i}
\max_{a \in A_K}
\left[p_\phi(y \mid G_i) - p_\phi(y \mid G_i^a)\right],
$$

computed over covered graphs or over applicable valid interventions, depending on the table context.

### FlipRate

$$
\mathrm{FlipRate}(A_K) =
\frac{
\left|\left\{G_i : \exists a,\ \phi(G_i^a) \ne y\right\}\right|
}{
\left|\left\{G_i : \exists a,\ \mathrm{valid}(G_i^a)\right\}\right|
}.
$$

### Cost

Report mean and median minimum cost among covered graphs:

$$
\min_{a \in A_K:\ a\ \mathrm{covers}\ G_i} d(G_i, G_i^a).
$$

### Coverage Redundancy

For each action $a$, define its covered set:

$$
C(a) = \{G_i : a\ \mathrm{covers}\ G_i\ \mathrm{under\ the\ CCRCov\ condition}\}.
$$

Coverage redundancy is:

$$
\mathrm{Red}_{cov}(A_K) =
\frac{2}{K(K-1)}
\sum_{i < j}
\mathrm{Jaccard}\left(C(a_i), C(a_j)\right).
$$

Lower coverage redundancy is preferred when coverage is comparable.

### ValidRate

$$
\mathrm{ValidRate} =
\frac{\mathrm{number\ of\ valid\ intervened\ graphs}}
{\mathrm{number\ of\ attempted\ interventions}}.
$$

### DeleteValidRate

DeleteValidRate is defined only for deletion-based methods. It measures whether hard deletion produces a parseable and acceptable residual graph.

### StructRed

StructRed is defined only for subgraph / fragment / rule methods. It may use Morgan Tanimoto similarity, graph similarity, or another explicitly documented structural similarity function.

SuppCov, StructRed, MatchRate, and DeleteValidRate are subgraph-specific / deletion-specific auxiliary metrics. They do not enter the main fair comparison table across all baselines.

## 5. Top-K Protocol

All methods should report:

$$K \in \{1, 5, 10, 20\}.$$

If a method naturally outputs top-K global explanations, actions, or rules, use those outputs directly.

If a method only outputs one local counterfactual result per input graph, use:

```text
local outputs -> candidate pool -> canonicalization -> deduplication -> greedy top-K
```

The default greedy selection objective is marginal CCRCov gain:

$$
a^* =
\arg\max_a
\Delta \mathrm{CCRCov}_\theta(a \mid A).
$$

An optional redundancy-penalized variant may be used for ablation:

$$
\mathrm{score}(a \mid A) =
\Delta \mathrm{CCRCov}_\theta(a \mid A)
- \lambda \cdot \max\_sim(a, A).
$$

The main fair comparison should prioritize greedy CCRCov gain without a redundancy penalty unless the experiment explicitly studies redundancy-aware selection.

## 6. Teacher Protocol

Each dataset uses exactly one frozen teacher classifier $\phi_D$ for evaluation.

Within the same dataset:

- label=0 and label=1 use the same teacher;
- all baselines use the same teacher;
- all methods are evaluated on the same split;
- the teacher is trained on the train split;
- the validation split is used for teacher checkpoint selection;
- the test split is reserved for final evaluation.

If a baseline needs its own internal classifier to generate candidates, that is allowed as a method-specific generation detail. However, the final reported CCRCov, CFDrop, FlipRate, and Cost must be recomputed by the unified dataset-level teacher $\phi_D$.

The dataset-level teacher registry should record:

- dataset name;
- split id;
- teacher architecture;
- checkpoint path;
- train accuracy;
- validation accuracy;
- test accuracy;
- label mapping;
- graph preprocessing pipeline;
- embedding layer used for embedding-distance evaluation.

## 7. Main Result Table

This is the main fair comparison table. SuppCov is intentionally excluded because not all baselines output subgraphs.

| Method | Output Type | K | CCRCov@0.10 ↑ | CCRCov@0.20 ↑ | CFDrop ↑ | FlipRate ↑ | Cost ↓ | CoverageRed ↓ | ValidRate ↑ |
| ------ | ----------- | -: | ------------: | ------------: | -------: | ---------: | -----: | ------------: | ----------: |

All entries in this table must be recomputed using the shared dataset split, shared frozen teacher, shared K, shared threshold, and shared cost function.

## 8. Deletion-only Auxiliary Table

This table is only for deletion-compatible methods:

- Ours;
- CF-GNNExplainer;
- SME / BRICS-Mask;
- projected deletion part of CLEAR;
- projected deletion part of RLHEX;
- projected deletion part of GCFExplainer, if extractable.

| Method | K | Deletion-CCRCov@0.20 ↑ | CFDrop ↑ | FlipRate ↑ | DeleteValidRate ↑ | StructRed ↓ | AvgDeletedAtomRatio ↓ |
| ------ | -: | ---------------------: | -------: | ---------: | ----------------: | ----------: | --------------------: |

This auxiliary table must not replace the native-action main table. It is intended for deletion-compatible analysis, not for the final cross-baseline ranking.
