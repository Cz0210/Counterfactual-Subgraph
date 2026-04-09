# Counterfactual Subgraph v3 Specification

## 1. Problem Statement

We consider a binary molecular classification setting. Each molecule is represented as a SMILES string and has a label `y ∈ {0,1}`.

The goal is to train a language model generator that, given a parent molecule `x`, outputs one connected fragment `g` in SMILES format such that deleting `g` from `x` is likely to reverse the predicted label.

This is a **counterfactual generation** problem.

Formally, let:

- `x`: parent molecule;
- `y`: original label of `x`;
- `g`: generated fragment;
- `x \ g`: residual molecule after deleting fragment `g` from `x`.

We want `g` to satisfy both structural validity and counterfactual effectiveness.

---

## 2. Core Objective

The v3 objective is:

> Generate a fragment whose deletion most likely changes the label of the original molecule.

This differs fundamentally from prior objectives such as:

- selecting a rationale that preserves the original prediction;
- selecting a highly class-indicative fragment;
- generating a concept subgraph that summarizes the class.

The current project should not drift back to those older objectives.

---

## 3. Desired Fragment Properties

An ideal generated fragment should satisfy the following requirements.

### 3.1 Syntax requirement

The output must be a valid SMILES string or at least a candidate that can be reliably parsed.

### 3.2 Chemical validity

The fragment should be chemically valid after parsing and sanitization whenever possible.

### 3.3 Connectivity requirement

The fragment must correspond to one connected subgraph rather than a disconnected set of atoms.

### 3.4 Parent-substructure requirement

The fragment must be a genuine substructure of the input parent molecule.

### 3.5 Counterfactual requirement

Removing the fragment from the parent should produce a residual structure whose predicted label is different from the original label with high probability.

### 3.6 Conciseness requirement

The fragment should not be unnecessarily long. Overly large fragments make the explanation less precise and may trivially force label changes.

### 3.7 Stability requirement

The generator should avoid collapse modes such as repeated characters, instruction leakage, or nearly constant outputs across diverse inputs.

---

## 4. Three-Stage Training Design

The project adopts a staged training strategy.

## 4.1 Stage A: Structure/format-oriented SFT

Purpose:

- teach the model to obey the output format;
- output only one fragment SMILES;
- reduce extra text contamination;
- build basic fragment generation ability.

Expected benefit:

This stage gives the generator a stable surface form before introducing more difficult structural and semantic constraints.

## 4.2 Stage B: Weakly supervised substructure SFT

Purpose:

- expose the model to connected fragments that are actual parent substructures;
- bias the model toward chemically plausible outputs;
- reduce parse failures and parent mismatch during RL.

Possible sources of weak targets include:

- Murcko scaffold related fragments;
- BRICS decomposition fragments;
- SMARTS-guided local motifs;
- hop-expanded motif candidates;
- other deterministic heuristics that extract connected parent subgraphs.

Expected benefit:

This stage narrows the search space for RL and improves sample efficiency.

## 4.3 Stage C: RL for true counterfactual optimization

Purpose:

- optimize directly for deletion-based label flipping;
- balance semantic effect with validity and compactness;
- avoid degeneration under policy optimization.

Expected benefit:

This stage aligns generation with the actual research objective rather than with heuristic weak labels.

---

## 5. Input and Output Contract

### 5.1 Minimal input sample

```json
{"id": 2, "smiles": "CC1(Cl)C(=O)NC(=O)NC1O", "label": 1}
```

### 5.2 Prompt contract

A typical prompt should clearly ask for exactly one connected fragment and forbid extra text.

Example prompt style:

```text
You are given a molecule SMILES. Output ONE connected substructure SMILES whose deletion is most likely to flip the molecule label.
The output fragment must be a valid connected substructure of the molecule.
Output SMILES only, no extra text.
MOLECULE_SMILES: {smiles}
FRAGMENT_SMILES:
```

### 5.3 Output contract

The model output should contain:

- exactly one fragment candidate;
- no explanation;
- no markdown formatting;
- no prompt echo.

---

## 6. Reward Design Principles

The exact implementation can evolve, but the v3 reward must follow these principles.

### 6.1 Structural terms

The reward should include terms that encourage:

- parseability;
- chemical validity;
- parent-substructure consistency;
- connectivity.

These terms prevent RL from exploiting degenerate strings.

### 6.2 Counterfactual effect term

The reward must include a term that measures how strongly deleting the generated fragment promotes label reversal.

Possible operationalizations include:

- direct label flip indicator from a classifier;
- probability drop of the original class after deletion;
- margin reversal score;
- class-specific shared reward under grouped optimization.

This term is the semantic core of the project.

### 6.3 Compactness term

The reward should discourage overly long fragments, because deleting very large portions of the molecule may artificially force a flip while reducing interpretability.

### 6.4 Anti-collapse term

The reward or auxiliary penalties should discourage outputs that show clear degeneration patterns, such as:

- repeated single characters;
- repeated motifs unrelated to the parent molecule;
- near-constant outputs across a batch.

### 6.5 KL/stability term

The RL objective should include policy regularization or KL control so that optimization does not diverge sharply from the SFT policy.

---

## 7. Known Failure Modes

This section captures the major issues observed in earlier versions.

### 7.1 Mid-training collapse

Training may begin with diverse outputs but later collapse into low-diversity or meaningless strings.

### 7.2 Repeated-token degeneration

A typical failure mode is generation of long repeated sequences such as `NNNNNN...`.

This indicates that the reward and sampling dynamics are no longer properly constraining the policy.

### 7.3 KL explosion or instability

Large-magnitude KL behavior can appear when the policy drifts too rapidly relative to the reference policy.

### 7.4 Reward hacking

The policy may exploit superficial validity or length statistics without actually producing useful counterfactual fragments.

---

## 8. Evaluation Protocol

A trained model should be evaluated from both structural and counterfactual perspectives.

### 8.1 Structural metrics

Recommended metrics include:

- format success rate;
- parseable rate;
- chemistry-valid rate;
- connected-fragment rate;
- parent-substructure rate.

### 8.2 Counterfactual metrics

Recommended metrics include:

- label-flip rate after deletion;
- original-class probability drop;
- margin change;
- reward mean and reward-term breakdown.

### 8.3 Collapse diagnostics

Recommended diagnostics include:

- repeated-token statistics;
- output entropy or diversity proxies;
- rate of duplicate generations;
- average output length;
- extreme collapse bucket counts.

### 8.4 Qualitative analysis

The evaluation should also save representative examples:

- parent SMILES;
- generated fragment;
- whether it is a valid substructure;
- deletion result;
- change in prediction.

---

## 9. Engineering Requirements Derived from the Objective

Because the project is being rebuilt from scratch, the codebase should explicitly separate:

- chemistry utilities;
- reward computation;
- model interfaces;
- train loops;
- evaluation logic;
- data processing;
- CLI wrappers.

The reward module should be independently testable.

The inference pipeline for one sample should be runnable without depending on the full training loop.

The evaluation module should be able to assess a saved checkpoint independently.

---

## 10. Non-Goals

The following are not the primary target of the v3 project:

1. discovering one universal fragment for all molecules regardless of input;
2. generating unrestricted free-form chemical text;
3. optimizing only for reconstruction or likelihood;
4. generating explanations that merely justify the original label;
5. maximizing class correlation of the fragment alone.

---

## 11. Initial Implementation Milestones

A reasonable first build sequence is:

1. define configs and data schema;
2. implement RDKit-based SMILES/fragment utilities;
3. implement reward dataclasses and reward breakdown computation;
4. implement single-sample inference;
5. implement SFT training entry;
6. implement RL training entry with logging;
7. implement evaluation scripts;
8. implement test cases and documentation.

---

## 12. Practical Decision Rule

Whenever there is uncertainty during implementation, use the following rule:

> Prefer the choice that better preserves the deletion-based counterfactual objective, even if it is less convenient from an engineering perspective.

This rule is intended to prevent accidental drift back to non-counterfactual formulations.
