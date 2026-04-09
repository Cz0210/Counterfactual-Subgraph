# Decisions Log

This file records major design decisions for the counterfactual subgraph v3 project.

It should be updated whenever a meaningful implementation, algorithmic, or interface decision is made.

---

## Template

```md
## [YYYY-MM-DD] Decision title

### Background
Why was this decision needed?

### Decision
What was decided?

### Alternatives considered
What other options were considered?

### Consequences
What changes because of this decision?

### Status
Proposed / Accepted / Deprecated / Superseded
```

---

## [2026-04-09] Rebuild repository from scratch with documentation-first workflow

### Background
The previous project evolved through multiple experimental iterations. It likely accumulated script coupling, outdated assumptions, and mixed objectives inherited from earlier versions. To prevent repeated confusion during reconstruction, the repository needs a stable written definition before code implementation begins.

### Decision
Rebuild the repository from an empty root using a documentation-first workflow. The initial authoritative files are:

- `README.md`
- `AGENTS.md`
- `docs/cf_subgraph_v3_spec.md`
- `docs/refactor_plan.md`
- `docs/decisions.md`

These files define the objective, engineering rules, roadmap, and future design log.

### Alternatives considered
1. Start coding immediately and add documentation later.
2. Recover the old repository structure first and refactor afterward.
3. Keep all design notes only in chat history.

### Consequences
- The repository gets a clear source of truth from the start.
- Codex can be guided by repository-local instructions instead of relying on conversational memory.
- Early progress is slower in appearance but more stable in direction.

### Status
Accepted

---

## [2026-04-09] Treat counterfactual fragment generation as the sole primary objective

### Background
Earlier project stages and surrounding discussions may involve concept extraction, class-indicative subgraphs, or rationale-like objectives. Those formulations are related but not identical to the present v3 goal.

### Decision
Define the project strictly as counterfactual fragment generation: the generated fragment should be useful because deleting it is likely to flip the label.

### Alternatives considered
1. Optimize for fragment-only label predictiveness.
2. Optimize for class-shared concept extraction.
3. Optimize for rationale sufficiency rather than deletion-based effect.

### Consequences
- Reward design must include deletion-based semantics.
- Evaluation must include flip-related metrics.
- Old code paths aligned to non-counterfactual targets should be treated as legacy behavior.

### Status
Accepted

---

## [2026-04-09] Use modular repository structure rather than monolithic training script

### Background
The project involves chemistry checks, prompt generation, SFT, RL, evaluation, and logging. A single large script would make the system difficult to debug and easy to break.

### Decision
Adopt a modular structure with separate folders for data, models, rewards, training, evaluation, chemistry utilities, and general utilities.

### Alternatives considered
1. Keep everything in one file for convenience.
2. Split only by training stage.
3. Organize by experiments rather than functionality.

### Consequences
- Initial setup requires more files.
- The system becomes easier to test, replace, and extend.
- Reward and chemistry behavior can be validated independently.

### Status
Accepted

---

## [2026-04-09] Prioritize chemistry and reward layers before large-scale training code

### Background
The most brittle parts of this project are likely to be chemistry correctness and reward semantics. If these layers are unstable, training results will be misleading.

### Decision
Implement chemistry utilities and reward computation before building full SFT/RL training pipelines.

### Alternatives considered
1. Start with full training script and fill utility functions later.
2. Start with model wrapper only.
3. Start with evaluation only.

### Consequences
- Early iteration focuses on correctness rather than speed.
- The train loop can remain thinner and easier to debug.
- Reward failures and chemistry mismatches can be unit-tested.

### Status
Accepted

---

## [2026-04-09] Bootstrap the repository with typed module interfaces before any training implementation

### Background
After the documentation-first reset, the repository still had no source tree, no config folders, and no stable interface boundaries. If training code were introduced at that point, chemistry logic, reward semantics, and CLI behavior would likely become coupled again.

### Decision
Create the repository skeleton first and define minimum typed interfaces for:

- `src/data/`
- `src/chem/`
- `src/rewards/`
- `src/models/`
- `src/train/`
- `src/eval/`
- `src/utils/`

Also add thin placeholder CLI files under `scripts/` and minimal smoke tests for prompt and reward contracts.

The bootstrap intentionally implements only safe low-level helpers such as JSONL IO, prompt construction, reward-term aggregation, and text-level collapse diagnostics. RDKit-backed chemistry logic and all training loops remain deferred.

### Alternatives considered
1. Start with one large training script and refactor later.
2. Implement RDKit parsing and training code immediately without freezing interfaces first.
3. Leave the directory layout undocumented and decide file boundaries ad hoc.

### Consequences
- Future chemistry, reward, inference, and training work now has a stable import surface.
- The repository stays aligned to the deletion-based counterfactual objective instead of drifting toward concept extraction.
- Scripts remain thin by design, because domain logic now has clear module ownership.
- The next implementation phase can focus on chemistry correctness rather than file organization.

### Status
Accepted
