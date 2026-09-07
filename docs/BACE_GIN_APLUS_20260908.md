# BACE GIN A+ execution

The 2026-09-08 user contract permits adoption of the already sealed 2607
train-only Reach rules (66 original and 2541 additional), not additional PPO.
The original GIN fixed-pool experiment and source results remain unchanged.

## Selector audit

The actual original B12 main driver runs the preregistered A1-A4 variants and
chooses by calibration prefix-weighted multi-threshold utility, then K10
coverage, capped cost and the remaining recorded tie-breaks. V1 does the same
in `src/experiments/bace_gin_fixed_pool.py::freeze`; its actual calibration
decision selected A1_SingleTheta. The older GNN sensitivity's fixed A4 is a
different experiment. An A1 name alone is not an implementation defect.

A+ is a new declared selector, not a rename of V1: 0.25 finite strict reach plus
0.75 original normalized threshold-grid coverage. Deterministic greedy and
two 1-swap passes preserve the legal V1 S10 calibration theta floor; S20
contains the newly ordered S10. No global optimality is claimed. Fixed141,
theta, cap, native operation, model and fitted temperature remain unchanged.
Previous test exposure and post-hoc development must be disclosed.

## Execution

`scripts/experiments/run_bace_gin_reach_v2.py` consumes an absolute sealed spec,
with plan/evaluate/ceiling/freeze/aggregate/status stages. Its paired CPU Slurm
script uses the user's task-specific intel/8CPU/32GiB setting rather than the
repository's generic GPU-training default. Set the immutable worktree with
`sbatch --chdir` and use an absolute spec. Do not run science on a login node.

Old66 train is evaluated completely; old66 calibration adopts the completed
V1 parent units under the identical GIN. Expanded train/calibration retain
those rows and compute only the new candidates. A new global calibration
freeze precedes test. Raw graph costs are separately adopted with graph,
schema, MolCLR, kernel and action provenance; old GINE flip minima are never
adopted. Per-parent completion is the resume boundary. This driver does not
write the main authority and cannot train any model.

Bounded new search remains contingent on the full train ceiling; it is not
dispatched by the fixed saved-pool evaluation. A 70% research target is not a
PASS requirement and cannot justify altered denominators or test-led search.
# Train-only adoption decision and saved-record audit

The complete 2607-pool train evaluation precedes the train adoption decision.
Any same-GIN source training parent without a strict-flip witness triggers the
bounded supplemental-search route. Calibration outcomes and all test records
are excluded from this decision. A new global selector cannot silently bypass
an unclosed supplemental-search decision. The observed 386-parent old-pool
result is R158/H141/L87; full calibration old66 is R19/H18/L7. These are measurements,
not target gates. A separate saved-application consistency audit precedes test.
