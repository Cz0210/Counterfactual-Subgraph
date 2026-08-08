# BBBP Future Runbook

Status: `FRAMEWORK_ONLY_NOT_RUN`

## Preflight inputs

Before any GPU submission, provide and fingerprint:

1. `data/raw/BBBP/bbbp.csv` and explicit raw column names when aliases are
   ambiguous;
2. ChemLLM stable SFT/PPO checkpoints and decoding configuration;
3. the BBBP GlobalGCE graph/label conversion manifest;
4. the BBBP GCFExplainer atom-channel and NeuroSED compatibility manifest;
5. BBBP COMRECGC GNN/native adapter inputs;
6. RF and MolCLR paths, hashes, and a calibration-only threshold root.

Run CPU-only validation first:

```bash
python scripts/plan_bbbp_experiments.py --plan all --validate-only
python scripts/plan_bbbp_experiments.py --plan all --emit-shell /tmp/bbbp_plan.sh
```

The planner cannot submit and does not write the experiment registry. The
generated shell begins with an authorization guard and uses
`scripts/exp_sbatch.sh`; it is a future plan, not an executable approval.

## Future execution order

1. BBBP prepare and split-leakage gate.
2. Train/select the BBBP RF teacher using train/validation only.
3. Ours candidate generation, persistence, audit, calibration selector, eval.
4. GlobalGCE, GCFExplainer, and COMRECGC native DAGs.
5. Common four-method paper artifact audit.
6. Held-out molecule protocol.
7. Cross-scaffold protocol.
8. PPO/SFT/random candidate-source ablation.
9. Selector-component ablation.
10. Nested budget scaling.
11. Seeds 0/1/2 aggregation.
12. Parent-level bootstrap and curve confidence bands.

Each future stage must be registered through `scripts/exp_sbatch.sh`, use a
fresh output root, and pass its dependencies with `afterok`. Test loading is
permitted only in final evaluation stages. A failed split, lineage, threshold,
or candidate-source audit stops the chain.

## Wrapper validation

Every BBBP wrapper supports `VALIDATE_ONLY=1` or `DRY_RUN=1`. Native baseline
stages whose dataset-specific inputs are not yet frozen return `INPUT_REQUIRED`
instead of inventing a command or falling back to AIDS/BACE settings. Slurm
resources and runtime metrics remain `null`/`NOT_RUN` until a future authorized
preflight resolves them from proven jobs.

Do not interpret this runbook or its generated commands as an experiment
authorization.
