# Scripts

This directory is reserved for thin command-line entrypoints only.

Some scripts are still bootstrap placeholders, but the repository now also
contains concrete local/HPC entrypoints for:

- SFT training
- SFT v3 HIV dataset rebuild and audit
- AIDS oracle training
- PPO RL training
- PPO candidate-pool audit
- PPO reward/teacher audit

Recommended HPC run layout for the rebuilt HIV-derived SFT v3 workflow:

```text
outputs/hpc/sft_v3_hiv_runs/<RUN_NAME>/
  dataset/
    sft_v3_hiv_train.jsonl
    sft_v3_hiv_val.jsonl
    *.summary.json
    *.report.txt
    *.dropped_summary.json
  audit/
    train/
      sft_v3_hiv_train.summary.json
      sft_v3_hiv_train.details.csv
    val/
      sft_v3_hiv_val.summary.json
      sft_v3_hiv_val.details.csv
  train/
    checkpoint-*
    trainer_state.json
  eval/
    summary.json
    details.jsonl
  logs/
    *.warn.log
```

The paired Slurm wrappers now default to this layout so a single `RUN_NAME`
can be reused across dataset build, audit, training, and checkpoint evaluation.

One-command HPC submission is now available through:

- `scripts/slurm/submit_sft_v3_hiv_pipeline.sh`

It submits the dependency-aware Slurm chain:

```text
build
  ├─ audit
  └─ train
       └─ eval
```

Example:

```bash
RUN_NAME=sft_v3_hiv_20260508_full \
ORACLE_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/oracle/aids_rf_model.pkl \
USE_ORACLE_RANKING=true \
MAX_STEPS=500 \
REPORT_TO=none \
bash scripts/slurm/submit_sft_v3_hiv_pipeline.sh
```

Planned entrypoints:

- `scripts/prepare_data.py`
- `scripts/prepare_sft_data.py`
- `scripts/run_prepare_sft_v3_core.sh`
- `scripts/infer_single.py`
- `scripts/train_aids_oracle.py`
- `scripts/train_sft.py`
- `scripts/run_sft_v3_core.sh`
- `scripts/train_ppo.py`
- `scripts/train_rl.py`
- `scripts/run_ppo_v3_core_smoke.sh`
- `scripts/run_ppo_v3_core_diagnose.sh`
- `scripts/audit_candidate_pool.py`
- `scripts/audit_reward_teacher.py`
- `scripts/eval_model.py`
- `scripts/run_eval_sft_v3_core.sh`
