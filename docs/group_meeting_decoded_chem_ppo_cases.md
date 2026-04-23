# Decoded Chemistry PPO PPT Cases

本文件只整理 PPT 可直接展示的案例，不重复审计完整 PPO 流程。  
案例来源分为两类：

- `hpc_smoke_test_log`: 来自 `logs/1395635.err` / `logs/1395635.out`
- `synthetic_unit_test_case`: 来自 unit test 设定与当前代码语义，明确不是 HPC 训练结果

## Case 1: Dummy atom normalization + logged teacher semantic reward

| Field | Value |
|---|---|
| case_id | `hpc_1395635_step1_case11` |
| source | `hpc_smoke_test_log` |
| parent_smiles | `O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1` |
| label | `1` |
| prompt | `You are given a molecule SMILES. Output ONE connected substructure SMILES whose deletion is most likely to flip the molecule label.<br>The output fragment must be a valid connected substructure of the molecule.<br>Output SMILES only, no extra text.<br>ORIGINAL_LABEL: 1<br>MOLECULE_SMILES: O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1<br>FRAGMENT_SMILES:` |
| generated_response | ` *CC(=O)O` |
| raw_fragment | `*CC(=O)O` |
| core_fragment | `CC(=O)O` |
| dummy_count | `1` |
| raw_parse_ok | `true` |
| core_parse_ok | `true` |
| valid | `1.0` |
| substructure | `1.0` |
| length_reward | `0.25` |
| teacher_input_smiles | `CC(=O)O` |
| teacher_prob | `0.0033333334140479565` |
| teacher_sem | `-0.9933333331719041` |
| fragment_teacher_sem | `null` |
| counterfactual_sem | `-5.0` |
| total_reward | `-3.493333333171904` |
| PPO update | `step=1; reward_mean=-3.4933; policy_loss=-0.0325; value_loss=8.0650; total_loss=4.0000; approx_kl=-0.5078` |

**What this case shows:** 真实 smoke test 已经展示了 `response -> raw_fragment -> core_fragment` 的 dummy atom 归一化链路，并且 teacher 已经在 `core_fragment=CC(=O)O` 上被实际调用；但该日志里的 `counterfactual_sem` 仍是 `-5.0`，所以这页更适合证明 reward pipeline 打通，而不是证明 counterfactual 行为已经成熟。

## Case 2: Second real smoke-test fragment + PPO step 2

| Field | Value |
|---|---|
| case_id | `hpc_1395635_step2_case16` |
| source | `hpc_smoke_test_log` |
| parent_smiles | `NNP(=S)(NN)c1ccccc1` |
| label | `1` |
| prompt | `You are given a molecule SMILES. Output ONE connected substructure SMILES whose deletion is most likely to flip the molecule label.<br>The output fragment must be a valid connected substructure of the molecule.<br>Output SMILES only, no extra text.<br>ORIGINAL_LABEL: 1<br>MOLECULE_SMILES: NNP(=S)(NN)c1ccccc1<br>FRAGMENT_SMILES:` |
| generated_response | ` *P(=S)(NN)c1ccccc1` |
| raw_fragment | `*P(=S)(NN)c1ccccc1` |
| core_fragment | `NN[PH](=S)c1ccccc1` |
| dummy_count | `1` |
| raw_parse_ok | `true` |
| core_parse_ok | `true` |
| valid | `1.0` |
| substructure | `1.0` |
| length_reward | `0.25` |
| teacher_input_smiles | `NN[PH](=S)c1ccccc1` |
| teacher_prob | `0.009999999776482582` |
| teacher_sem | `-0.9800000004470348` |
| fragment_teacher_sem | `null` |
| counterfactual_sem | `-5.0` |
| total_reward | `-3.480000000447035` |
| PPO update | `step=2; reward_mean=-3.4800; policy_loss=-0.0487; value_loss=3.3114; total_loss=1.6070; approx_kl=-0.2328` |

**What this case shows:** 第二个真实 case 说明 smoke test 不是只打通了一个特殊分子；另一条带 dummy atom 的含磷片段也被成功归一化到 core fragment，并且在 step 2 之后确实发生了 PPO update。

## Case 3: Synthetic unit test case where fragment-level teacher is positive but counterfactual signal is missing

| Field | Value |
|---|---|
| case_id | `synthetic_teacher_sem_core_fragment_positive` |
| source | `synthetic_unit_test_case` |
| parent_smiles | `O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1` |
| label | `0` |
| prompt | `null` |
| generated_response | `null` |
| raw_fragment | `*CC(=O)O` |
| core_fragment | `CC(=O)O` |
| dummy_count | `1` |
| raw_parse_ok | `true` |
| core_parse_ok | `true` |
| valid | `1.0` |
| substructure | `1.0` |
| length_reward | `0.25` |
| teacher_input_smiles | `CC(=O)O` |
| teacher_prob | `0.8` |
| teacher_sem | `-5.0` |
| fragment_teacher_sem | `0.6000000000000001` |
| counterfactual_sem | `-5.0` |
| total_reward | `-2.5` |
| PPO update | `null` |

**What this case shows:** 这个 synthetic case 来自 `tests/test_teacher_sem_reward.py` 的设置，它说明 fragment-level teacher 在 core fragment 上可以给出正分 `0.6`，但如果没有 counterfactual teacher，当前进入 total 的语义项仍会落到 missing penalty，因此不能把“teacher 命中”误解为“counterfactual reward 已经成功”。

## Case 4: Synthetic unit test case where counterfactual semantic reward enters total

| Field | Value |
|---|---|
| case_id | `synthetic_counterfactual_sem_enters_total` |
| source | `synthetic_unit_test_case` |
| parent_smiles | `O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1` |
| label | `1` |
| prompt | `null` |
| generated_response | `null` |
| raw_fragment | `*CC(=O)O` |
| core_fragment | `CC(=O)O` |
| dummy_count | `1` |
| raw_parse_ok | `true` |
| core_parse_ok | `true` |
| valid | `1.0` |
| substructure | `1.0` |
| length_reward | `0.25` |
| teacher_input_smiles | `CC(=O)O` |
| teacher_prob | `0.8` |
| teacher_sem | `0.5` |
| fragment_teacher_sem | `0.6000000000000001` |
| counterfactual_sem | `0.5` |
| total_reward | `3.0` |
| PPO update | `null` |

**What this case shows:** 这个 synthetic case 来自 `tests/test_counterfactual_teacher.py` 的假 scorer 设置，它专门用来说明“当 counterfactual scorer 返回正分时，进入 total 的是 `counterfactual_sem`，不是 fragment-level `fragment_teacher_sem`”。

## How to use these cases in PPT

1. 先用 Case 1 做一页 “raw fragment -> core fragment”：
   `*CC(=O)O -> CC(=O)O`，同时标出 `dummy_count=1 / raw_parse_ok=True / core_parse_ok=True`。
2. 再用 Case 1 和 Case 2 做一页 “真实 smoke test 已经发生 PPO update”：
   直接对比 `step=1` 与 `step=2` 的 `reward_mean / policy_loss / value_loss / approx_kl`。
3. 用 Case 3 做一页 “fragment-level teacher semantic 不是 total reward”：
   强调 `fragment_teacher_sem=0.6`，但 `teacher_sem=-5.0`、`total_reward=-2.5`。
4. 用 Case 4 做一页 “counterfactual semantic enters total”：
   画成 `format + valid + sub + len + counterfactual_sem = total_reward`。
5. 最后一页用 Case 1/2 对照 Case 4：
   左边是真实 smoke test 的负例日志，右边是 synthetic 正向逻辑例子；结论写成“pipeline 已打通，但还不能把 smoke test 夸大成成熟反事实结果”。

## Caveats

1. 来自 `logs/1395635` 的案例是 smoke test，不是完整训练结果。
2. smoke test 的 `max_steps` 很小，只跑了 2 步，不能说明模型已经收敛。
3. 如果 teacher 输入是 `core_fragment`，那它表达的是 fragment-level semantic score，而不是 deletion-based residual score。
4. 在 `logs/1395635` 的真实案例里，`counterfactual_sem` 仍然是 `-5.0`；就这次 smoke test 展示出来的 run 而言，可以把它理解成 deletion-based counterfactual oracle 还没有在日志结果里完整体现出来，因此这些案例更适合展示 reward pipeline 已打通，而不适合声称 deletion-based counterfactual reward 已在该次 smoke test 中稳定工作。
5. 当前这些案例用于展示 decode、dummy atom normalization、teacher 调用和 PPO update 的链路，不代表最终的类别级反事实子图集合结果。
6. 不能声称已经完成 class-level low-redundancy selector；当前材料还只是 single-example / single-fragment 级别。
7. `logs/1395635` 的日志字段命名和当前 audit 文档里的“语义分量命名澄清”并不完全一致，因此组会上最好显式区分“日志原字段”和“当前代码语义解释”。
8. synthetic case 不是 HPC 实验结果；它们只用于补足展示逻辑，不能和真实 smoke test 数字混为一谈。
9. synthetic case 中的 `prompt`、`generated_response`、`ppo_step` 和 PPO loss 字段为 `null`，因为 unit test 直接注入 fragment 或 fake scorer，并没有真实 rollout/update。
