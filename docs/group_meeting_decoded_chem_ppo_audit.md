# Decoded Chemistry PPO 组会审计

日期：2026-04-20  
审计范围：`scripts/train_ppo.py`、`src/rewards/reward_wrapper.py`、`src/rewards/teacher_semantic.py`、`src/rewards/reward_calculator.py`、`scripts/slurm/debug_decoded_chem_ppo_smoketest.sh`、`scripts/slurm/train_ppo.sh`、`tests/test_teacher_sem_reward.py`、`tests/test_reward_wrapper_dummy_atoms.py`、`tests/test_train_ppo_helpers.py`、`docs/decisions.md`

## 0. 先说结论

### 当前架构概览

- 当前 `decoded_chem` 路径确实是“明文 prompt -> ChemLLM generate -> decode text -> 抽 fragment -> chemistry reward -> local PPO update”，不是把 reward 黑盒塞给 TRL 处理；这和 `docs/decisions.md` 2026-04-18 / 2026-04-19 的记录一致。代码主入口在 `scripts/train_ppo.py:2157-2607`，文档决策见 `docs/decisions.md:32-191`。
- 当前真正进入 `total reward` 的语义项是 deletion-based 的 `counterfactual_sem`，不是 fragment-level 的 `teacher_sem`。`reward_wrapper` 文件头也明确写了“fragment-level teacher scores are retained only as diagnostic fields” (`src/rewards/reward_wrapper.py:10-16`)。
- 但日志字段命名有一个必须在组会上讲清楚的点：`reward_logs["teacher_sem"]` 目前不是 fragment-level teacher score，而是 `sem_r` 的别名；而成功路径里 `sem_r == cf_r == counterfactual_reward`。真正的 fragment-level teacher 分量在 `reward_logs["fragment_teacher_sem"]`。关键代码在 `src/rewards/reward_wrapper.py:772-902`、`scripts/train_ppo.py:2442-2464`。

### 主要发现

1. 当前 decoded PPO 的奖励主项已经是 deletion-based counterfactual reward，不再是“fragment alone predicts original label”的旧目标。
2. `teacher_sem` 仍然实现为 fragment-level `2 * p_y - 1`，其中 `p_y = p(original_label | core_fragment)`；但它当前只做诊断，不进总奖励。代码在 `src/rewards/teacher_semantic.py:236-296`、`src/rewards/reward_wrapper.py:748-779`。
3. local decoded PPO loop 没有调用 TRL 原生 `step()`；日志里明确写着 “Using local PPO loss” (`scripts/train_ppo.py:2181-2184`)。
4. `mini_batch_size` 在 decoded local PPO 里没有真正被消费；`run_decoded_chem_ppo_loop(...)` 根本不接这个参数，只有 TRL experimental 配置路径会用到。见 `scripts/train_ppo.py:2157-2171`、`scripts/train_ppo.py:2705-2708`。
5. `save_steps` 在 decoded local PPO 里也没有中途落 checkpoint；当前只在训练结束时做一次最终保存，见 `scripts/train_ppo.py:1802-1819`、`scripts/train_ppo.py:2601-2607`。

### 风险与歧义

- `ChemRLRewarder` 在初始化时会加载 `oracle_path` bundle（`src/rewards/reward_wrapper.py:328-334`），但当前 decoded reward 主路径实际依赖的是 `teacher_path -> TeacherSemanticScorer / CounterfactualTeacherScorer`。`self._oracle_model` 在 `reward_wrapper.py` 中没有后续读取。
- `chem_reward_model = build_reward_model_wrapper(...)` 会被创建并传入 decoded loop，但 decoded loop 里真正被调用的是 `chem_rewarder.compute_rewards_from_decoded(...)`，`chem_reward_model` 只出现在日志与函数签名中。见 `scripts/train_ppo.py:2177-2179`、`scripts/train_ppo.py:2321-2334`、`scripts/train_ppo.py:2795-2833`。
- `src/rewards/reward_wrapper.py` 里有一套 `_delete_to_residual_smiles()` / `_build_deletion_query()` / `_sanitize_residual_molecule()` helper（`src/rewards/reward_wrapper.py:1127-1291`），但当前有效的 counterfactual 路径实际走的是 `src/rewards/counterfactual_oracle.py:75-167, 194-363` 的 `delete_one_substructure()`。

### 建议的增量计划

1. 先把日志字段命名澄清：`teacher_sem`、`fragment_teacher_sem`、`counterfactual_sem` 三者不要再混名。
2. 明确保留一套 deletion oracle 实现，避免 `reward_wrapper.py` 和 `counterfactual_oracle.py` 双轨漂移。
3. 如果下一阶段要做“类别级低冗余 selector”，需要新增 group-level / class-level redundancy 目标，而不是继续在单样本 fragment reward 上堆权重。

## 1. decoded chemistry PPO 的总体流程

当前真实流程如下。

1. `parent_smiles + label -> prompt construction`
   代码：`src/data/prompts.py:8-33`，`scripts/train_ppo.py:497-549`，`scripts/train_ppo.py:415-473`
2. `prompt -> tokenization`
   代码：`scripts/train_ppo.py:1302-1325`
3. `tokenized query -> ChemLLM generate`
   代码：`scripts/train_ppo.py:2260-2276`
4. `generated ids -> decode response/full text`
   代码：`scripts/train_ppo.py:2278-2293`
5. `decode response -> extract fragment`
   代码：`scripts/train_ppo.py:1186-1205`、`scripts/train_ppo.py:1583-1594`、主调用在 `scripts/train_ppo.py:2295-2305`
6. `fragment -> dummy atom normalization`
   代码：`src/rewards/reward_wrapper.py:189-245`
7. `raw/core fragment -> RDKit parse / connected / substructure`
   代码：`src/rewards/reward_wrapper.py:564-747`
8. `core fragment -> teacher_sem`
   代码：`src/rewards/reward_wrapper.py:748-753, 958-1002`；teacher 本体在 `src/rewards/teacher_semantic.py:187-296`
9. `parent - fragment -> counterfactual_sem`
   代码：`src/rewards/reward_wrapper.py:754-763, 1004-1125`；deletion oracle 在 `src/rewards/counterfactual_oracle.py:194-363`
10. `reward breakdown -> total reward tensor`
    代码：`src/rewards/reward_wrapper.py:772-902`、`src/rewards/reward_wrapper.py:401-486`
11. `reward tensor -> local PPO update`
    代码：`scripts/train_ppo.py:2467-2594`

一句话总结：当前 decoded path 已经是“显式 decoded chemistry PPO”，不是“隐藏状态 reward adapter 兼容性测试”。这一点也被 `docs/decisions.md:135-191` 明确记录。

## 2. PPO prompt

### 2.1 prompt 模板原文

canonical prompt 来自 `src/data/prompts.py:8-33`：

```text
You are given a molecule SMILES. Output ONE connected substructure SMILES whose deletion is most likely to flip the molecule label.
The output fragment must be a valid connected substructure of the molecule.
Output SMILES only, no extra text.
ORIGINAL_LABEL: {record.label}   # 仅 include_label=True 时出现
MOLECULE_SMILES: {record.smiles}
FRAGMENT_SMILES:
```

### 2.2 审计结论

- 是否包含 `parent_smiles`：是，写在 `MOLECULE_SMILES: ...`，见 `src/data/prompts.py:27-31`。
- 是否包含 `label`：canonical prompt 支持，但只有 `include_label=True` 才写入 `ORIGINAL_LABEL: ...`，见 `src/data/prompts.py:24-27`。
- 是否使用 chat template：没有。我没有在 `scripts/train_ppo.py` 里找到 `apply_chat_template` 或 `tokenizer.chat_template`；tokenizer 直接吃 plain text prompt，见 `scripts/train_ppo.py:647-667`、`scripts/train_ppo.py:1307-1314`。
- 是否要求 output only SMILES：是，第三行明确写了 `Output SMILES only, no extra text.`，见 `src/data/prompts.py:9-14`。
- 是否要求 one connected substructure：是，第一、二行都写了 `ONE connected substructure` / `valid connected substructure`，见 `src/data/prompts.py:8-14`。
- 是否说明 dummy atom / attachment point：没有。prompt 没提 `*`、dummy atom、attachment point。

### 2.3 当前训练实际会不会把 label 放进 prompt

- 从原始 HIV CSV 构建 prompt 时，`--include-label-in-prompt` 默认是 `True`，见 `scripts/train_ppo.py:283-286`，实际加载在 `scripts/train_ppo.py:497-549`。
- 但如果输入是现成 JSONL prompt 文件，loader 会优先保留文件中已有的 `prompt`/`instruction`/`query`/`text`，不会强制重渲染 prompt；只有 prompt 缺失时才 fallback 到 `build_counterfactual_prompt(..., include_label=True)`，见 `scripts/train_ppo.py:424-463`。
- 当前 HPC 默认数据源是原始 HIV CSV，因此默认训练运行里 prompt 会包含 `ORIGINAL_LABEL`。配置见 `configs/hpc.yaml:18-23`、`configs/base.yaml:28-43`。

## 3. PPO 数据流

### 3.1 数据源

- 支持两类数据源：
  - 原始 HIV CSV：`scripts/train_ppo.py:497-549`
  - JSONL prompt 文件：`scripts/train_ppo.py:476-494`
- 当前 HPC 默认数据源：`/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv`，见 `configs/hpc.yaml:18-23`。
- 默认会过滤成 `only_positive=True`，即只保留原始 label 为 1 的分子做 PPO，见 `scripts/train_ppo.py:277-280`、`scripts/train_ppo.py:458-459`、`scripts/train_ppo.py:529-530`。

### 3.2 batch 字段

`PromptExample.to_dataset_row()` 生成的数据列是：

| 字段 | 来自哪里 | 代码位置 | 用途 |
|---|---|---|---|
| `index` | 数据行号 | `scripts/train_ppo.py:68-83` | 作为 reward log 的 `id/index` |
| `query` | prompt 文本 | `scripts/train_ppo.py:68-83` | tokenization、generation 输入、fragment 抽取时用于剥离 prompt 前缀 |
| `parent_smiles` | 父分子 | `scripts/train_ppo.py:68-83` | reward 计算、日志 |
| `original_label` | 原始标签 | `scripts/train_ppo.py:68-83` | reward 计算中的 label |
| `input_ids` | tokenizer(query) | `scripts/train_ppo.py:1307-1314` | policy generate、policy/reference/value forward |
| `attention_mask` | data collator padding 生成 | `scripts/train_ppo.py:1328-1376` | generate、logprob/value 计算 |

### 3.3 各字段如何使用

- `parent_smiles`
  - 从 batch 中读出：`scripts/train_ppo.py:2249-2253`
  - 送入 `chem_rewarder.compute_rewards_from_decoded(...)`：`scripts/train_ppo.py:2321-2334`
  - 记录到 reward logs：`src/rewards/reward_wrapper.py:428-471`
- `original_label`
  - 从 batch 中读出：`scripts/train_ppo.py:2254`
  - 作为 teacher / counterfactual teacher 的 label：`src/rewards/reward_wrapper.py:748-763`
- `query`
  - tokenizer 输入：`scripts/train_ppo.py:1307-1314`
  - generation 的 prompt 文本：`scripts/train_ppo.py:2249`
  - 抽 fragment 时用来从 `full_text` 里剥掉 prompt：`scripts/train_ppo.py:2296-2305`
- `input_ids`
  - `policy_model.generate(...)` 的输入：`scripts/train_ppo.py:2260-2276`
  - 计算 old/current/ref logprobs 与 old/current values：`scripts/train_ppo.py:2467-2491`、`scripts/train_ppo.py:2529-2544`
- `attention_mask`
  - generation 输入：`scripts/train_ppo.py:2260-2276`
  - 组装 combined prompt+response 序列时参与 logprob/value forward：`scripts/train_ppo.py:1706-1723`、`scripts/train_ppo.py:1737-1760`
- `index`
  - batch 中解析：`scripts/train_ppo.py:2255-2258`
  - 填到 `metas` 里：`scripts/train_ppo.py:2325-2331`
  - 最终进入 reward 日志 `id`：`src/rewards/reward_wrapper.py:423-471`

## 4. response decode 和 fragment extraction

### 4.1 generation 参数

decoded PPO 生成参数在 `scripts/train_ppo.py:2265-2276`：

- `max_new_tokens = args.max_new_tokens`，默认 64
- `do_sample = True`
- `top_p = args.top_p`，默认 0.95
- `temperature = args.temperature`，默认 0.8
- `pad_token_id = tokenizer.pad_token_id`
- `eos_token_id = tokenizer.eos_token_id`
- `use_cache = False`

### 4.2 response 如何 decode

- response token ids：`response_ids = generated_ids[:, input_ids.shape[1]:]`，见 `scripts/train_ppo.py:2278`
- response text：`tokenizer.batch_decode(response_ids, skip_special_tokens=True)`，见 `scripts/train_ppo.py:2286-2289`
- full text：`tokenizer.batch_decode(generated_ids, skip_special_tokens=True)`，见 `scripts/train_ppo.py:2290-2293`

### 4.3 fragment 如何抽取

主逻辑：

1. 先走 `_extract_fragment_from_text(full_text, prompt_text)`，见 `scripts/train_ppo.py:2296-2305`
2. 若 `full_text` 以前缀 prompt 开头，则直接取 suffix，再过 `clean_generated_smiles(...)`，见 `scripts/train_ppo.py:1189-1197`
3. 否则若存在 `FRAGMENT_SMILES:`，取其后的 suffix，再清洗，见 `scripts/train_ppo.py:1199-1203`
4. 还不行就直接清洗 `normalized_combined`，见 `scripts/train_ppo.py:1205`
5. 如果这一步仍为空，则 fallback 到 `extract_fragment_smiles(response_text)`，它会取第一条非空行并做 `clean_generated_smiles(...)`，见 `scripts/train_ppo.py:1583-1594`

### 4.4 示例

示例 `response=" *CC(=O)O"` 时：

- `extract_fragment_smiles(" *CC(=O)O") -> "*CC(=O)O"`，逻辑见 `scripts/train_ppo.py:1583-1594`
- 若 `full_text = prompt + " *CC(=O)O"`，则 `_extract_fragment_from_text(full_text, prompt) -> "*CC(=O)O"`，逻辑见 `scripts/train_ppo.py:1186-1205`

也就是说，当前抽取逻辑会保留前导 dummy atom `*`，不会在 decode 阶段就把它删掉。

## 5. dummy atom 处理

dummy atom 规范化完全在 reward path 里做，核心函数是 `normalize_fragment_with_dummy_atoms(...)`，见 `src/rewards/reward_wrapper.py:189-245`。

### 5.1 字段含义

- `raw_fragment_smiles`
  - 含义：模型解码后抽出来的原始 fragment，可能带 `*`
  - 代码：`RewardTrace` 字段定义 `src/rewards/reward_wrapper.py:78-80`，填充在 `src/rewards/reward_wrapper.py:782-788`
- `core_fragment_smiles`
  - 含义：把 dummy atom 图级删除并重新 sanitize 后的 dummy-free core
  - 代码：`src/rewards/reward_wrapper.py:225-245`
- `dummy_count`
  - 含义：fragment 中 atomic number = 0 的 atom 数
  - 代码：`src/rewards/reward_wrapper.py:135-141`、`223`
- `raw_parse_ok`
  - 含义：原始带 dummy 的 fragment 能否用 `allow_capped_fragments=True` 成功 parse/sanitize
  - 代码：`src/rewards/reward_wrapper.py:209-219`
- `core_parse_ok`
  - 含义：删掉 dummy 之后的 core 是否仍可 sanitize 成有效分子
  - 代码：`src/rewards/reward_wrapper.py:225-245`

### 5.2 为什么 teacher 输入使用 core fragment

- `TeacherSemanticScorer` 要做 Morgan fingerprint；它吃的是一个可用于 fingerprint 的普通分子 SMILES，见 `src/rewards/teacher_semantic.py:236-241`
- reward wrapper 显式把 teacher 输入设成 `core_smiles`，见 `src/rewards/reward_wrapper.py:599-603`、`748-753`、`809-815`
- tests 也明确检查了这一点：
  - `*CC(=O)O -> core = CC(=O)O`，见 `tests/test_teacher_sem_reward.py:80-102`
  - `trace.teacher_input_smiles == "CC(=O)O"`，见 `tests/test_reward_wrapper_dummy_atoms.py:53-74`

结论：teacher 输入使用 core fragment，是因为当前 teacher 是 fingerprint classifier，不是 attachment-point-aware classifier；dummy atom 是 rewarder 认可的 decoded syntax，但不是 teacher 模型真正要看的语义输入。

## 6. reward 函数总表

当前 reward 分量可以整理成下表。

| Reward component | Code variable | Meaning | Condition | Weight / value | Enters total? | Code location |
|---|---|---|---|---|---|---|
| format reward | `format_r` | 文本格式是否像一个单行 fragment SMILES | 非空、无换行/制表/分号，且 `raw_parse_ok=True` 时通过 | `+0.25`；失败 `-0.25` | Yes | `src/rewards/reward_wrapper.py:265-266, 839-851, 884-893, 895-902` |
| valid reward | `valid_r` | raw/core 解析有效性 | `raw_parse_ok && core_parse_ok`；或仅 raw parse 成功 | `+1.0`；仅 raw parse 成功时 `+0.3`；失败 `-2.0` | Yes | `src/rewards/reward_wrapper.py:267-269, 853-858, 884-902` |
| substructure reward | `subgraph_r` | core fragment 是否为 parent 的精确子结构 | 成功路径 `HasSubstructMatch + is_parent_substructure` 都真 | 成功 `+1.0`；subgraph 分支失败 `-2.0`；更早失败时常为 `0.0` | Yes | `src/rewards/reward_wrapper.py:270-271, 676-747, 884-902` |
| length reward | `length_r` | core fragment 长度约束 | `core_atom_count > 0` | `<=12` 个原子时 `+0.25`；超过后每多 1 个原子减 `0.05`，下界 `-0.25` | Yes | `src/rewards/reward_wrapper.py:272-274, 860-871, 884-902` |
| semantic reward | `sem_r` | 当前“真正加进 total 的语义项” | 成功路径设为 `counterfactual_reward` | `teacher_sem_scale * counterfactual_sem` 或缺失惩罚 | Yes | `src/rewards/reward_wrapper.py:772-780, 884-902` |
| teacher_sem | `teacher_sem_r` | 目前只是 `sem_r` 的别名，不是 fragment teacher 分量 | 与 `sem_r` 同步 | 与 `sem_r` 完全相同 | Yes，但本质上等价于 `sem_r` | `src/rewards/reward_wrapper.py:884-892` |
| fragment teacher_sem | `fragment_teacher_sem_r` | fragment-level teacher 诊断分量 | `core_parse_ok` 且 `teacher_scorer` 可用且 `teacher_result_ok=True` | `teacher_sem_scale * (2 * p_y - 1)`；否则 `teacher_sem_missing_penalty` | No | `src/rewards/reward_wrapper.py:748-753, 958-1002, 884-892` |
| counterfactual_sem | `cf_r` | deletion-based counterfactual reward | valid + connected + substructure + counterfactual teacher 可用 | `teacher_sem_scale * (cf_drop + flip_bonus * int(cf_flip))`；否则 `teacher_sem_missing_penalty` | Indirectly yes, because `sem_r` is set to this same value | `src/rewards/reward_wrapper.py:754-763, 1004-1125, 772-780` |
| missing penalty | `teacher_sem_missing_penalty` | teacher / counterfactual teacher 不可用或被跳过时的回退惩罚 | teacher 未创建、teacher fail、counterfactual teacher fail、invalid_or_not_substructure 等 | 默认 `-5.0` | Not standalone; injected into `sem_r` / `cf_r` / `fragment_teacher_sem_r` | `scripts/train_ppo.py:247-250, 2789-2793`; `src/rewards/reward_wrapper.py:278-279, 543-779, 958-1125` |
| invalid format penalty | `format_penalty` | 空输出/多行/分号/解析失败时的格式惩罚 | `_compute_format_reward` fail | 默认 `-0.25` | Yes | `src/rewards/reward_wrapper.py:266, 844-851` |
| invalid smiles penalty | `invalid_smiles_penalty` | raw fragment 无法视为合法候选时的惩罚 | `_compute_valid_reward` fail | 默认 `-2.0` | Yes | `src/rewards/reward_wrapper.py:269, 853-858` |
| invalid subgraph penalty | `invalid_subgraph_penalty` | core 不是 parent 子结构时的惩罚 | subgraph 分支 fail | 默认 `-2.0` | Yes | `src/rewards/reward_wrapper.py:271, 676-747` |
| minimum reward | `minimum_reward` | 非正常入口失败时的保底值，如 label / parent 本身无效 | invalid label 或 parent parse fail | 默认 `-5.0` | Indirectly via `cf_r` in those branches | `src/rewards/reward_wrapper.py:264, 497-541` |
| KL reward term | `-kl_coef * (policy_logprobs - ref_logprobs)` | PPO 中每个 response token 的 reference-policy penalty | 所有有效 response token | `kl_coef = args.init_kl_coef`，默认 `0.1` | Not in chemistry total; yes in PPO token rewards | `scripts/train_ppo.py:1763-1781, 2493-2500` |
| approx_kl | `last_approx_kl` | 日志指标，当前 policy 相对 reference 的 masked mean logprob 差 | 每个 PPO step 日志 | 仅记录，不直接进 chemistry total | No | `scripts/train_ppo.py:2579-2593` |

## 7. 当前 total reward 公式

### 7.1 真实代码公式

`_reward_from_breakdown(...)` 的总和公式非常直接，见 `src/rewards/reward_wrapper.py:895-902`：

```text
total_reward
= format_r
+ valid_r
+ subgraph_r
+ length_r
+ sem_r
```

关键点：

- `fragment_teacher_sem_r` 不进入 total。
- 成功路径里，`sem_r` 被设置成 `counterfactual_reward`，见 `src/rewards/reward_wrapper.py:772-780`。
- 同一个 `_build_breakdown(...)` 里又把 `teacher_sem_r` 也写成了同一个 `semantic_reward`，见 `src/rewards/reward_wrapper.py:884-892`。

所以成功路径等价于：

```text
total_reward
= format_reward
+ valid_reward
+ subgraph_pass_reward
+ length_reward
+ counterfactual_reward
```

其中：

```text
counterfactual_reward
= teacher_sem_scale * counterfactual_sem
counterfactual_sem
= cf_drop + flip_bonus * int(cf_flip)
```

### 7.2 这意味着什么

- `teacher_sem` 这个名字当前很容易误导人，因为在 reward log 里它不是 fragment-level teacher score。
- 当前真正进 total 的语义项，是 deletion-based 的 counterfactual reward。
- fragment-level teacher score 目前只是 `fragment_teacher_sem`，只做日志/trace，不做 additive total。

### 7.3 如何解释你给的日志例子

给定你举的例子：

```text
format=0.25
valid=1.0
sub=1.0
len=0.25
teacher_sem=-0.9933
counterfactual_sem=-5.0
total=-3.4933
```

如果把这 6 项全加起来，确实会得到：

```text
0.25 + 1.0 + 1.0 + 0.25 - 0.9933 - 5.0 = -3.4933
```

但这不是当前代码的 total 公式。基于当前代码：

- total 只会加 `format + valid + sub + len + sem_r`
- 成功路径里 `sem_r == counterfactual_reward`
- `fragment_teacher_sem_r` 不进入 total

所以在当前实现下：

1. 如果 `counterfactual_sem = -5.0` 真正进入 total，且前四项是 `0.25/1/1/0.25`，那么 total 应该是 `-2.5`。
2. 如果某条日志里还出现 `teacher_sem = -0.9933`，它要么是旧版本日志，要么是把 fragment-level teacher 分量手工又加了一次。
3. 更重要的是，当前 `reward_logs["teacher_sem"]` 并不是 fragment teacher 分量；真正的 fragment-level score 在 `reward_logs["fragment_teacher_sem"]`。

结论：如果组会上还会展示类似 `total=-3.4933` 的日志，请明确注明那不是当前代码路径的实时 total 公式，或者是旧日志。

## 8. teacher_sem

### 8.1 teacher_path 从哪里传入

- CLI 参数：`--teacher-path`，默认等于 `DEFAULT_TEACHER_PATH = DEFAULT_ORACLE_PATH`，见 `scripts/train_ppo.py:53-54, 117-120`
- main 中构建 `TeacherSemanticScorer`：`scripts/train_ppo.py:2755-2774`
- `train_ppo.sh` 和 smoke test 脚本也都会显式传 `--teacher-path "${TEACHER_PATH}"`，见 `scripts/slurm/train_ppo.sh:43-51`、`scripts/slurm/debug_decoded_chem_ppo_smoketest.sh:129-144`

### 8.2 teacher 文件格式

- 首选：sklearn bundle
  - 要求是 `dict`，且包含 `model`、`fingerprint_radius`、`fingerprint_bits`
  - `model` 必须实现 `predict_proba(...)`
  - 代码：`src/rewards/teacher_semantic.py:90-107`、`src/rewards/reward_calculator.py:46-88`
- 兼容：torch payload
  - 只有当 payload 明确暴露 fingerprint 配置时才接受
  - 代码：`src/rewards/teacher_semantic.py:118-185`

### 8.3 是否 sklearn bundle

当前默认就是 sklearn bundle 路径 `outputs/hpc/oracle/aids_rf_model.pkl`，`docs/decisions.md:61-65` 也明确写了当前优先支持 scikit-learn style bundle。

### 8.4 Morgan fingerprint 如何构造

- `TeacherSemanticScorer.score_smiles(...)` 会调用：

```text
smiles_to_morgan_array(
    normalized_smiles,
    radius=fingerprint_radius,
    n_bits=fingerprint_bits,
    clean_dummy_atoms=False,
)
```

- 代码：`src/rewards/teacher_semantic.py:236-241`
- 实际 Morgan 生成逻辑在 `src/rewards/reward_calculator.py:171-219`

### 8.5 teacher_prob 如何得到

- 若底层 model 有 `predict_proba`：
  - `probabilities = model.predict_proba(fingerprint.reshape(1, -1))[0]`
  - `teacher_prob = probabilities[int(label)]`
  - 代码：`src/rewards/teacher_semantic.py:255-285, 298-301`

### 8.6 `teacher_sem = 2 * p_y - 1`

是，精确代码是：

```text
teacher_prob = float(probabilities[int(label)])
teacher_sem = float(2.0 * teacher_prob - 1.0)
```

见 `src/rewards/teacher_semantic.py:283-286`。

### 8.7 `p_y` 到底是什么

这里的 `p_y` 是：

```text
p_y = p(original_label | core_fragment)
```

不是：

```text
p(original_label | parent_without_fragment)
```

原因：

- `TeacherSemanticScorer.score_smiles(...)` 只吃一个 `smiles`
- 传入的是 `core_fragment_smiles`
- 传入的 `label` 是 `original_label`
- `parent_smiles` 参数在 scorer 内部被显式 `del parent_smiles, meta` 丢掉，见 `src/rewards/teacher_semantic.py:187-194`

### 8.8 当前 teacher_sem 是 fragment-level 还是 deletion-based

是 fragment-level，不是 deletion-based。

## 9. counterfactual_sem 当前状态

### 9.1 是否已经实现 deletion oracle

已经实现。

- effective scorer：`CounterfactualTeacherScorer.score_counterfactual(...)`，见 `src/rewards/counterfactual_oracle.py:194-363`
- effective deletion helper：`delete_one_substructure(...)`，见 `src/rewards/counterfactual_oracle.py:75-167`

### 9.2 是否计算 `parent_without_fragment_smiles`

是。删除成功后会产生 `parent_without_fragment_smiles`，见 `src/rewards/counterfactual_oracle.py:165-167, 304-363`。

### 9.3 是否计算 `p_before / p_after / cf_drop / cf_flip`

是，代码如下：

```text
p_before = teacher(parent_smiles)[label]
p_after = teacher(parent_without_fragment_smiles)[label]
cf_drop = p_before - p_after
cf_flip = pred_after != label
counterfactual_sem = cf_drop + flip_bonus * int(cf_flip)
```

对应实现见 `src/rewards/counterfactual_oracle.py:276-343`。

### 9.4 是否进入 total reward

是。

- `_score_counterfactual_semantic(...)` 返回 `counterfactual_reward`
- 成功路径把 `semantic_reward=counterfactual_reward`
- `_reward_from_breakdown(...)` 把 `sem_r` 加进 total

代码串联见 `src/rewards/reward_wrapper.py:754-780, 884-902`。

### 9.5 当前是否尚未完成

不是。就当前 repo 而言，deletion-based counterfactual oracle 已经接进 total reward。

但需要额外说明两点：

1. 当前 deletion 是“删掉第一处匹配到的 substructure 实例”，见 `src/rewards/counterfactual_oracle.py:123-138`。
2. `reward_wrapper.py` 里还有一套更复杂的 residual helper 没被当前主路径用上，见 `src/rewards/reward_wrapper.py:1127-1291`。

## 10. PPO update

### 10.1 当前使用 TRL 原生 PPO step 还是 local PPO loss

当前 `decoded_chem` 路径用的是 local PPO loss，不是 TRL 原生 `step()`。

- 直接日志：`[DECODED_CHEM_PPO_API] Using local PPO loss ...`，见 `scripts/train_ppo.py:2181-2184`
- local loop 主体：`scripts/train_ppo.py:2157-2607`

### 10.2 `policy_model / reference_model / value_model`

- `policy_model`
  - 基座：4-bit ChemLLM
  - 适配器：SFT LoRA checkpoint
  - 可训练
  - 代码：`scripts/train_ppo.py:1097-1134, 2714-2721`
- `reference_model`
  - 同一基座 + 同一 SFT adapter
  - 冻结
  - 代码：`scripts/train_ppo.py:1097-1134, 2722-2729`
- `value_model`
  - `AutoModelForCausalLMWithValueHead.from_pretrained(base_model)`
  - 只训练 `v_head`，其余参数全冻结
  - 代码：`scripts/train_ppo.py:710-768, 2730-2739`

### 10.3 value head

- `value_model` 通过 `AutoModelForCausalLMWithValueHead` 构造，见 `scripts/train_ppo.py:727-735`
- `v_head` 被保留为唯一 trainable 部分，见 `scripts/train_ppo.py:751-759`
- 计算 values 时直接取最后一层 hidden states 过 `v_head`，见 `scripts/train_ppo.py:1742-1760`

### 10.4 reward tensor 如何进入 update

1. sequence-level chemistry reward：
   - `reward_tensor, reward_logs = chem_rewarder.compute_rewards_from_decoded(...)`
   - 代码：`scripts/train_ppo.py:2321-2334`
2. 变成 token-level reward：
   - 先给所有 response token 加上 KL penalty：`-kl_coef * (policy_logprobs - ref_logprobs)`
   - 再把 sequence reward 加到最后一个有效 response token 上
   - 代码：`scripts/train_ppo.py:1763-1781, 2493-2500`
3. 再做 return / advantage：
   - `returns = discounted_cumsum(token_rewards)`
   - `advantages = returns - old_values`
   - 代码：`scripts/train_ppo.py:2501-2518`

### 10.5 `policy_loss / value_loss / total_loss / approx_kl` 含义

- `policy_loss`
  - 标准 clipped PPO surrogate
  - 代码：`scripts/train_ppo.py:2546-2556`
- `value_loss`
  - clipped value regression loss
  - 代码：`scripts/train_ppo.py:2558-2569`
- `total_loss`
  - `policy_loss + 0.5 * value_loss`
  - 代码：`scripts/train_ppo.py:2521, 2571-2574`
- `approx_kl`
  - 当前实现记录的是 `masked_mean(current_logprobs - ref_logprobs)`
  - 它更像“当前 policy 相对 reference model 的平均 logprob 差”，不是 old/new policy 之间的 textbook KL
  - 代码：`scripts/train_ppo.py:2579-2581`

### 10.6 smoke test 的 `batch_size / mini_batch_size / max_steps`

`scripts/slurm/debug_decoded_chem_ppo_smoketest.sh:129-144` 只显式覆盖了：

- `--max-steps 2`
- `--max-prompt-examples 8`

`batch_size` 和 `mini_batch_size` 没在脚本里显式传，所以走 parser/config 逻辑：

- `batch_size`
  - parser 默认 64：`scripts/train_ppo.py:143-148`
  - 但 `configs/base.yaml` 里 `training.batch_size: 1`，而 `apply_config_overrides(...)` 会在参数仍是默认值时用配置覆盖它，见 `configs/base.yaml:37-43`、`scripts/train_ppo.py:347-352`
  - 因此 smoke test 实际 `args.batch_size = 1`
- `mini_batch_size`
  - parser 默认 4：`scripts/train_ppo.py:149-154`
  - configs 里没有覆盖项
  - 之后被 `actual_mini_batch_size = max(1, min(args.mini_batch_size, actual_batch_size))` clamp 成 1，见 `scripts/train_ppo.py:2705-2708`

所以 smoke test 的“表面配置”可以写成：

- `batch_size = 1`
- `mini_batch_size = 1`
- `max_steps = 2`

但还要补一句：

- decoded local PPO loop 实际没有使用 `mini_batch_size`，因为 `run_decoded_chem_ppo_loop(...)` 根本不接它；`mini_batch_size` 只是在外层计算和日志里出现，见 `scripts/train_ppo.py:2157-2171, 2705-2708`

### 10.7 checkpoint 输出路径

- smoke test 显式输出到：
  - `outputs/hpc/rl_checkpoints/debug_decoded_chem_ppo`
  - 代码：`scripts/slurm/debug_decoded_chem_ppo_smoketest.sh:133-144`
- 普通 `train_ppo.sh` 没显式传 `--output-dir`，默认走：
  - `configs/hpc.yaml paths.output_root` + `/rl_checkpoints`
  - 代码：`configs/hpc.yaml:6-10`、`scripts/train_ppo.py:339-342`
- decoded local PPO 最终保存内容：
  - `policy_model.save_pretrained(output_dir)`
  - `tokenizer.save_pretrained(output_dir)`
  - `decoded_chem_value_head.pt`
  - 代码：`scripts/train_ppo.py:1802-1819, 2601-2607`

## 11. 组会 FAQ（15 个）

1. **问：当前 PPO 的输入到底是什么？**  
   答：输入是 plain-text prompt，不是 chat messages。核心字段是 `query`、`parent_smiles`、`original_label`、`input_ids`、`attention_mask`。

2. **问：label 会不会输入给模型？**  
   答：默认从原始 HIV CSV 构建 prompt 时会输入 `ORIGINAL_LABEL`，但如果是预写好的 JSONL prompt，则取决于文件本身是否包含该行。

3. **问：模型被要求输出什么？**  
   答：模型被要求输出“一个 connected substructure 的 SMILES，且只输出 SMILES，不带解释文本”。

4. **问：当前 PPO prompt 用了 chat template 吗？**  
   答：没有。当前是直接 tokenization 的 plain text。

5. **问：为什么 reward 能接受 `*CC(=O)O` 这种带星号的输出？**  
   答：因为当前 reward 把 `*` 视为 attachment point / dummy atom，不是纯垃圾字符；raw view 保留它，core view 再图级删除。

6. **问：`raw fragment` 和 `core fragment` 有什么区别？**  
   答：`raw fragment` 是模型原始输出，可能带 `*`；`core fragment` 是删掉 dummy atom 后、真正用于 substructure check 和 teacher 输入的 dummy-free 分子。

7. **问：substructure check 是对 raw fragment 还是 core fragment 做的？**  
   答：对 core fragment 做。当前不会拿带 `*` 的 raw fragment 直接做 parent-substructure。

8. **问：当前 reward 分量有哪些？**  
   答：format、valid、substructure、length、counterfactual semantic；另外还有 fragment-level teacher semantic，但它当前不进 total。

9. **问：reward 权重现在是多少？**  
   答：默认是 format `+0.25/-0.25`，valid `+1.0/0.3/-2.0`，substructure `+1.0/-2.0`，length `+0.25` 起步，teacher/counterfactual missing penalty `-5.0`，flip bonus `+1.0`。

10. **问：teacher 模型到底是什么？**  
    答：当前默认就是一个带 `predict_proba` 的 sklearn bundle，里面还要显式带 Morgan fingerprint 的 `radius` 和 `bits`。

11. **问：`teacher_sem` 和 `counterfactual_sem` 的区别是什么？**  
    答：`teacher_sem` 是 fragment-level 的 `2 * p(original_label | core_fragment) - 1`；`counterfactual_sem` 是 deletion-based 的 `p_before - p_after + flip_bonus * I(flip)`。

12. **问：当前是不是严格反事实？**  
    答：reward 主项已经是 deletion-based，方向上是反事实；但实现仍有“只删第一处匹配”“删除 helper 双实现”等工程层面的近似与漂移风险。

13. **问：PPO 现在到底怎么更新？**  
    答：不是 TRL `step()`；当前是本地 PPO loss，sequence reward 被加到最后一个有效 response token，上面再叠一层 reference-policy KL penalty。

14. **问：当前日志如果 `fragment_teacher_sem` 很好但 total 很差，说明什么？**  
    答：说明 fragment 自身可能“像原类”，但 deletion-based residual 没有有效打掉原类概率；按当前实现，总奖励还是以 counterfactual term 为主。

15. **问：离“类别级低冗余 selector”还差什么？**  
    答：还差 class-level/group-level 目标函数、跨样本 redundancy 约束、selector 式候选池与筛选机制，以及按类别统计的覆盖度/重复率评估；当前仍是单样本单 fragment 的生成式 PPO。

## 12. PPT 图建议

1. **decoded chemistry PPO 流程图**  
   画成 10 步：`parent_smiles + label -> prompt -> tokenization -> ChemLLM generate -> decode -> fragment extraction -> raw/core normalization -> chemistry checks -> teacher/counterfactual scoring -> PPO update`。建议把 `decoded_chem` 和 `trl_experimental` 分成两条颜色不同的分支，强调本次汇报只讲前者。

2. **raw fragment -> core fragment 图**  
   用一个具体例子：`*CC(=O)O -> CC(=O)O`。左边展示 raw text，右边展示 dummy atom 删除后的 core；旁边标 `dummy_count=1 / raw_parse_ok=True / core_parse_ok=True`。

3. **reward components breakdown 图**  
   用 stacked bar 或 waterfall。把 `format / valid / sub / len / counterfactual_sem` 画成进入 total 的条，再把 `fragment_teacher_sem` 画成旁边的灰色 diagnostic 条，明确“记录但不进 total”。

4. **teacher_sem 计算流程图**  
   画成：`core_fragment_smiles -> Morgan fingerprint -> predict_proba -> p(original_label|fragment) -> teacher_sem = 2p_y - 1`。旁边标一句“fragment-level diagnostic only”。

5. **fragment-level teacher_sem vs deletion-based counterfactual reward 对比图**  
   两列对照：
   - 左列：`fragment -> teacher(fragment)`  
   - 右列：`parent -> delete(fragment) -> residual -> teacher(parent), teacher(residual)`  
   底部写“当前 total 用右列，不用左列”。

6. **当前阶段 -> 下一阶段路线图**  
   三段式：
   - 当前：single-sample decoded PPO，counterfactual reward 已接上  
   - 下一步：清理日志命名、统一 deletion oracle、补强诊断  
   - 再下一步：class-level low-redundancy selector / grouped objective / diversity-coverage evaluation

## 附：几条组会上值得主动先说的“易混点”

- 当前 `teacher_sem` 这个日志名容易被误读。真正 fragment-level 的那一项叫 `fragment_teacher_sem`。
- 当前 prompt 没有显式告诉模型 dummy atom / attachment point 规则，`*` 是 reward path 后处理认可的语法，不是 prompt 里教会模型的 contract。
- 当前 smoke test 的 `mini_batch_size` 基本只是配置噪音；decoded local PPO 没有真的按 minibatch 切。
- 当前语义主项已经是 deletion-based，但工程上还存在“未使用的旧 helper / 兼容层 / 冗余路径”，后续最好清理。
