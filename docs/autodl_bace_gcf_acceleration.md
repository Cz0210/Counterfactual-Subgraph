# AutoDL BACE GCFExplainer 等价加速与低显存共享槽

## 边界

本路线只适用于 AutoDL，不提交或连接 HPC。已有的 50,000-step VRRW 进程和输出保持只读；优化运行必须使用 fresh root。MPS 默认且始终关闭。

优化不改变 `M=50000`、seed、父分子顺序、edit action 枚举、sampling/random 调用次数、transition 插入顺序或 frozen BACE GINE。

首次 m500 replay 已明确失败：legacy 和 ordered-v2 的 RNG 末态及第一个 transition 相同，但第二个 transition 不同。逐项回放证明 5,913 个 action、graph tensor 和并行 RDKit decode 顺序完全一致；差异来自 555 个 valid row 中有 244 个重复 row，旧优化将其去重为 311 个 SMILES 并按 256/55 分批，而 legacy 一次性评价完整 555-row batch。官方 VRRW 以 raw embedding bytes 作为图 identity，因此改变 batch shape 造成的低位差异会改变 transition。

修复后的 `ordered_v2` 只做以下等价变换：

- 按原位置恢复结果的 CPU 邻居构造；
- 按原顺序保留重复 row，并使用与 legacy 完全相同的单个 valid-row GINE batch；
- 仅对完整、顺序相同且 call context 相同的 importance batch 做缓存；任何 partial miss 均原样调用完整 batch；
- 每固定 step 输出一次缓冲进度和分阶段计时。

明确禁止 canonical-SMILES 行级去重、partial-row importance cache 和 GINE chunking。这三项会改变 native raw-byte identity，不是安全优化。

## 快速诊断 replay

修复后先运行 fresh 50/100-step diagnostic replay：

```bash
AUTODL_PHYSICAL_GPU_UUID=<由 exp_run 注入> \
AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python \
BACE_GCF_DATASET_DIR=<frozen BACE GCF dataset> \
GCF_OFFICIAL_ROOT=<vendored official root> \
BACE_GINE_CHECKPOINT=<frozen calibrated GINE bundle> \
BACE_NEUROSED_CHECKPOINT=<repaired NeuroSED checkpoint> \
BACE_NEUROSED_MANIFEST=<matching projection manifest> \
BACE_GCF_QUICK_REPLAY_OUTPUT=<fresh persistent root> \
scripts/autodl/run_bace_gcf_quick_replay.sh
```

输出 `QUICK_REPLAY_PASS.json` 始终带有
`diagnostic_only=true` 和 `eligible_for_full_acceleration_gate=false`。即使
50/100 均 PASS，也不能替代下面正式的 500/1000 gate。

Quick profile 还会写入 `lockstep_trace.json`。它逐次记录 restart、
importance 和 move 的 RNG 状态、canonical 输入顺序、GINE collated batch
tensor、hidden/logit row digest、coverage 以及最终动作。gate 在两侧均有该文件
时报告 `lockstep_comparison.first_divergence`，包含第一个 event、step 和字段；
不再只用最终 coverage 猜测原因。trace 只允许 M=50/100，且包装器不调用 RNG、
不修改返回值。

Frozen-GINE 评分统一走 `FrozenGINEBatchScorer`。默认 cache capacity 为 0；
显式启用时只缓存完整且顺序完全相同的 batch。partial hit、行级 dedup 和
chunking 仍然禁止。CPU/GPU 数值与 repeated-cold raw-byte identity 可用以下
有界诊断生成 JSON（该结果不授权替换 VRRW）：

```bash
python scripts/autodl/benchmark_bace_frozen_gine_batch.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir <frozen BACE dataset> \
  --checkpoint-dir <frozen calibrated GINE> \
  --output-dir <fresh root>
```

`scripts/slurm/benchmark_bace_frozen_gine_batch.sh` 是同步的 HPC wrapper。
正式推理吞吐矩阵另使用
`scripts/autodl/benchmark_bace_gnn_inference_matrix.py`，固定覆盖 batch
`1,8,32,128,512`，分别报告 ordered collation/device transfer、prepared-batch
pure model 和 collation-to-logits end-to-end 的 CPU/GPU median/p95/rows/s。
每个 batch 同时报告 argmax、hidden/logit max-absolute-difference、allclose 和
repeated raw-byte digest；temperature-scaled logits 还逐 batch 报告 calibrated
softmax probability 的 finite、normalized、allclose、最大绝对误差和 digest，
顶层汇总最佳 end-to-end device/batch/rows-per-second。输出固定为
`bace_gnn_inference_benchmark.json`。
benchmark 完成 PASS 只表示所有 argmax/allclose 与 CPU repeat gate 通过，CUDA
raw-byte repeat 另由 `exact_replay_status` fail closed，不能授权 VRRW 替换。
配对 HPC 入口为 `scripts/slurm/benchmark_bace_gnn_inference_matrix.sh`。
正式 benchmark 强制
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` 和
`TOKENIZERS_PARALLELISM=false`；CLI、paired Slurm、controller manifest、输入
manifest 与 child environment 同时绑定这些值。

AutoDL GPU2 benchmark 不跟踪 ComRecGC 的瞬时 science PID。持久入口
`scripts/autodl/run_deferred_bace_gnn_inference_benchmark.py` 绑定 pair run ID、
`launch_spec.json` SHA-256、pair worker PID/start-ticks、registry terminal state、
GPU index/UUID 和 immutable execution commit。它每 60 秒发布 heartbeat；pair
仍运行、UUID exclusive lock 未释放、GPU 仍有 compute PID 或资源不足时保持
`WAITING_RESOURCE`。只有所有条件连续满足 60 秒，才同时取得 project slot 与
UUID exclusive lock，建立一次 fresh benchmark run 并启动 child。预先存在的
output root/run-state、provenance 不一致或 PID 复用在非 terminal registry 下均
`BLOCKED`，不会续写。child 继承两个 lock FD，因此 controller 意外退出也不会
在 benchmark 仍运行时提前释放 GPU 所有权。

AutoDL 上的 `scripts/autodl/run_bace_gcf_lockstep_quick50.sh` 先运行该
benchmark，再按 legacy-A、legacy-B、ordered-v2 顺序运行 fresh Quick-50；
legacy-A/B 不完全一致时会立即停止，不运行 ordered-v2。
benchmark 显式兼容冻结 BACE GCF cohort 的 `canonical_smiles` 字段，并保留
通用数据记录的 `smiles` / `original_smiles` 确定性回退顺序。
scorer identity 同时接受项目 `MolecularGraphData` 的 tuple-backed portable
features 与 collated torch tensor；两者都按 dtype、shape 和 C-order bytes 绑定。

## CUDA raw-byte 非确定性与 CPU lockstep

GPU2 的 64-row repeated-cold 审计证明：同一进程、同一 checkpoint、同一完整
有序 batch 的 GINE GPU hidden/logits 五次调用产生五组不同 SHA-256；预测标签
完全一致、`allclose=true`、最大 logit 差 `3.5762786865234375e-07`。相同 CPU
调用则 hidden/logits 均 5/5 byte-exact。输入顺序、重复 row、batch shape、RNG
都未改变，因此该结果是 CUDA `index_add_` message aggregation/pooling 的 raw-byte
非确定性证据，不是 ordered neighbour、dedup、cache 或 batch 语义 bug。由于官方
VRRW 直接 hash raw embedding bytes，GPU exact replay 不能据此发布 PASS。

CPU-only lockstep 使用：

```bash
CUDA_VISIBLE_DEVICES="" \
BACE_GCF_CPU_LOCKSTEP_OUTPUT=<fresh root> \
scripts/autodl/run_bace_gcf_cpu_lockstep.sh
```

该路线把 GINE 和 NeuroSED 都固定到 CPU，不获取 GPU lock。它按 Quick-50
legacy-A、legacy-B、ordered-v2，再按相同顺序运行 Quick-100；任一逐调用 bit
exact gate 失败会立即停止。官方 `importance.call` 对 CPU 错误调用
`torch.cuda.set_device('cpu')`，项目 compatibility context 只将这一 housekeeping
调用变为 no-op，并在退出时恢复；模型、图、RNG 与 transition 不变。

Quick-50/100 的四个 exact gate 全部 PASS 后，正式 CPU M=500 诊断可使用
`scripts/autodl/run_bace_gcf_cpu_m500.sh`。它先验证 Quick terminal manifest 的
SHA-256、PASS 文件和四个 evidence hash，之后才创建 fresh root，顺序执行
`legacy_m500` 与 `patched_m500`。该路线固定 `profile=smoke`、`parent_limit=64`、
`M=500`、seed 13、CPU GINE/NeuroSED；不运行 M=1000、不获取 GPU lock，也不
授权替换旧 50k。canonical transition/candidate/RNG equivalence 通过后才最后写
`PASS`。

## 50k 前置硬门禁

先在同一物理 GPU 上顺序执行 legacy/ordered-v2 的 500 和 1000 step fresh smoke：

```bash
AUTODL_PHYSICAL_GPU_UUID=<由 exp_run 注入> \
AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python \
BACE_GCF_DATASET_DIR=<frozen BACE GCF dataset> \
GCF_OFFICIAL_ROOT=<vendored official root> \
BACE_GINE_CHECKPOINT=<frozen calibrated GINE bundle> \
BACE_NEUROSED_CHECKPOINT=<repaired NeuroSED checkpoint> \
BACE_NEUROSED_MANIFEST=<matching projection manifest> \
BACE_GCF_ACCELERATION_OUTPUT=<fresh persistent root> \
scripts/autodl/run_bace_gcf_acceleration_smokes.sh
```

只有以下条件同时成立，才生成 `GCF_ACCELERATION_GATE.json: PASS`：

- 500 和 1000 的 canonical graph transition 序列、candidate 顺序/频率/importance/coverage、图集合与 Python/NumPy/Torch RNG 末态完全一致；
- legacy 与 optimized 使用同一 GPU UUID；
- optimized wall-clock throughput 至少提升 20%；
- peak reserved VRAM 不超过物理卡的 70%；
- optimized config fingerprint 在两个预算间一致。

full `ordered_v2` 必须显式传入该 gate；否则 fail closed。未通过 gate 时继续保留 legacy 50k 结果，不以近似输出替换。

## shared-lowmem 调度接口

任务 manifest 新增：

```json
{
  "resource": "gpu",
  "gpu_lock_mode": "shared_lowmem_slot_0",
  "gpu_memory_reservation_mb": 12000,
  "gpu_shared_workload_class": "bace_gcfexplainer_vrrw",
  "gpu_colocation_gate": "/abs/persistent/GPU_COLOCATION_BENCHMARK_GATE.json",
  "gpu_colocation_gate_sha256": "<sha256>"
}
```

这里的 gate 与 GCF `ordered_v2` 自身的 legacy-vs-optimized 性能 gate 不同；
它必须来自真实的单任务/同卡双任务 10--15 分钟 A/B。要求 aggregate throughput
至少提升 20%、canonical result 与 scientific config 不变、无 OOM/error、CPU
不持续饱和、磁盘无明显抖动、MPS 关闭。详见
`docs/AUTODL_ACCELERATION_RELEASE_GATES.md`。

`gpu_lock_mode` 只能为 `exclusive`、`shared_lowmem_slot_0`、`shared_lowmem_slot_1`。同一 GPU 最多两个 shared task；共享 task 全生命周期持有 legacy UUID 文件的 shared advisory lock，因此与历史/新 exclusive task 互斥。第二个任务只有在现存 GPU compute PID 能由 active shared-slot metadata 精确解释时才可加入：slot 记录 launcher child PID 及 Linux `/proc` start-ticks，CUDA PID 可以是该 child 或其经完整父链验证的后代。父链断裂、PID 复用或无法读取 `/proc` 均 fail closed。准入使用 `max(nvidia-smi used, active reservations) + new reservation <= 70% total VRAM`，并在真正 worker acquire 时再次原子检查。

首次占用空卡仍需连续空闲 60 秒。共享槽不启用 MPS，也不允许 `CUDA_MPS*` 环境变量。A/B gate PASS 仅证明该优化配置可候选部署，不授权停止正在运行的 legacy 任务。

## profiling 产物

每个 instrumented VRRW fresh root 包含：

- `performance_profile.json`：random-walk wall time、phase time/calls、steps/s、RSS、peak GPU memory、adapter cache counters；
- `equivalence_trace.json`：去除进程随机 Python hash 后的 canonical transition/candidate/RNG digest；
- `lockstep_trace.json`（仅 Quick-50/100）：第一处分歧所需的逐调用精确证据；
- 原有 `counterfactuals.pt`、`run_manifest.json`、`_RUN_COMPLETE.json`。

本轮未提交、连接或修改任何 HPC 作业；新增 benchmark CLI 的 paired Slurm
仅是静态可审查入口。现有 full wrapper 仍显式固定为 legacy 模式。
