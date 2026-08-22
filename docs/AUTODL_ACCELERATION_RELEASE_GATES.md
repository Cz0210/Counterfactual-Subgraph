# AutoDL 加速发布门禁

本文记录两个彼此独立、均为 fail-closed 的发布边界。它们不会停止、迁移或
改写任何已经运行的 legacy/diagnostic 任务，也不会自动启动 full 或
`shared_lowmem` 任务。

## 1. BACE ComRecGC optimized 50k

ComRecGC 的 50,000-step random walk 仍是一条有状态轨迹，禁止按 generation
index 强拆。`ordered_bounded_rdkit_process_pool_v1` 只并行纯 RDKit/特征预处理。
正式 optimized full 必须先聚合两个独立 fresh gate root：exact 500 和 exact
1000。聚合器会重新检查而不是只相信 `PASS` 文件：

- equivalence audit schema、self-hash、legacy/optimized roots；
- candidate/trace parity、graph order/state、coverage、float tolerance；
- raw `counterfactuals.pt`、completion/diagnostic markers、trace chunks 及 SHA；
- frozen GINE、distance checkpoint、dataset/cohort、strict-flip、360 parents；
- batch size、worker/inflight/cache 配置在 500/1000/full 三者完全一致；
- full 参数精确等于 preregistered 50,000-step 配置，且不加载 calibration/test。

生成 fresh 聚合 gate：

```bash
python scripts/baselines/comrecgc/build_full_acceleration_gate.py \
  --config configs/hpc.yaml \
  --m500-root /abs/persistent/m500 \
  --m1000-root /abs/persistent/m1000 \
  --output-dir /abs/persistent/full-gate
```

正式 optimized full 必须同时传 gate 文件和文件 SHA：

```bash
python scripts/baselines/comrecgc/run_generation.py \
  --config configs/hpc.yaml \
  --route project --dataset bace --mode full \
  ... \
  --bace-preprocess-engine ordered_bounded_rdkit_process_pool_v1 \
  --bace-acceleration-gate /abs/persistent/full-gate/FULL_ACCELERATION_GATE.json \
  --bace-acceleration-gate-sha256 <sha256>
```

runtime 在创建 fresh output root 前重新读取两个原始 replay root、全部关键
文件和 hash。gate、source audit、raw payload、checkpoint、cohort、batch 或
runtime config 任一漂移都会阻止启动。legacy sequential 50k 不受该 optimized
release gate 影响。

## 2. shared_lowmem co-location

低显存共卡不是看到瞬时 `0%` 利用率后的手工开关。必须先在同一物理 GPU 上
保存两段各 10--15 分钟的单任务 profile，再保存一段相同两任务的共卡 profile。
每秒采样，并为每个 benchmark key 保存 output root、result manifest SHA、
canonical result SHA、scientific config SHA、throughput 和 peak VRAM。

只有以下条件全部成立才发布 `[GPU_COLOCATION_BENCHMARK_PASS]`：

- 两个 single profile 的最大并发为 1，paired profile 的最大并发为 2；
- 三个 profile 使用同一 GPU UUID/model/memory 和同一 throughput metric；
- 每个 profile 的 aggregate throughput 严格等于其 task throughput 之和；
- paired aggregate 相对两个 single throughput 的均值提升至少 20%；
- 相同 benchmark key 的 workload class、scientific config SHA 和 canonical
  result SHA 完全一致；
- OOM/error 均为 0，CPU 无持续饱和，磁盘无明显抖动，MPS 为 false；
- measured peak VRAM 加 safety margin 严格小于总显存的 70%。

发布 gate：

```bash
python scripts/autodl/gate_gpu_colocation_benchmark.py \
  --config configs/hpc.yaml \
  --single-profile /abs/profile-a.json \
  --single-profile /abs/profile-b.json \
  --colocated-profile /abs/profile-pair.json \
  --output-dir /abs/persistent/colocation-gate
```

controller task 必须声明完整契约：

```json
{
  "resource": "gpu",
  "gpu_lock_mode": "shared_lowmem_slot_0",
  "gpu_memory_reservation_mb": 12000,
  "gpu_shared_workload_class": "bace_gcfexplainer_vrrw",
  "gpu_colocation_gate": "/abs/persistent/colocation-gate/GPU_COLOCATION_BENCHMARK_GATE.json",
  "gpu_colocation_gate_sha256": "<sha256>"
}
```

允许的 workload class 只有 BACE GCFExplainer VRRW、BACE ComRecGC generation
和已验证 candidate scoring。controller 在 manifest schema 与 launch 前分别
重验 gate；`exp_run` 在创建 run spec 和 worker acquire 前再次重验。slot metadata
还绑定 exact gate SHA、authorized workload pair 和 science child PID，因此不同
gate、未 benchmark 的两任务组合、低于 measured peak 的 reservation、外部 CUDA
PID 或超过 70% 的组合都不能获得第二个 slot。exclusive/CPU task 携带这些字段
同样会失败。

对应 Slurm 文件仅保持 CLI/parity，可用于静态校验或在 HPC 工作流中显式生成
gate；本次变更不提交作业，也不授权在远端自动启用共卡。
