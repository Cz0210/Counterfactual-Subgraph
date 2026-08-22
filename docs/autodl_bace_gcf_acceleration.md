# AutoDL BACE GCFExplainer 等价加速与低显存共享槽

## 边界

本路线只适用于 AutoDL，不提交或连接 HPC。已有的 50,000-step VRRW 进程和输出保持只读；优化运行必须使用 fresh root。MPS 默认且始终关闭。

优化不改变 `M=50000`、seed、父分子顺序、edit action 枚举、sampling/random 调用次数、transition 插入顺序或 frozen BACE GINE。`ordered_v2` 只做以下等价变换：

- 按原位置恢复结果的 CPU 邻居构造；
- canonical graph/lineage 对应的 RDKit 特征、frozen-GINE 与 NeuroSED coverage 结果 LRU 缓存；
- 固定大小 GINE batch；
- 每固定 step 输出一次缓冲进度和分阶段计时。

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
  "gpu_memory_reservation_mb": 12000
}
```

`gpu_lock_mode` 只能为 `exclusive`、`shared_lowmem_slot_0`、`shared_lowmem_slot_1`。同一 GPU 最多两个 shared task；共享 task 全生命周期持有 legacy UUID 文件的 shared advisory lock，因此与历史/新 exclusive task 互斥。第二个任务只有在现存 GPU compute PID 能由 active shared-slot metadata 精确解释时才可加入。准入使用 `max(nvidia-smi used, active reservations) + new reservation <= 70% total VRAM`，并在真正 worker acquire 时再次原子检查。

首次占用空卡仍需连续空闲 60 秒。共享槽不启用 MPS，也不允许 `CUDA_MPS*` 环境变量。A/B gate PASS 仅证明该优化配置可候选部署，不授权停止正在运行的 legacy 任务。

## profiling 产物

每个 instrumented VRRW fresh root 包含：

- `performance_profile.json`：random-walk wall time、phase time/calls、steps/s、RSS、peak GPU memory、adapter cache counters；
- `equivalence_trace.json`：去除进程随机 Python hash 后的 canonical transition/candidate/RNG digest；
- 原有 `counterfactuals.pt`、`run_manifest.json`、`_RUN_COMPLETE.json`。

AutoDL-only 指令与仓库默认 Slurm 同步规则在本轮冲突，因此没有新增 HPC wrapper；没有连接、提交或修改任何 HPC 作业。
