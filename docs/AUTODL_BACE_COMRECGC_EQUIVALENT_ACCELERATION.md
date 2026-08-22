# AutoDL BACE ComRecGC 等价加速审计与部署

日期：2026-08-23
范围：BACE ComRecGC 原生 50,000-step generation；不涉及 HPC 或论文目录。

## 结论

不能把 generation index `0..49999` 独立切成 8 份再合并。固定上游
`122f9341a360e9f06bb58a2f5823bb596021f6bf` 的每一步都消费同一 RNG 流，
并依赖/更新共享的 `graph_map`、`transitions`、候选频次与顺序、覆盖状态和
restart 权重。不同 shard seed/root 产生的是八条不同随机游走，不是原始一条
50k trajectory 的等价并行实现，lineage/frequency/order 也没有合法 merge。

安全并行边界位于单 producer 之下：保持随机游走、RNG 和状态更新完全串行，
只把纯 CPU 的 native graph decode + RDKit featurization 放入有序、bounded、
`spawn` process pool。worker 不持有 RNG 或 CUDA context；结果严格按输入顺序
交回主进程。Frozen GINE 仍在主进程按原批次一次评分。

## 只读 profiling 证据

原始科学进程保持运行，未发送 signal、未使用 ptrace、未写其 output root。
20 分钟 evidence 位于：

```text
/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/profiles/
  bace_comrecgc_generation/20260822T180222Z/
```

该轮取得 230 个样本、1200.001 秒；CPU median/p95/max 为
`99.578/99.823/100.012%`，GPU utilization median/p95/max 为
`0/1/1%`，RSS 从 `1,827,404` 增至 `1,861,148 KiB`，日志增长
`4,107,077 bytes`，实际磁盘 read/write median 都为 `0 MiB/s`。
最后一次 `nvidia-smi` 发生单次 4 秒 timeout，因此 collector 正确保留
`INCOMPLETE.json`，没有伪造 profiling PASS；CPU 单核/RDKit 瓶颈结论仍由
其余完整样本支持。

当时 persistent `progress.json` 为 `400/50000`，累计
`17017.64s`、`84.62 steps/hour`，尚未产生第一个 500-step checkpoint。
旧 PID/output 必须继续只读保留。

部署前资源快照为 112 logical CPUs、约 1.081 TB RAM（约 1.013 TB
available）、persistent data 约 155.6 GB free、NVMe 约 32.9 GB free。
因此首个 gate 采用 4 个 RDKit workers，而不是让每个任务占满全机 CPU；cache
保持有界且 checkpoint/output 全部落 persistent root。

## 实现边界

- 默认 engine 仍为 `legacy_sequential_rdkit_v1`；旧命令不被静默加速。
- opt-in engine 为 `ordered_bounded_rdkit_process_pool_v1`。
- source/candidate cache 分离，均为有界 LRU。
- cache 的 global graph content identity 不含 parent metadata；但 decode key
  必须绑定 `source_index`、node lineage、feature schema 和 checkpoint provenance，
  防止跨 parent sidecar 污染。
- legacy engine 拒绝非零 worker/cache 参数。
- diagnostic prefix 只接受 500 或 1000 steps，强制
  `diagnostic_only=true`、`paper_eligible=false`、calibration/test 未加载。
- equivalence auditor 比较候选 topology/order/frequency/importance、graph map、
  index map、traversed hashes、coverage 和 selected-action trace chunks；任何差异
  fail closed，只有全部相同才写 `PASS`。

## Fresh A/B gate

分别使用两个 fresh root 运行 500 与 1000 gate：

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
WT=/root/autodl-tmp/worktrees/run-bace-comrecgc-equivalent-<commit>
COMMON=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --python "$PY"
  --upstream-root "$COMRECGC_ROOT"
  --dataset-dir "$DATASET_DIR"
  --gnn-checkpoint "$GNN_CHECKPOINT"
  --distance-checkpoint "$DISTANCE_CHECKPOINT"
  --workers 4
  --max-inflight 64
  --source-cache-capacity 1024
  --candidate-cache-capacity 8192
)
cd "$WT"
"$PY" scripts/baselines/comrecgc/run_generation_equivalence_pair.py \
  "${COMMON[@]}" --steps 500 --output-dir "$FRESH_GATE_500"
"$PY" scripts/baselines/comrecgc/run_generation_equivalence_pair.py \
  "${COMMON[@]}" --steps 1000 --output-dir "$FRESH_GATE_1000"
```

两个 root 都必须真实存在以下闭包，才能调度 optimized 50k：

```text
PASS
audit/PASS
audit/equivalence_summary.json  (status=PASS)
legacy/_RUN_COMPLETE.json
optimized/_RUN_COMPLETE.json
```

不得复制 marker，不得以单元测试替代真实 BACE replay，不得在旧失败或正在写的
root 中重试。500/1000 任一失败时保留完整 evidence，optimized 50k 保持 BLOCKED。

## Safe checkpoint / controller plan

1. 不停止当前旧 generation。其首次安全边界是 persistent 500-step checkpoint
   及 mirror 均完成且 checksum/`LATEST` 一致；在此之前没有可迁移 checkpoint。
2. 在独立 immutable execution worktree 中先运行静态 shardability audit 和 CPU
   process-pool smoke。
3. 由 shared-lowmem scheduler 依次运行 fresh 500、1000 A/B gate；每个 gate
   内 legacy 与 optimized 使用同一张卡串行，避免同时占用旧科学任务的资源。
4. 只有两个真实 PASS 后，新建 fresh optimized 50k root，使用单 producer、
   workers=4、max_inflight=64、source cache=1024、candidate cache=8192。
5. optimized 50k 仍按 500 steps 写原有 full-state checkpoint 和独立 persistent
   mirror。controller resume 只能从同一 command/config/hash closure 的 `LATEST`
   恢复，不得把旧 legacy checkpoint 注入 optimized root。
6. optimized 产出完全 PASS 前，旧 run 和所有 evidence 均保持只读；不得删除或
   覆盖来腾空间。

静态审计命令：

```bash
"$PY" scripts/baselines/comrecgc/audit_generation_shardability.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --upstream-root "$COMRECGC_ROOT" \
  --output-dir "$FRESH_SHARD_AUDIT_ROOT"
```

状态判定应以 persistent manifest/marker 为准，不以一次 GPU utilization 采样或
目录名为准。
