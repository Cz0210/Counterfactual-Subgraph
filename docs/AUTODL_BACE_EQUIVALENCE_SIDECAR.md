# AutoDL BACE 等价性 Sidecar

主 four-by-four controller 的 manifest 在首次启动后由 SHA 和任务拓扑冻结，
不能追加任务。该 sidecar 只排以下 GCFExplainer 科学诊断：

1. duplicate-preserving quick M=50；
2. quick M=100；
3. formal M=500。

三项均使用全局 exclusive GPU UUID lock、同一个 frozen BACE GINE、同一个
NeuroSED checkpoint、train-only 64-parent profile 和 fresh attempt root。任何
一项失败都会阻止后续项。quick gate 永远是 diagnostic-only；M=500 也不能单独
释放 optimized 50k，仍需要 formal M=1000 和性能 gate。

构建前会只读核验四个受保护任务：legacy GCF full、GlobalGCE v5、现有
ComRecGC M=500 pair、legacy ComRecGC full。构建器不发送信号，也不修改其输出。
现有 ComRecGC M=500 本身已经按 legacy → optimized → audit 串行运行，因此不会
再加入 sidecar。

构建：

```bash
python scripts/autodl/build_bace_equivalence_sidecar_manifest.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --controller-id "$CONTROLLER_ID" \
  --runtime-root "$AUTODL_RUNTIME_ROOT" \
  --python "$AUTODL_PYTHON" \
  --output-root "$OUTPUT_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --build-audit "$BUILD_AUDIT" \
  --dataset-dir "$BACE_GCF_DATASET_DIR" \
  --gcf-official-root "$GCF_OFFICIAL_ROOT" \
  --gine-checkpoint "$BACE_GINE_CHECKPOINT" \
  --neurosed-checkpoint "$BACE_NEUROSED_CHECKPOINT" \
  --neurosed-manifest "$BACE_NEUROSED_MANIFEST"
```

使用通用持久 controller 启动：

```bash
FOUR_BY_FOUR_MANIFEST="$OUTPUT_MANIFEST" \
  scripts/autodl/launch_four_by_four.sh
```

Sidecar 在没有可用 UUID lock 时保持 heartbeat 和 `WAITING_RESOURCE`，不会生成
dummy workload，也不会抢占现有任务。
