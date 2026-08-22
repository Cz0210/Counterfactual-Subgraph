# AutoDL 四卡实验只读监控站点

## 目标与边界

新版站点只读取持久化 controller 状态、任务实例、GPU UUID lock 和一次
`nvidia-smi` 快照。它没有启动、停止、恢复、发信号或执行任意命令的 HTTP
接口，也不修改科学输出。

数据源不再是固定的旧 run root。每次 API 请求都会自动扫描：

```text
$AUTODL_CONTROL_ROOT/four_methods_four_datasets_continuation/
```

只有同时具有物理 `controller_manifest.json` 以及物理状态/心跳文件，且
manifest 中 `controller_id` 与目录名一致的物理目录才会展示。符号链接候选
会被拒绝并出现在诊断区。这样主 controller、repair controller 和后续 fresh
continuation 可以同时显示。

## 启动

从专用 immutable dashboard execution worktree 启动；不要在正在执行科学任务
的 worktree 内改代码：

```bash
AUTODL_PROJECT_ROOT=/root/autodl-tmp/worktrees/run-four-by-four-dashboard-<commit> \
AUTODL_DATA_ROOT=/autodl-fs/data \
AUTODL_CONTROL_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime/control \
AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python \
AUTODL_DASHBOARD_PORT=8766 \
scripts/autodl/launch_four_by_four_dashboard.sh
```

默认端口使用 `8766`，从而可以在核验新版期间保留旧的 `8765` 站点。launcher
使用 `nohup`，写入独立 PID 和日志，并在返回前检查 `/healthz`。它不会终止或
覆盖其他进程。

一次性终端/JSON 状态：

```bash
PYTHONPATH=$PWD /root/miniconda3/envs/smiles_pip118/bin/python \
  scripts/autodl/serve_four_by_four_dashboard.py \
  --project-root "$PWD" \
  --data-root /autodl-fs/data \
  --control-root /autodl-fs/data/counterfactual-subgraph-runtime/control \
  once --format table
```

## 本人和朋友访问

站点强制只监听远端 `127.0.0.1`。每位访问者在自己的电脑上建立 SSH
隧道，然后打开自己的本地地址：

```bash
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 18766:127.0.0.1:8766 \
  -p 53731 \
  root@connect.nma1.seetacloud.com
```

浏览器打开：

```text
http://127.0.0.1:18766
```

`127.0.0.1` 在这里指访问者自己的电脑；SSH 把该端口转发到 AutoDL 容器内
的站点。多人可以各自建立隧道，若本地 `18766` 已被占用，可改成其他本地
端口，例如 `28766:127.0.0.1:8766`。

不要把站点改绑 `0.0.0.0`。站点会 fail closed 拒绝非 loopback host，因为
它没有公网身份认证或 TLS，并会显示 PID、路径和任务状态。

共享 `root` 密码等同于授予对方整台 AutoDL 容器的 root shell，而不只是只读
网页权限。更安全的长期方案是为每位朋友配置独立 SSH key，并通过
`authorized_keys` 的 `restrict,port-forwarding,permitopen="127.0.0.1:8766"`
约束只允许该端口转发；不要共享主 root 私钥或长期密码。是否建立这种受限
SSH 身份属于系统权限变更，需要单独审核和授权。

## 刷新与陈旧判断

网页默认每 5 秒重新读取 `/api/status`，请求禁用缓存。切回后台标签页或网络
恢复时会立即刷新；超时后自动重试。页面明确区分：

- 服务器采样时间：本次 API 实际采集时间；
- 页面更新时间：浏览器完成渲染的本地时间；
- Controller 心跳年龄；
- 每个运行任务的 worker 心跳年龄；
- `FRESH`、`STALE` 或 `PROCESS_MISSING`。

旧站点看似很久没有更新，并不是其 HTTP 进程停止。旧进程仍每 5 秒生成新
API 时间，但启动时没有传 `--run`，所以一直读取脚本硬编码的
`/autodl-fs/data/runs/autodl_three_lines_20260821_v1`，并固定显示两个旧 aux
launcher。当前四方法 controller 则位于新的 persistent control namespace，
旧站点从未读取那里。新版删除了该旧 root 默认值，改为动态 controller
发现。

## HTTP 安全契约

允许的路径只有：

```text
GET /
GET /api/status
GET /healthz
```

其他方法返回 `405`。所有响应使用 `no-store`、CSP、frame deny、MIME sniff
保护和 no-referrer。页面仅用 `textContent` 写入状态，不渲染 controller
提供的 HTML。

## AutoDL-only 说明

该站点只服务 AutoDL 持久 controller，不属于 HPC/Slurm 科学任务。本轮没有
新增同名 Slurm wrapper；这是一项有意的安全例外，避免产生一个会在 HPC
节点长期监听 Web 端口的提交脚本。已有 `status_four_by_four.sh` 仍用于原有
只读 CLI parity，本项目不会因此连接或提交 HPC。
