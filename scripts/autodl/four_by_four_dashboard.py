#!/usr/bin/env python3
"""Read-only web/terminal dashboard for persistent AutoDL controllers.

This module contains the collection and presentation logic.  The public CLI is
``scripts/autodl/serve_four_by_four_dashboard.py``.  It deliberately exposes
only a fixed HTML page, one JSON status endpoint, and a health endpoint.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any
from urllib.parse import urlsplit

from scripts.autodl import run_four_gpu_recovery_controller as engine
from scripts.autodl import status_four_gpu_recovery as status_engine
from src.utils.autodl_runtime import AutoDLRuntimeError, read_json_object, utc_now


DEFAULT_NAMESPACE = "four_methods_four_datasets_continuation"
DEFAULT_STALE_SECONDS = 180.0
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ACTIVE_TASK_STATES = {"STARTING", "RUNNING"}
READY_TASK_STATES = {"READY", "WAITING_RESOURCE"}
WAITING_TASK_STATES = {"WAITING_DEPENDENCY", "NOT_STARTED"}


@dataclass(frozen=True)
class ControllerDiscovery:
    controller_ids: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DashboardServerConfig:
    layout: Any
    namespace: str = DEFAULT_NAMESPACE
    interval_seconds: float = 5.0
    stale_seconds: float = DEFAULT_STALE_SECONDS


def _age_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _physical_kind(path: Path, expected: str) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return False
    if expected == "dir":
        return stat.S_ISDIR(info.st_mode)
    return stat.S_ISREG(info.st_mode)


def _physical_path_chain(path: Path) -> tuple[bool, str | None]:
    """Reject symlinks in every existing component without resolving them."""

    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except OSError as exc:
            return False, f"路径不可读：{current}（{type(exc).__name__}）"
        if stat.S_ISLNK(info.st_mode):
            return False, f"拒绝符号链接路径：{current}"
    return True, None


def discover_controller_ids(control_root: Path, namespace: str) -> ControllerDiscovery:
    """Discover physical controller directories below one exact namespace."""

    if not SAFE_COMPONENT.fullmatch(namespace):
        raise AutoDLRuntimeError(f"Unsafe controller namespace: {namespace!r}")
    namespace_root = control_root.expanduser().absolute() / namespace
    valid_chain, chain_error = _physical_path_chain(namespace_root)
    if not valid_chain:
        return ControllerDiscovery((), (chain_error or "Controller 路径不可读",))
    if not _physical_kind(namespace_root, "dir"):
        return ControllerDiscovery((), (f"Controller 命名空间不是物理目录：{namespace_root}",))

    controller_ids: list[str] = []
    warnings: list[str] = []
    try:
        entries = sorted(os.scandir(namespace_root), key=lambda item: item.name)
    except OSError as exc:
        return ControllerDiscovery(
            (),
            (f"无法扫描 Controller 命名空间：{type(exc).__name__}",),
        )
    for entry in entries:
        if entry.is_symlink():
            warnings.append(f"已拒绝符号链接 Controller 候选：{entry.name}")
            continue
        if not SAFE_COMPONENT.fullmatch(entry.name):
            warnings.append(f"已忽略不安全的目录名：{entry.name}")
            continue
        if not entry.is_dir(follow_symlinks=False):
            continue
        candidate = namespace_root / entry.name
        manifest_path = candidate / "controller_manifest.json"
        state_path = candidate / "controller_state.json"
        heartbeat_path = candidate / "heartbeat.json"
        if not _physical_kind(manifest_path, "file"):
            continue
        if not (
            _physical_kind(state_path, "file")
            or _physical_kind(heartbeat_path, "file")
        ):
            warnings.append(f"Controller 缺少状态和心跳：{entry.name}")
            continue
        try:
            manifest = read_json_object(manifest_path)
        except AutoDLRuntimeError as exc:
            warnings.append(f"Controller manifest 无法读取：{entry.name}（{exc}）")
            continue
        if manifest.get("controller_id") != entry.name:
            warnings.append(f"Controller ID 与目录不一致：{entry.name}")
            continue
        controller_ids.append(entry.name)
    return ControllerDiscovery(tuple(controller_ids), tuple(warnings))


def _controller_freshness(
    controller: Mapping[str, Any], *, stale_seconds: float
) -> dict[str, Any]:
    pid = controller.get("pid")
    identity = controller.get("process_identity")
    process_alive = False
    if isinstance(pid, int) and isinstance(identity, Mapping):
        process_alive = engine.process_identity_matches(identity, pid)
    heartbeat_age = controller.get("heartbeat_age_seconds")
    if not isinstance(heartbeat_age, (int, float)):
        heartbeat_age = _age_seconds(controller.get("heartbeat_at"))
    heartbeat_stale = heartbeat_age is None or heartbeat_age > stale_seconds
    if not process_alive:
        freshness = "PROCESS_MISSING"
    elif heartbeat_stale:
        freshness = "STALE"
    else:
        freshness = "FRESH"
    return {
        "process_alive": process_alive,
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_stale": heartbeat_stale,
        "freshness": freshness,
    }


def _flatten_tasks(controllers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in controllers:
        controller = payload.get("controller") or {}
        controller_id = str(controller.get("controller_id") or "-")
        for task in payload.get("queue") or []:
            instances = task.get("instances") or []
            common = {
                "controller_id": controller_id,
                "task_id": task.get("task_id"),
                "dataset": task.get("dataset"),
                "stage": task.get("stage"),
                "status": task.get("state"),
                "gate": task.get("gate"),
                "reason": task.get("reason"),
                "priority": task.get("priority"),
                "updated_at": task.get("updated_at"),
            }
            if not instances:
                rows.append(
                    {
                        **common,
                        "instance_id": None,
                        "run_id": None,
                        "gpu_index": None,
                        "gpu_uuid": None,
                        "worker_pid": None,
                        "child_pid": None,
                        "duration_seconds": None,
                        "heartbeat_age_seconds": None,
                        "output_root": None,
                        "attempt": None,
                    }
                )
                continue
            for instance in instances:
                rows.append(
                    {
                        **common,
                        "instance_id": instance.get("instance_id"),
                        "run_id": instance.get("run_id"),
                        "gpu_index": instance.get("gpu_index"),
                        "gpu_uuid": instance.get("gpu_uuid"),
                        "worker_pid": instance.get("worker_pid"),
                        "child_pid": instance.get("child_pid"),
                        "duration_seconds": instance.get("duration_seconds"),
                        "heartbeat_age_seconds": instance.get(
                            "heartbeat_age_seconds"
                        ),
                        "output_root": instance.get("output_root"),
                        "attempt": instance.get("attempt"),
                        "adopted": instance.get("adopted", False),
                    }
                )
    return rows


def collect_dashboard_snapshot(
    layout: Any,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    gpu_collector: Callable[
        [Any], tuple[list[dict[str, Any]], str | None]
    ] = status_engine.collect_gpu_status,
    controller_collector: Callable[..., dict[str, Any]] = status_engine.collect_status,
) -> dict[str, Any]:
    """Collect all current controllers while probing GPUs exactly once."""

    discovery = discover_controller_ids(layout.control_root, namespace)
    shared_gpu_status = gpu_collector(layout)
    status_engine.CONTROLLER_NAME = namespace
    controllers: list[dict[str, Any]] = []
    errors: list[str] = []
    for controller_id in discovery.controller_ids:
        try:
            payload = controller_collector(
                layout,
                controller_id=controller_id,
                shared_gpu_status=shared_gpu_status,
            )
        except (AutoDLRuntimeError, OSError, ValueError) as exc:
            errors.append(
                f"Controller 状态读取失败：{controller_id}（{type(exc).__name__}: {exc}）"
            )
            continue
        payload["freshness"] = _controller_freshness(
            payload.get("controller") or {}, stale_seconds=stale_seconds
        )
        controllers.append(payload)

    controllers.sort(
        key=lambda item: (
            item.get("freshness", {}).get("freshness") != "FRESH",
            str(item.get("controller", {}).get("controller_id")),
        )
    )
    tasks = _flatten_tasks(controllers)
    logical_tasks = [
        task
        for controller in controllers
        for task in (controller.get("queue") or [])
    ]
    task_counts = Counter(
        str(row.get("state") or "UNKNOWN") for row in logical_tasks
    )
    for row in tasks:
        age = row.get("heartbeat_age_seconds")
        row["heartbeat_stale"] = bool(
            row.get("status") in ACTIVE_TASK_STATES
            and (not isinstance(age, (int, float)) or age > stale_seconds)
        )
    gpu_rows = [dict(item) for item in shared_gpu_status[0]]
    for gpu in gpu_rows:
        index = gpu.get("gpu_index")
        gpu["task_ids"] = sorted(
            {
                str(row["task_id"])
                for row in tasks
                if row.get("status") in ACTIVE_TASK_STATES
                and row.get("gpu_index") == index
                and row.get("task_id")
            }
        )

    warnings = list(discovery.warnings)
    if shared_gpu_status[1]:
        warnings.append(f"GPU 状态读取失败：{shared_gpu_status[1]}")
    fresh_controllers = sum(
        item.get("freshness", {}).get("freshness") == "FRESH"
        for item in controllers
    )
    overall_status = "RUNNING" if task_counts.get("RUNNING", 0) else "ATTENTION"
    if not controllers:
        overall_status = "NO_CONTROLLERS"
    elif (
        task_counts.get("FAILED", 0)
        or task_counts.get("BLOCKED", 0)
        or any(
            item.get("freshness", {}).get("freshness") != "FRESH"
            or item.get("controller", {}).get("state") in {"FAILED", "BLOCKED"}
            for item in controllers
        )
    ):
        overall_status = "ATTENTION"
    return {
        "schema_version": "autodl_four_by_four_dashboard_v1",
        "sampled_at": utc_now(),
        "read_only": True,
        "source": "dynamic_controller_discovery",
        "namespace": namespace,
        "namespace_root": str(layout.control_root / namespace),
        "overall_status": overall_status,
        "summary": {
            "controllers": len(controllers),
            "fresh_controllers": fresh_controllers,
            "tasks": len(logical_tasks),
            "running": task_counts.get("RUNNING", 0),
            "ready": sum(task_counts.get(item, 0) for item in READY_TASK_STATES),
            "waiting": sum(
                task_counts.get(item, 0) for item in WAITING_TASK_STATES
            ),
            "blocked": task_counts.get("BLOCKED", 0),
            "failed": task_counts.get("FAILED", 0),
            "passed": task_counts.get("PASS", 0),
        },
        "controllers": controllers,
        "tasks": tasks,
        "gpus": gpu_rows,
        "gpu_error": shared_gpu_status[1],
        "warnings": warnings,
        "errors": errors,
    }


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'">
<title>AutoDL 四卡实验监控</title>
<style>
:root{color-scheme:dark;background:#0f141d;color:#e8edf5;font:14px system-ui,-apple-system,sans-serif}body{margin:22px}h1{font-size:24px;margin:0}.sub{color:#9aa7bb;margin:6px 0 16px}.badge{display:inline-block;padding:3px 7px;border-radius:4px;background:#20483a;color:#8fe0bd;font-size:12px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:12px 0 18px}.card,.panel{background:#181f2b;border:1px solid #2d384b;border-radius:9px;padding:12px}.label{color:#9aa7bb;font-size:12px}.value{font-size:18px;margin-top:4px;word-break:break-word}.scroll{overflow:auto;border:1px solid #2d384b;border-radius:9px;margin-bottom:18px}table{width:100%;border-collapse:collapse;background:#181f2b}th,td{padding:9px;border-bottom:1px solid #2d384b;text-align:left;vertical-align:top;white-space:nowrap}th{color:#a8b4c7;position:sticky;top:0;background:#181f2b}.wrap{white-space:normal;min-width:240px;max-width:520px;word-break:break-all}.RUNNING,.STARTING,.FRESH,.PASS{color:#5dd986}.READY,.WAITING_RESOURCE,.WAITING_DEPENDENCY,.NOT_STARTED{color:#efc85d}.FAILED,.BLOCKED,.STALE,.PROCESS_MISSING,.ATTENTION,.NO_CONTROLLERS{color:#ff7f8b}.small{font-size:12px;color:#9aa7bb}.warning{white-space:pre-wrap;color:#ffc077}.ok{color:#62d9ca}footer{margin-top:18px;color:#77859a}code{color:#c6d7f4}h2{font-size:19px;margin-top:22px}
</style></head><body>
<h1>AutoDL 四卡实验监控 <span class="badge">只读</span></h1>
<div class="sub"><span id="namespace">加载中…</span> · <span id="connection">正在连接</span></div>
<div class="cards">
 <div class="card"><div class="label">总体 status</div><div id="overall" class="value">UNKNOWN</div></div>
 <div class="card"><div class="label">控制器</div><div id="controller-count" class="value">-</div></div>
 <div class="card"><div class="label">任务统计</div><div id="task-count" class="value">-</div></div>
 <div class="card"><div class="label">服务器采样时间</div><div id="sampled" class="value">-</div></div>
 <div class="card"><div class="label">页面更新时间</div><div id="page-updated" class="value">-</div></div>
</div>
<h2>控制器状态</h2>
<div class="scroll"><table><thead><tr><th>控制器</th><th>status</th><th>工作负载 status</th><th>PID</th><th>进程</th><th>心跳年龄</th><th>数据新鲜度</th><th>任务计数</th></tr></thead><tbody id="controllers"></tbody></table></div>
<h2>GPU</h2>
<div class="scroll"><table><thead><tr><th>GPU</th><th>型号</th><th>利用率</th><th>显存</th><th>锁状态</th><th>计算 PID</th><th>当前任务</th></tr></thead><tbody id="gpus"></tbody></table></div>
<h2>任务与依赖队列</h2>
<div class="scroll"><table><thead><tr><th>控制器</th><th>数据集</th><th>任务</th><th>阶段</th><th>status</th><th>门禁</th><th>GPU</th><th>Run ID</th><th>工作进程 / 子进程 PID</th><th>运行时长</th><th>心跳年龄</th><th>输出目录</th><th>阻塞 / 失败原因</th></tr></thead><tbody id="tasks"></tbody></table></div>
<h2>只读诊断</h2><div id="diagnostics" class="panel warning">无</div>
<footer>页面只提供 GET 读取，不包含启动、停止、恢复或 shell 接口。请通过 SSH 隧道访问，不要将端口直接暴露到公网。</footer>
<script>
const INTERVAL=__REFRESH_MS__;let timer=null;let request=null;
const txt=v=>v===null||v===undefined||v===''?'-':String(v);
const td=(tr,v,cls='')=>{const e=document.createElement('td');e.textContent=txt(v);if(cls)e.className=cls;tr.appendChild(e)};
const localTime=v=>{if(!v)return '-';const d=new Date(v);return Number.isNaN(d.valueOf())?txt(v):d.toLocaleString('zh-CN',{hour12:false})};
const age=v=>typeof v==='number'?Math.round(v)+' 秒':'-';
const duration=v=>typeof v==='number'?Math.round(v)+' 秒':'-';
function render(d){
 document.getElementById('namespace').textContent=d.namespace_root;
 const o=document.getElementById('overall');o.textContent=d.overall_status;o.className='value '+d.overall_status;
 document.getElementById('controller-count').textContent=d.summary.fresh_controllers+' 新鲜 / '+d.summary.controllers+' 总计';
 document.getElementById('task-count').textContent=d.summary.running+' 运行 · '+d.summary.ready+' 就绪 · '+d.summary.waiting+' 等待 · '+d.summary.failed+' 失败 · '+d.summary.blocked+' 阻塞';
 document.getElementById('sampled').textContent=localTime(d.sampled_at);
 document.getElementById('page-updated').textContent=new Date().toLocaleString('zh-CN',{hour12:false});
 const cb=document.getElementById('controllers');cb.replaceChildren();
 for(const item of d.controllers||[]){const c=item.controller||{},f=item.freshness||{},tr=document.createElement('tr');td(tr,c.controller_id);td(tr,c.state,c.state);td(tr,c.workload_state||c.state,c.workload_state||c.state);td(tr,c.pid);td(tr,f.process_alive?'存在':'不存在',f.process_alive?'FRESH':'PROCESS_MISSING');td(tr,age(f.heartbeat_age_seconds),f.heartbeat_stale?'STALE':'FRESH');td(tr,f.freshness,f.freshness);td(tr,JSON.stringify(c.task_counts||{}),'wrap');cb.appendChild(tr)}
 const gb=document.getElementById('gpus');gb.replaceChildren();
 for(const g of d.gpus||[]){const tr=document.createElement('tr');td(tr,g.gpu_index);td(tr,g.name||g.gpu_name||'-');td(tr,txt(g.utilization_gpu_percent)+'%');td(tr,txt(g.memory_used_mb)+' / '+txt(g.memory_total_mb)+' MiB');td(tr,g.lock_state,g.lock_state);td(tr,(g.compute_pids||[]).join(', ')||'-');td(tr,(g.task_ids||[]).join(', ')||'-','wrap');gb.appendChild(tr)}
 const tb=document.getElementById('tasks');tb.replaceChildren();
 for(const x of d.tasks||[]){const tr=document.createElement('tr');td(tr,x.controller_id);td(tr,x.dataset);td(tr,x.task_id);td(tr,x.stage);td(tr,x.status,x.status);td(tr,x.gate,x.gate);td(tr,x.gpu_index);td(tr,x.run_id);td(tr,txt(x.worker_pid)+' / '+txt(x.child_pid));td(tr,duration(x.duration_seconds));td(tr,(x.heartbeat_stale?'STALE · ':'')+age(x.heartbeat_age_seconds),x.heartbeat_stale?'STALE':'');td(tr,x.output_root,'wrap');td(tr,x.reason,'wrap');tb.appendChild(tr)}
 const messages=[...(d.warnings||[]),...(d.errors||[])];document.getElementById('diagnostics').textContent=messages.length?messages.join('\n'):'无';
}
function schedule(ms=INTERVAL){clearTimeout(timer);timer=setTimeout(refresh,ms)}
async function refresh(){clearTimeout(timer);if(request)request.abort();request=new AbortController();const timeout=setTimeout(()=>request.abort(),Math.max(4000,INTERVAL));try{const r=await fetch('/api/status',{cache:'no-store',signal:request.signal});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json());const c=document.getElementById('connection');c.textContent='连接正常';c.className='ok';schedule()}catch(e){const c=document.getElementById('connection');c.textContent='刷新失败：'+e.message;c.className='warning';schedule(Math.max(INTERVAL,10000))}finally{clearTimeout(timeout);request=null}}
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')refresh()});window.addEventListener('online',refresh);refresh();
</script></body></html>"""


def render_terminal(snapshot: Mapping[str, Any]) -> str:
    summary = snapshot.get("summary") or {}
    lines = [
        f"AutoDL 四卡实验监控 sampled_at={snapshot.get('sampled_at')}",
        f"namespace={snapshot.get('namespace_root')}",
        (
            f"status={snapshot.get('overall_status')} controllers="
            f"{summary.get('fresh_controllers')}/{summary.get('controllers')} "
            f"tasks={summary.get('tasks')} running={summary.get('running')} "
            f"ready={summary.get('ready')} waiting={summary.get('waiting')} "
            f"failed={summary.get('failed')} blocked={summary.get('blocked')}"
        ),
        "",
        "CONTROLLERS",
    ]
    for item in snapshot.get("controllers") or []:
        controller = item.get("controller") or {}
        freshness = item.get("freshness") or {}
        lines.append(
            f"{controller.get('controller_id')} status={controller.get('state')} "
            f"workload={controller.get('workload_state', controller.get('state'))} "
            f"pid={controller.get('pid')} heartbeat_age="
            f"{freshness.get('heartbeat_age_seconds')} freshness={freshness.get('freshness')}"
        )
    lines.extend(("", "GPU"))
    for gpu in snapshot.get("gpus") or []:
        lines.append(
            f"GPU {gpu.get('gpu_index')} util={gpu.get('utilization_gpu_percent')}% "
            f"memory={gpu.get('memory_used_mb')}/{gpu.get('memory_total_mb')}MiB "
            f"lock={gpu.get('lock_state')} pids={gpu.get('compute_pids')} "
            f"tasks={gpu.get('task_ids')}"
        )
    diagnostics = list(snapshot.get("warnings") or []) + list(
        snapshot.get("errors") or []
    )
    if diagnostics:
        lines.extend(("", "诊断"))
        lines.extend(f"- {message}" for message in diagnostics)
    return "\n".join(lines) + "\n"


def is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def make_handler(
    config: DashboardServerConfig,
    *,
    snapshot_provider: Callable[..., dict[str, Any]] = collect_dashboard_snapshot,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoDLFourByFourReadOnly/1"

        def _send(
            self,
            status_code: int,
            content_type: str,
            body: bytes,
            *,
            allow: str | None = None,
        ) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            content_policy = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
                if content_type.startswith("text/html")
                else "default-src 'none'; frame-ancestors 'none'"
            )
            self.send_header("Content-Security-Policy", content_policy)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            if allow:
                self.send_header("Allow", allow)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                refresh_ms = max(1000, int(config.interval_seconds * 1000))
                body = PAGE.replace("__REFRESH_MS__", str(refresh_ms)).encode(
                    "utf-8"
                )
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", body)
                return
            if path == "/api/status":
                try:
                    snapshot = snapshot_provider(
                        config.layout,
                        namespace=config.namespace,
                        stale_seconds=config.stale_seconds,
                    )
                    body = json.dumps(
                        snapshot, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    self._send(
                        HTTPStatus.OK, "application/json; charset=utf-8", body
                    )
                except (AutoDLRuntimeError, OSError, ValueError) as exc:
                    body = json.dumps(
                        {
                            "status": "ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "application/json; charset=utf-8",
                        body,
                    )
                return
            if path == "/healthz":
                self._send(
                    HTTPStatus.OK,
                    "text/plain; charset=utf-8",
                    b"autodl four-by-four dashboard ok\n",
                )
                return
            self._send(
                HTTPStatus.NOT_FOUND,
                "text/plain; charset=utf-8",
                b"not found\n",
            )

        def _method_not_allowed(self) -> None:
            self._send(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "text/plain; charset=utf-8",
                b"read-only: GET only\n",
                allow="GET",
            )

        do_POST = _method_not_allowed
        do_PUT = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_HEAD = _method_not_allowed
        do_OPTIONS = _method_not_allowed

        def log_message(self, format_string: str, *args: Any) -> None:
            path = urlsplit(self.path).path
            sys.stderr.write(
                f"dashboard request from {self.client_address[0]}: "
                f"{self.command} {path}\n"
            )

    return Handler
