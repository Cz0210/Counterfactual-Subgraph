from __future__ import annotations

import json
from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace

from scripts.autodl import run_four_gpu_recovery_controller as engine
from scripts.autodl import serve_four_by_four_dashboard as cli
from scripts.autodl.four_by_four_dashboard import (
    PAGE,
    DashboardServerConfig,
    collect_dashboard_snapshot,
    discover_controller_ids,
    is_loopback_host,
    make_handler,
)
from src.utils.autodl_runtime import utc_now


NAMESPACE = "four_methods_four_datasets_continuation"


def _controller(root: Path, controller_id: str) -> None:
    path = root / NAMESPACE / controller_id
    path.mkdir(parents=True)
    (path / "controller_manifest.json").write_text(
        json.dumps({"controller_id": controller_id, "tasks": []}) + "\n",
        encoding="utf-8",
    )
    (path / "heartbeat.json").write_text(
        json.dumps({"controller_id": controller_id, "heartbeat_at": utc_now()})
        + "\n",
        encoding="utf-8",
    )


def test_discovers_only_physical_controller_directories(tmp_path: Path) -> None:
    control = tmp_path / "control"
    _controller(control, "main")
    _controller(control, "repair")
    _controller(control, "four_methods_four_datasets_am_repair_v2")
    (control / NAMESPACE / "linked").symlink_to(
        control / NAMESPACE / "main", target_is_directory=True
    )
    mismatch = control / NAMESPACE / "mismatch"
    mismatch.mkdir()
    (mismatch / "controller_manifest.json").write_text(
        '{"controller_id":"someone_else"}\n', encoding="utf-8"
    )
    (mismatch / "controller_state.json").write_text("{}\n", encoding="utf-8")

    discovered = discover_controller_ids(control, NAMESPACE)

    assert discovered.controller_ids == (
        "four_methods_four_datasets_am_repair_v2",
        "main",
        "repair",
    )
    assert any("符号链接" in value and "linked" in value for value in discovered.warnings)
    assert any("不一致" in value and "mismatch" in value for value in discovered.warnings)


def test_dynamic_snapshot_probes_gpu_once_for_all_controllers(
    tmp_path: Path, monkeypatch
) -> None:
    control = tmp_path / "control"
    _controller(control, "main")
    _controller(control, "repair")
    _controller(control, "four_methods_four_datasets_am_repair_v2")
    layout = SimpleNamespace(control_root=control, locks_dir=tmp_path / "locks")
    gpu_calls: list[object] = []
    controller_calls: list[str] = []
    identity = {"pid": os.getpid(), "start_ticks": 1, "command_sha256": "x"}
    monkeypatch.setattr(engine, "process_identity_matches", lambda expected, pid: True)

    def collect_gpus(observed_layout):
        gpu_calls.append(observed_layout)
        return (
            [
                {
                    "gpu_index": 0,
                    "gpu_name": "NVIDIA A800 80GB PCIe",
                    "gpu_uuid": "GPU-zero",
                    "utilization_gpu_percent": 7,
                    "memory_used_mb": 900,
                    "memory_total_mb": 81920,
                    "compute_pids": [456],
                    "lock_state": "LOCKED",
                }
            ],
            None,
        )

    def collect_controller(
        observed_layout, *, controller_id: str, shared_gpu_status
    ):
        assert observed_layout is layout
        assert shared_gpu_status[0][0]["gpu_uuid"] == "GPU-zero"
        controller_calls.append(controller_id)
        status = "RUNNING" if controller_id == "main" else "WAITING_DEPENDENCY"
        return {
            "controller": {
                "controller_id": controller_id,
                "state": "RUNNING",
                "workload_state": "RUNNING",
                "pid": os.getpid(),
                "process_identity": identity,
                "heartbeat_at": utc_now(),
                "heartbeat_age_seconds": 0.0,
                "task_counts": {status: 1},
            },
            "queue": [
                {
                    "task_id": f"{controller_id}_task",
                    "dataset": "bace",
                    "stage": "stage",
                    "state": status,
                    "gate": "NOT_EVALUATED",
                    "reason": "waiting" if controller_id == "repair" else None,
                    "priority": 1,
                    "updated_at": utc_now(),
                    "instances": [
                        {
                            "instance_id": "main",
                            "run_id": f"run-{controller_id}",
                            "gpu_index": 0 if controller_id == "main" else None,
                            "gpu_uuid": "GPU-zero" if controller_id == "main" else None,
                            "worker_pid": 123,
                            "child_pid": 456 if controller_id == "main" else None,
                            "duration_seconds": 12.0,
                            "heartbeat_age_seconds": 1.0,
                            "output_root": f"/persistent/{controller_id}",
                            "attempt": 0,
                        }
                    ],
                }
            ],
        }

    payload = collect_dashboard_snapshot(
        layout,
        namespace=NAMESPACE,
        gpu_collector=collect_gpus,
        controller_collector=collect_controller,
    )

    assert len(gpu_calls) == 1
    assert controller_calls == [
        "four_methods_four_datasets_am_repair_v2",
        "main",
        "repair",
    ]
    assert payload["source"] == "dynamic_controller_discovery"
    assert payload["summary"]["running"] == 1
    assert payload["summary"]["waiting"] == 2
    assert payload["gpus"][0]["task_ids"] == ["main_task"]
    assert all(item["freshness"]["freshness"] == "FRESH" for item in payload["controllers"])


def test_frontend_is_chinese_and_has_resilient_refresh_hooks() -> None:
    for value in (
        "AutoDL 四卡实验监控",
        "服务器采样时间",
        "页面更新时间",
        "控制器状态",
        "任务与依赖队列",
        "阻塞 / 失败原因",
        "只读诊断",
        "GPU",
        "Run ID",
        "status",
        "visibilitychange",
        "window.addEventListener('online'",
        "setTimeout(refresh",
    ):
        assert value in PAGE


def test_dashboard_has_no_legacy_three_line_root() -> None:
    source = Path("scripts/autodl/four_by_four_dashboard.py").read_text(
        encoding="utf-8"
    )
    entrypoint = Path(
        "scripts/autodl/serve_four_by_four_dashboard.py"
    ).read_text(encoding="utf-8")
    assert "autodl_three_lines" not in source
    assert "autodl_three_lines" not in entrypoint
    assert DEFAULT_OLD_ROOT_NOT_PRESENT not in source


DEFAULT_OLD_ROOT_NOT_PRESENT = "/autodl-fs/data/runs/"


def test_http_surface_is_get_only_and_no_store() -> None:
    config = DashboardServerConfig(
        layout=SimpleNamespace(), interval_seconds=5, stale_seconds=180
    )
    snapshot = {
        "sampled_at": utc_now(),
        "namespace_root": "/persistent/control/current",
        "namespace": NAMESPACE,
        "overall_status": "RUNNING",
        "summary": {
            "fresh_controllers": 1,
            "controllers": 1,
            "tasks": 1,
            "running": 1,
            "ready": 0,
            "waiting": 0,
            "failed": 0,
            "blocked": 0,
        },
        "controllers": [],
        "tasks": [],
        "gpus": [],
        "warnings": [],
        "errors": [],
    }

    def provider(layout, *, namespace, stale_seconds):
        assert namespace == NAMESPACE
        assert stale_seconds == 180
        return snapshot

    handler_type = make_handler(config, snapshot_provider=provider)

    def request(path: str, method: str = "GET") -> bytes:
        handler = handler_type.__new__(handler_type)
        handler.path = path
        handler.command = method
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.wfile = BytesIO()
        handler._headers_buffer = []
        getattr(handler, f"do_{method}")()
        return handler.wfile.getvalue()

    page_response = request("/")
    assert b"HTTP/1.0 200 OK" in page_response
    assert b"Cache-Control: no-store, max-age=0" in page_response
    assert b"script-src 'self' 'unsafe-inline'" in page_response
    assert b"const INTERVAL=5000" in page_response
    api_response = request("/api/status")
    assert b'"overall_status":"RUNNING"' in api_response
    assert b"HTTP/1.0 200 OK" in request("/healthz")
    post_response = request("/api/status", "POST")
    assert b"HTTP/1.0 405 Method Not Allowed" in post_response
    assert b"Allow: GET" in post_response


def test_non_loopback_bind_is_rejected_before_layout(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli, "_layout", lambda args: (_ for _ in ()).throw(AssertionError())
    )
    assert (
        cli.main(
            [
                "--project-root",
                str(Path.cwd()),
                "serve",
                "--host",
                "0.0.0.0",
            ]
        )
        == 2
    )
    assert "refuses non-loopback" in capsys.readouterr().err
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")


def test_launcher_is_autodl_only_loopback_and_non_destructive() -> None:
    launcher = Path(
        "scripts/autodl/launch_four_by_four_dashboard.sh"
    ).read_text(encoding="utf-8")
    assert 'HOST="127.0.0.1"' in launcher
    assert "serve_four_by_four_dashboard.py" in launcher
    assert "nohup" in launcher
    assert "healthz" in launcher
    assert "0.0.0.0" not in launcher
    assert "sbatch" not in launcher
    assert "kill " not in launcher
