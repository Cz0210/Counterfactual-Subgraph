"""Fresh AIDS postprocessing that adopts a completed exact DBSCAN read-only."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping

from scripts.autodl.run_comrecgc_standardized_continuation import (
    _validate_common_recourse_completion,
    run_continuation,
)
from src.baselines.comrecgc.contracts import sha256_file, write_json
from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (
    EXACT_STAGE,
    load_bound_controller_manifest,
    validate_stage_terminal,
)
from src.utils.autodl_aids_comrecgc_exact_recovery_stages_v1 import (
    _continuation_inputs,
)


POSTPROCESS_HEARTBEAT_SCHEMA = "aids_comrecgc_exact_postprocess_heartbeat_v1"
MAX_POSTPROCESS_WORKERS = 8


class AIDSExactPostprocessError(RuntimeError):
    """The exact result cannot be safely adopted into fresh postprocessing."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _bounded_cpu_environment(max_workers: int) -> Iterator[None]:
    if (
        isinstance(max_workers, bool)
        or not 1 <= int(max_workers) <= MAX_POSTPROCESS_WORKERS
    ):
        raise AIDSExactPostprocessError(
            f"max_workers must be in [1, {MAX_POSTPROCESS_WORKERS}]"
        )
    workers = str(int(max_workers))
    values = {
        "AIDS_POSTPROCESS_MAX_WORKERS": workers,
        "CUDA_VISIBLE_DEVICES": "",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "OMP_NUM_THREADS": workers,
        "MKL_NUM_THREADS": workers,
        "OPENBLAS_NUM_THREADS": workers,
        "NUMEXPR_NUM_THREADS": workers,
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _exact_receipt_path(manifest: Mapping[str, Any]) -> Path:
    matches = [
        Path(str(stage["terminal_path"]))
        for stage in manifest["stages"]
        if stage.get("stage_id") == EXACT_STAGE
    ]
    if len(matches) != 1:
        raise AIDSExactPostprocessError("exact stage terminal is not unique")
    return matches[0].resolve(strict=True)


class _Heartbeat:
    def __init__(
        self,
        *,
        path: Path,
        controller_manifest: Path,
        exact_receipt: Path,
        output_root: Path,
        max_workers: int,
        interval_seconds: float,
    ) -> None:
        self.path = path
        self.controller_manifest = controller_manifest
        self.exact_receipt = exact_receipt
        self.output_root = output_root
        self.max_workers = int(max_workers)
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.stop_event = threading.Event()
        self.state = "VALIDATING_EXACT_TERMINAL"
        self.detail: dict[str, Any] = {}
        self.thread: threading.Thread | None = None

    def publish(self) -> None:
        write_json(
            self.path,
            {
                "schema_version": POSTPROCESS_HEARTBEAT_SCHEMA,
                "pid": os.getpid(),
                "state": self.state,
                "controller_manifest_path": str(self.controller_manifest),
                "controller_manifest_sha256": sha256_file(
                    self.controller_manifest
                ),
                "exact_receipt_path": str(self.exact_receipt),
                "exact_receipt_sha256": sha256_file(self.exact_receipt),
                "output_root": str(self.output_root),
                "gpu_used": False,
                "max_workers": self.max_workers,
                "dbscan_rerun": False,
                "detail": dict(self.detail),
                "updated_at": _utc_now(),
            },
        )

    def start(self) -> None:
        self.publish()

        def _run() -> None:
            while not self.stop_event.wait(self.interval_seconds):
                self.publish()

        self.thread = threading.Thread(
            target=_run,
            name="aids-exact-postprocess-heartbeat",
            daemon=True,
        )
        self.thread.start()

    def update(self, state: str, **detail: Any) -> None:
        self.state = state
        self.detail = dict(detail)
        self.publish()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def run_aids_exact_postprocess(
    *,
    controller_manifest_path: str | Path,
    exact_receipt_path: str | Path,
    output_root: str | Path,
    heartbeat_path: str | Path,
    resume: bool,
    max_workers: int = MAX_POSTPROCESS_WORKERS,
    heartbeat_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    """Validate exact science, adopt DBSCAN, and run the existing continuation."""

    if (
        isinstance(max_workers, bool)
        or not 1 <= int(max_workers) <= MAX_POSTPROCESS_WORKERS
    ):
        raise AIDSExactPostprocessError(
            f"max_workers must be in [1, {MAX_POSTPROCESS_WORKERS}]"
        )
    controller_logical = Path(controller_manifest_path).expanduser()
    receipt_logical = Path(exact_receipt_path).expanduser()
    output_logical = Path(output_root).expanduser()
    heartbeat_logical = Path(heartbeat_path).expanduser()
    if not all(
        path.is_absolute()
        for path in (
            controller_logical,
            receipt_logical,
            output_logical,
            heartbeat_logical,
        )
    ):
        raise AIDSExactPostprocessError("authority paths must be absolute")
    if any(
        path.is_symlink()
        for path in (
            controller_logical,
            receipt_logical,
            output_logical,
            heartbeat_logical,
        )
    ):
        raise AIDSExactPostprocessError("authority paths may not be symlinks")
    controller_path = controller_logical.resolve(strict=True)
    receipt_path = receipt_logical.resolve(strict=True)
    output = output_logical.resolve(strict=False)
    heartbeat_file = heartbeat_logical.resolve(strict=False)
    if output.is_symlink() or heartbeat_file.is_symlink():
        raise AIDSExactPostprocessError(
            "output and heartbeat paths may not be symlinks"
        )
    if heartbeat_file == output or output in heartbeat_file.parents:
        raise AIDSExactPostprocessError(
            "heartbeat authority must remain outside the scientific output root"
        )
    if output.exists() and not resume:
        raise AIDSExactPostprocessError(
            "fresh postprocess output already exists; use explicit --resume"
        )
    heartbeat = _Heartbeat(
        path=heartbeat_file,
        controller_manifest=controller_path,
        exact_receipt=receipt_path,
        output_root=output,
        max_workers=int(max_workers),
        interval_seconds=heartbeat_interval_seconds,
    )
    with _bounded_cpu_environment(int(max_workers)):
        heartbeat.start()
        try:
            manifest = load_bound_controller_manifest(controller_path)
            expected_receipt = _exact_receipt_path(manifest)
            if receipt_path != expected_receipt:
                raise AIDSExactPostprocessError(
                    "exact receipt is not the controller-bound terminal"
                )
            exact = validate_stage_terminal(manifest, stage_id=EXACT_STAGE)
            receipt = exact["stage_receipt"]
            dbscan_manifest = Path(
                str(receipt["dbscan_manifest_path"])
            ).resolve(strict=True)
            if (
                receipt.get("run_complete") is not True
                or receipt.get("dbscan_partition_proven") is not True
                or receipt.get("dbscan_manifest_sha256")
                != sha256_file(dbscan_manifest)
            ):
                raise AIDSExactPostprocessError(
                    "controller-bound exact terminal did not prove its DBSCAN partition"
                )
            heartbeat.update(
                "EXACT_TERMINAL_VALIDATED",
                dbscan_manifest_path=str(dbscan_manifest),
                dbscan_manifest_sha256=sha256_file(dbscan_manifest),
            )
            values = replace(
                _continuation_inputs(manifest, output),
                external_dbscan_source_manifest=dbscan_manifest,
                external_dbscan_source_receipt=receipt_path,
                common_recourse_resume=True,
            )
            heartbeat.update(
                "RUNNING_FRESH_POSTPROCESS",
                dbscan_manifest_path=str(dbscan_manifest),
                dbscan_manifest_sha256=sha256_file(dbscan_manifest),
            )
            terminal = run_continuation(values)
            common_terminal_path = output / "common_recourse/_RUN_COMPLETE.json"
            common_terminal = json.loads(
                common_terminal_path.read_text(encoding="utf-8")
            )
            _validate_common_recourse_completion(
                marker=common_terminal_path, terminal=common_terminal
            )
            if (
                terminal.get("status") != "PASS"
                or terminal.get("run_complete") is not True
                or (output / "PASS").read_bytes() != b"PASS\n"
            ):
                raise AIDSExactPostprocessError(
                    "fresh standardized continuation did not publish PASS last"
                )
            result = {
                "status": "PASS",
                "run_complete": True,
                "pid": os.getpid(),
                "output_root": str(output),
                "continuation_terminal_path": str(output / "_RUN_COMPLETE.json"),
                "continuation_terminal_sha256": sha256_file(
                    output / "_RUN_COMPLETE.json"
                ),
                "common_terminal_path": str(common_terminal_path),
                "common_terminal_sha256": sha256_file(common_terminal_path),
                "dbscan_source_manifest_path": str(dbscan_manifest),
                "dbscan_source_manifest_sha256": sha256_file(dbscan_manifest),
                "dbscan_rerun": False,
                "gpu_used": False,
                "max_workers": int(max_workers),
            }
            heartbeat.update("PASS", **result)
            return result
        except BaseException as exc:
            heartbeat.update(
                "FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            heartbeat.stop()


__all__ = [
    "AIDSExactPostprocessError",
    "MAX_POSTPROCESS_WORKERS",
    "POSTPROCESS_HEARTBEAT_SCHEMA",
    "run_aids_exact_postprocess",
]
