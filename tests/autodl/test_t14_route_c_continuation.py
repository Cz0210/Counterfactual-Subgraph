from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import pytest

import src.baselines.tastemolnet_t14_route_c_continuation as continuation
from src.baselines.tastemolnet_t14_route_c_continuation import (
    CELL_ID,
    T14RouteCContinuationError,
    build_continuation_spec,
    continuation_command,
    find_fast16_publishers,
    launch_continuation_owner,
    publish_generation_handoff,
    run_continuation,
    write_continuation_spec,
)
from src.baselines.tastemolnet_t14_route_c_fresh import (
    build_spec as build_route_c_spec,
    write_spec as write_route_c_spec,
)
from src.eval.fast16_matrix_authority_pointer import (
    DEFAULT_LOCK_PATH,
    DEFAULT_STATE_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _route_and_continuation(
    tmp_path: Path,
    *,
    guard_deferred_open: pytest.MonkeyPatch | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, Path, Path]:
    attempt = str(uuid4())
    owner_root = tmp_path / f"owner-{attempt}"
    generation_root = tmp_path / f"generation-{attempt}"
    counters = tmp_path / "cgroup"
    counters.mkdir()
    for name, value in (("limit", "1000000"), ("current", "1"), ("failcnt", "0")):
        (counters / name).write_text(value, encoding="utf-8")
    t3_root = tmp_path / "t3"
    route = build_route_c_spec(
        attempt_uuid=attempt,
        execution_commit="1" * 40,
        python=Path(sys.executable).resolve(),
        science_wrapper=REPO_ROOT
        / "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh",
        owner_entrypoint=REPO_ROOT / "scripts/autodl/run_t14_route_c_owner.py",
        output_root=generation_root,
        owner_root=owner_root,
        cgroup_limit_path=counters / "limit",
        cgroup_current_path=counters / "current",
        cgroup_failcnt_path=counters / "failcnt",
        forbidden_legacy_root=tmp_path / "forbidden-legacy-12500",
        science_environment={
            "RUN_TASTEMOLNET": "1",
            "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
            "TASTE_PAPER_RESULTS_ALLOWED": "1",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            "RUN_GNN_ABLATION": "0",
            "RUN_LLM_ABLATION": "0",
            "TASTEMOLNET_T3_OUTPUT_ROOT": str(t3_root),
        },
        max_process_rss_bytes=100,
        launch_headroom_bytes=100,
        runtime_headroom_bytes=50,
    )
    route_path = owner_root / "T14_ROUTE_C_TASK_SPEC.json"
    write_route_c_spec(route_path, route)
    descriptor_path = owner_root / "T14_ROUTE_C_CONTINUATION_SPEC.json"
    locator = tmp_path / "control" / "t14-route-c-cell-root-locator.json"
    matrix_output = tmp_path / "matrix-after-t14"
    queue_path = tmp_path / "control" / "fast16-queue.json"
    _json(
        queue_path,
        {
            "schema_version": "fast16_matrix_publisher_queue_v1",
            "initial_authority_root": str(tmp_path / "authority-12"),
            "authority_state_path": str(DEFAULT_STATE_PATH),
            "authority_lock_path": str(DEFAULT_LOCK_PATH),
            "poll_seconds": 60,
            "taste": {
                "t3_root": str(t3_root),
                "policy_path": str(tmp_path / "taste-policy.yaml"),
                "policy_receipt": str(tmp_path / "taste-policy-receipt.json"),
                "prepared_root": str(tmp_path / "prepared"),
                "graph_cache_root": str(tmp_path / "graph-cache"),
            },
            "cells": [
                {
                    "dataset": "TasteMolNet",
                    "method": "ComRecGC",
                    "terminal_root_locator": str(locator),
                    "output_root": str(matrix_output),
                }
            ],
        },
    )
    calibration = tmp_path / "deferred" / "calibration.csv"
    test = tmp_path / "deferred" / "test.csv"
    environment = {
        **continuation.POSTPROCESS_FIXED_ENVIRONMENT,
        "TASTEMOLNET_CALIBRATION_CSV": str(calibration),
        "TASTEMOLNET_TEST_CSV": str(test),
        "TASTEMOLNET_T3_OUTPUT_ROOT": str(t3_root),
        "MOLCLR_ROOT": str(tmp_path / "molclr"),
        "MOLCLR_CHECKPOINT": str(tmp_path / "molclr.pth"),
        "TASTEMOLNET_WNODE_THRESHOLD_JSON": str(tmp_path / "threshold.json"),
        "WNODE_CACHE_DB": str(tmp_path / "cache" / "wnode.sqlite"),
        "NODE_EMBEDDING_CACHE_DIR": str(tmp_path / "cache" / "nodes"),
        "AUTODL_DATA_ROOT": "/autodl-fs/data",
        "AUTODL_RUNTIME_ROOT": "/autodl-fs/data/counterfactual-subgraph-runtime",
        "AUTODL_CONTROL_ROOT": str(DEFAULT_STATE_PATH.parent.parent),
        "AUTODL_PYTHON": str(Path(sys.executable).resolve()),
    }
    original_open = Path.open
    if guard_deferred_open is not None:
        deferred = {calibration, test}

        def guarded_open(self: Path, *args: object, **kwargs: object):
            if self in deferred:
                raise AssertionError("deferred split was opened while building the spec")
            return original_open(self, *args, **kwargs)

        guard_deferred_open.setattr(Path, "open", guarded_open)
    spec = build_continuation_spec(
        descriptor_path=descriptor_path,
        route_c_spec_path=route_path,
        config_path=REPO_ROOT / "configs/hpc.yaml",
        continuation_entrypoint=REPO_ROOT
        / "scripts/autodl/run_t14_route_c_continuation.py",
        postprocess_wrapper=REPO_ROOT
        / "scripts/autodl/run_tastemolnet_t14_comrecgc_postprocess.sh",
        postprocess_science_root=tmp_path / f"postprocess-science-{attempt}",
        postprocess_final_root=tmp_path / f"postprocess-final-{attempt}",
        locator_path=locator,
        publisher_queue_manifest=queue_path,
        publisher_heartbeat=tmp_path / "control" / "fast16-heartbeat.json",
        publisher_pid_file=tmp_path / "control" / "fast16.pid",
        postprocess_environment=environment,
        poll_seconds=5,
    )
    write_continuation_spec(descriptor_path, spec)
    return route, route_path, spec, descriptor_path, calibration, test


def _generation_terminal(
    generation_root: Path, *, method_cell_pass: bool = False
) -> dict[str, Any]:
    generation_root.mkdir(parents=True)
    (generation_root / "GENERATION_PASS").write_text(
        "[TASTE_T14_COMRECGC_FULL_GENERATION_PASS]\n", encoding="utf-8"
    )
    _json(
        generation_root / "generation_manifest.json",
        {
            "status": "PASS",
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "paper_result_eligible": False,
            "method_cell_pass": method_cell_pass,
        },
    )
    return {
        "status": "PASS",
        "marker": "[TASTE_T14_COMRECGC_FULL_GENERATION_PASS]",
        "output_root": str(generation_root),
        "inventory_sha256": "a" * 64,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "paper_result_eligible": False,
        "method_cell_pass": method_cell_pass,
    }


def test_descriptor_does_not_open_calibration_or_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _route, _route_path, spec, _descriptor, calibration, test = (
        _route_and_continuation(tmp_path, guard_deferred_open=monkeypatch)
    )
    assert not calibration.exists()
    assert not test.exists()
    assert spec["deferred_inputs"] == [
        "TASTEMOLNET_CALIBRATION_CSV",
        "TASTEMOLNET_TEST_CSV",
    ]
    assert spec["deferred_inputs_opened_before_generation_freeze"] is False
    assert spec["generation_pass_is_method_cell_pass"] is False


def test_generation_handoff_is_pending_not_method_pass(tmp_path: Path) -> None:
    route, _route_path, spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    verification = _generation_terminal(Path(route["output_root"]))
    handoff = publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=verification,
        owner_pid=123,
        owner_start_ticks=456,
    )
    terminal = json.loads(
        Path(spec["generation_owner_terminal"]).read_text(encoding="utf-8")
    )
    assert handoff["status"] == "GENERATION_PASS_PENDING_POSTPROCESS"
    assert handoff["method_cell_pass"] is False
    assert terminal["status"] == "GENERATION_PASS_PENDING_POSTPROCESS"
    assert terminal["method_cell_pass"] is False
    assert terminal["paper_result_eligible"] is False
    assert terminal["publisher_started"] is False


def test_generation_handoff_rejects_method_cell_pass(tmp_path: Path) -> None:
    route, _route_path, _spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    verification = _generation_terminal(
        Path(route["output_root"]), method_cell_pass=True
    )
    with pytest.raises(T14RouteCContinuationError, match="method PASS"):
        publish_generation_handoff(
            continuation_spec_path=descriptor,
            generation_verification=verification,
            owner_pid=123,
            owner_start_ticks=456,
        )


def test_generation_handoff_repairs_crash_window_idempotently(tmp_path: Path) -> None:
    route, _route_path, spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    verification = _generation_terminal(Path(route["output_root"]))
    first = publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=verification,
        owner_pid=123,
        owner_start_ticks=456,
    )
    terminal_path = Path(spec["generation_owner_terminal"])
    expected_terminal = terminal_path.read_bytes()

    repeated = publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=verification,
        owner_pid=789,
        owner_start_ticks=987,
    )
    assert repeated == first
    assert terminal_path.read_bytes() == expected_terminal

    # Model a crash after the handoff rename/fsync but before terminal rename.
    terminal_path.unlink()
    repaired = publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=verification,
        owner_pid=789,
        owner_start_ticks=987,
    )
    assert repaired == first
    assert terminal_path.read_bytes() == expected_terminal


def _fake_proc_process(root: Path, pid: int, tokens: list[str]) -> None:
    process = root / str(pid)
    process.mkdir(parents=True)
    tail = ["S", *(["0"] * 18), str(pid + 100)]
    (process / "stat").write_text(
        f"{pid} (python worker) " + " ".join(tail), encoding="utf-8"
    )
    (process / "cmdline").write_bytes(
        b"\0".join(token.encode("utf-8") for token in tokens) + b"\0"
    )


def test_duplicate_fast16_publishers_are_visible(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    tokens = ["python", "scripts/autodl/run_fast16_matrix_publisher_queue.py"]
    _fake_proc_process(proc, 101, tokens)
    _fake_proc_process(proc, 102, tokens)
    assert [row["pid"] for row in find_fast16_publishers(proc_root=proc)] == [101, 102]


def test_only_one_live_publisher_may_claim_t14_among_shared_authority_queues(
    tmp_path: Path,
) -> None:
    _route, _route_path, spec, _descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    matrix = spec["matrix"]
    proc = tmp_path / "proc"
    proc.mkdir()
    target_pid = 17420
    target_tokens = [
        "python",
        str(REPO_ROOT / "scripts/autodl/run_fast16_matrix_publisher_queue.py"),
        "--queue-manifest",
        matrix["queue_manifest"],
        "--heartbeat-path",
        matrix["queue_heartbeat"],
    ]
    _fake_proc_process(proc, target_pid, target_tokens)
    Path(matrix["queue_pid_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(matrix["queue_pid_file"]).write_text(f"{target_pid}\n", encoding="utf-8")
    _json(
        Path(matrix["queue_heartbeat"]),
        {
            "schema_version": "fast16_matrix_publisher_heartbeat_v1",
            "pid": target_pid,
            "queue_manifest_path": matrix["queue_manifest"],
            "queue_manifest_sha256": matrix["queue_manifest_sha256"],
            "authority_state_path": matrix["authority_state_path"],
            "authority_lock_path": matrix["authority_lock_path"],
            "cells": {CELL_ID: {"state": "WAITING"}},
        },
    )
    for pid, cell in ((21156, "TasteMolNet/GCFExplainer"), (22020, "TasteMolNet/NeuroSED")):
        queue = tmp_path / "control" / f"queue-{pid}.json"
        heartbeat = tmp_path / "control" / f"heartbeat-{pid}.json"
        dataset, method = cell.split("/", 1)
        _json(
            queue,
            {
                "schema_version": "fast16_matrix_publisher_queue_v1",
                "authority_state_path": str(DEFAULT_STATE_PATH),
                "authority_lock_path": str(DEFAULT_LOCK_PATH),
                "cells": [{"dataset": dataset, "method": method}],
            },
        )
        _fake_proc_process(
            proc,
            pid,
            [
                "python",
                str(REPO_ROOT / "scripts/autodl/run_fast16_matrix_publisher_queue.py"),
                "--queue-manifest",
                str(queue),
                "--heartbeat-path",
                str(heartbeat),
            ],
        )

    publisher = continuation._validate_unique_publisher(spec, proc_root=proc)
    assert publisher is not None
    assert publisher["pid"] == target_pid
    assert publisher["claimed_cells"] == [CELL_ID]

    duplicate_queue = tmp_path / "control" / "duplicate-t14.json"
    duplicate_heartbeat = tmp_path / "control" / "duplicate-t14-heartbeat.json"
    _json(
        duplicate_queue,
        {
            "schema_version": "fast16_matrix_publisher_queue_v1",
            "authority_state_path": str(DEFAULT_STATE_PATH),
            "authority_lock_path": str(DEFAULT_LOCK_PATH),
            "cells": [{"dataset": "TasteMolNet", "method": "ComRecGC"}],
        },
    )
    _fake_proc_process(
        proc,
        23000,
        [
            "python",
            str(REPO_ROOT / "scripts/autodl/run_fast16_matrix_publisher_queue.py"),
            "--queue-manifest",
            str(duplicate_queue),
            "--heartbeat-path",
            str(duplicate_heartbeat),
        ],
    )
    with pytest.raises(T14RouteCContinuationError, match="multiple live.*claim"):
        continuation._validate_unique_publisher(spec, proc_root=proc)


def test_continuation_orders_freeze_before_deferred_inputs_and_matrix_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route, _route_path, spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    verification = _generation_terminal(Path(route["output_root"]))
    publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=verification,
        owner_pid=123,
        owner_start_ticks=456,
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    events: list[str] = []

    def writers(_root: Path, *, proc_root: str) -> dict[str, Any]:
        events.append("writer_audit")
        return {"procfs_verified": True, "writable_fd_count": 0, "proc_root": proc_root}

    def generation(_root: Path) -> dict[str, Any]:
        events.append("generation_freeze_validation")
        return verification

    def deferred(_spec: dict[str, Any]) -> dict[str, Any]:
        events.append("deferred_inputs_opened")
        return {"status": "OPENED_AFTER_GENERATION_FREEZE"}

    def postprocess(
        _spec: dict[str, Any], *, root: Path, sleep: object
    ) -> None:
        del root, sleep
        events.append("postprocess_and_final_verifier")

    def final(_spec: dict[str, Any]) -> dict[str, Any]:
        events.append("final_verified")
        return {"terminal_root": _spec["postprocess_final_root"]}

    def matrix(_spec: dict[str, Any]) -> dict[str, Any]:
        events.append("matrix_applied")
        return {
            "authority_root": "/authority-13",
            "matrix_complete_cells": 13,
            "matrix_status_sha256": "b" * 64,
            "combined_audit_sha256": "c" * 64,
            "standardized_output_root": _spec["postprocess_final_root"],
        }

    monkeypatch.setattr(continuation, "scan_live_writers", writers)
    monkeypatch.setattr(continuation, "validate_t14_full_output", generation)
    monkeypatch.setattr(continuation, "_open_deferred_inputs", deferred)
    monkeypatch.setattr(continuation, "_run_postprocess", postprocess)
    monkeypatch.setattr(continuation, "_validate_final", final)
    monkeypatch.setattr(continuation, "_matrix_result", matrix)
    result = run_continuation(descriptor, once=True, proc_root=proc)
    assert events == [
        "writer_audit",
        "generation_freeze_validation",
        "deferred_inputs_opened",
        "postprocess_and_final_verifier",
        "final_verified",
        "matrix_applied",
    ]
    assert result["status"] == "PASS"
    assert result["generation_pass_was_method_cell_pass"] is False
    assert result["method_cell_pass"] is True
    assert result["publisher_started"] is False
    assert Path(spec["matrix"]["locator_path"]).is_file()


def test_only_existing_postprocess_and_publisher_queue_are_used(tmp_path: Path) -> None:
    _route, _route_path, spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    assert spec["publisher_started_by_continuation"] is False
    assert Path(spec["postprocess_wrapper"]).name == (
        "run_tastemolnet_t14_comrecgc_postprocess.sh"
    )
    command = continuation_command(descriptor)
    assert Path(command[3]).name == "run_t14_route_c_continuation.py"
    source = (
        REPO_ROOT / "src/baselines/tastemolnet_t14_route_c_continuation.py"
    ).read_text(encoding="utf-8")
    assert "launch_fast16_matrix_publisher_queue.sh" not in source
    assert "append_tastemolnet_matrix_authority.py" not in source
    assert CELL_ID in source


def test_restarted_continuation_waits_for_inherited_postprocess_writer_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "continuation"
    root.mkdir()
    final = tmp_path / "final"
    lock_path = root / "postprocess_writer.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=(descriptor,),
    )
    # Model the old continuation dying: only its live child retains the lock.
    os.close(descriptor)
    waits: list[str] = []

    def finish_prior_writer(_seconds: float) -> None:
        waits.append("prior_writer_observed")
        final.mkdir()
        child.terminate()
        child.wait(timeout=5)

    try:
        continuation._run_postprocess(
            {
                "generation_root": str(tmp_path / "generation"),
                "postprocess_science_root": str(tmp_path / "science"),
                "postprocess_final_root": str(final),
                "poll_seconds": 5,
            },
            root=root,
            sleep=finish_prior_writer,
        )
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
    assert waits == ["prior_writer_observed"]


def test_continuation_launcher_adopts_live_and_relaunches_dead_same_roots(
    tmp_path: Path,
) -> None:
    route, _route_path, spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    verification = _generation_terminal(Path(route["output_root"]))
    publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=verification,
        owner_pid=123,
        owner_start_ticks=456,
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    calls: list[list[str]] = []
    next_pid = iter((501, 502))

    class FakeChild:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def terminate(self) -> None:  # pragma: no cover - failure cleanup only
            raise AssertionError("unexpected child termination")

    def fake_popen(command: list[str], **_kwargs: object) -> FakeChild:
        pid = next(next_pid)
        calls.append(list(command))
        _fake_proc_process(proc, pid, list(command))
        return FakeChild(pid)

    first = launch_continuation_owner(
        descriptor, popen=fake_popen, proc_root=proc
    )
    assert first["launch_status"] == "STARTED"
    assert first["launch_ordinal"] == 1
    adopted = launch_continuation_owner(
        descriptor, popen=fake_popen, proc_root=proc
    )
    assert adopted["launch_status"] == "ADOPTED_LIVE_OWNER"
    assert len(calls) == 1

    dead = proc / str(first["pid"])
    for child in dead.iterdir():
        child.unlink()
    dead.rmdir()
    relaunched = launch_continuation_owner(
        descriptor, popen=fake_popen, proc_root=proc
    )
    assert relaunched["launch_status"] == "STARTED"
    assert relaunched["launch_ordinal"] == 2
    assert relaunched["relaunch_same_descriptor"] is True
    assert len(calls) == 2
    assert relaunched["postprocess_science_root"] == spec["postprocess_science_root"]
    assert relaunched["postprocess_final_root"] == spec["postprocess_final_root"]
    assert relaunched["matrix_locator"] == spec["matrix"]["locator_path"]


def test_continuation_launch_refuses_applied_matrix_before_creating_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route, _route_path, spec, descriptor, _calibration, _test = (
        _route_and_continuation(tmp_path)
    )
    publish_generation_handoff(
        continuation_spec_path=descriptor,
        generation_verification=_generation_terminal(Path(route["output_root"])),
        owner_pid=123,
        owner_start_ticks=456,
    )
    root = Path(spec["continuation_root"])
    assert not root.exists()

    def already_applied() -> dict[str, Any]:
        raise T14RouteCContinuationError(
            "fast16 authority already contains TasteMolNet/ComRecGC"
        )

    monkeypatch.setattr(
        continuation, "require_unpublished_matrix_cell", already_applied
    )
    with pytest.raises(T14RouteCContinuationError, match="already contains"):
        launch_continuation_owner(descriptor, proc_root=tmp_path / "proc")
    assert not root.exists()
