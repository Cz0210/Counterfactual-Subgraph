from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import pytest

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    _controller_root,
    _source_controller_root_from_manifest,
    keep_alive_after_all_terminal,
    load_controller_manifest,
)
from src.utils.autodl_bace_continuation import (
    BaceContinuationError,
    assert_continuation_predecessor_quiescent,
    build_bace_continuation_payload,
)
from src.utils.autodl_runtime import build_runtime_layout, sha256_file, sha256_paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _namespaced_source_fixture(tmp_path: Path) -> tuple[Any, Any, Path]:
    project_root = tmp_path / "project"
    data_root = tmp_path / "data"
    project_root.mkdir()
    data_root.mkdir()
    control_root = data_root / "counterfactual-subgraph-runtime" / "control"
    layout = build_runtime_layout(
        project_root=project_root,
        data_root=data_root,
        control_root=control_root,
    ).ensure()
    payload = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/autodl/four_gpu_recovery.live_candidate.json"
        ).read_text(encoding="utf-8")
    )
    source_id = payload["controller_id"]
    namespace_root = control_root / "four_gpu_recovery"
    source_manifest = namespace_root / "manifests" / f"{source_id}-0ad1494.json"
    _write_json(source_manifest, payload)
    source = load_controller_manifest(source_manifest)
    source_root = namespace_root / source_id
    source_root.mkdir(parents=True)
    snapshot = dict(payload)
    snapshot["source_manifest"] = str(source.path)
    snapshot["source_manifest_sha256"] = source.sha256
    _write_json(source_root / "controller_manifest.json", snapshot)
    return layout, source, source_root


def test_four_by_four_alias_resolves_source_from_old_persistent_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, source, source_root = _namespaced_source_fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.autodl.run_four_gpu_recovery_controller.CONTROLLER_NAME",
        "four_methods_four_datasets_continuation",
    )

    assert _source_controller_root_from_manifest(layout, source) == source_root
    assert _controller_root(layout, "four_methods_four_datasets_continuation_v1") == (
        layout.control_root
        / "four_methods_four_datasets_continuation"
        / "four_methods_four_datasets_continuation_v1"
    )


def test_source_controller_namespace_resolution_fails_closed(
    tmp_path: Path,
) -> None:
    layout, source, source_root = _namespaced_source_fixture(tmp_path)
    snapshot_path = source_root / "controller_manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["controller_id"] = "different-controller"
    _write_json(snapshot_path, snapshot)
    with pytest.raises(ControllerError, match="controller_id mismatch"):
        _source_controller_root_from_manifest(layout, source)

    outside = tmp_path / "outside.json"
    _write_json(outside, json.loads(source.path.read_text(encoding="utf-8")))
    outside_source = load_controller_manifest(outside)
    with pytest.raises(ControllerError, match="escapes persistent control root"):
        _source_controller_root_from_manifest(layout, outside_source)


def _materialize_required_outputs(output: Path, task: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for relative in task.get("required_output_files", []):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PASS\n", encoding="utf-8")
    for alternatives in task.get("required_output_any", []):
        path = output / alternatives[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact\n", encoding="utf-8")


def _run_spec_for_source_instance(
    root: Path,
    *,
    task: dict[str, Any],
    instance_id: str,
    run_id: str,
    output: Path,
    gpu_index: int | None,
) -> dict[str, Any]:
    input_manifest = root / "inputs" / f"{run_id}.json"
    config = root / "configs" / f"{run_id}.json"
    log = root / "logs" / f"{run_id}.log"
    _write_json(input_manifest, {"status": "FROZEN", "run_id": run_id})
    _write_json(config, {"status": "FROZEN", "run_id": run_id})
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(str(task["required_log_marker"]) + "\n", encoding="utf-8")
    runner_stage = str(task.get("runner_stage", task["stage"])).replace(
        "{shard_id}", instance_id
    )
    spec = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": "2026-08-22T00:00:00+00:00",
        "project_root": str(root / "immutable-project"),
        "data_root": str(root / "data"),
        "control_root": str(root / "control"),
        "python_executable": str(Path(sys.executable).resolve()),
        "dataset": task.get("runner_dataset", task["dataset"]),
        "stage": runner_stage,
        "command": ["bash", "/opt/counterfactual-subgraph/run-stage.sh", run_id],
        "environment": {
            "OMP_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "3",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "gpu_index": gpu_index,
        "gpu_uuid": f"GPU-fixture-{gpu_index}" if gpu_index is not None else None,
        "min_free_memory_mb": 16000,
        "idle_util_threshold": 10,
        "max_gpus": 4,
        "gpu_hard_limit": 4,
        "git_commit": "a" * 40,
        "config_files": [str(config)],
        "config_hash": sha256_paths([config]),
        "input_manifest": str(input_manifest),
        "input_hash": sha256_file(input_manifest),
        "expected_output": str(output),
        "required_output_files": list(task.get("required_output_files", [])),
        "required_output_any": list(task.get("required_output_any", [])),
        "required_absolute_output_files": [],
        "required_log_marker": task["required_log_marker"],
        "log_path": str(log),
        "launcher": "nohup",
        "tmux_session": None,
        "heavy": gpu_index is not None,
    }
    return spec


def _fixture() -> tuple[Path, dict[str, Any]]:
    root = Path(tempfile.mkdtemp(prefix="bace-continuation-", dir="/private/tmp"))
    (root / "immutable-project").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "control").mkdir()
    runs_root = root / "runs"
    source_controller_root = root / "source-controller"
    (source_controller_root / "controller.lock").parent.mkdir(parents=True)
    (source_controller_root / "controller.lock").write_bytes(b"")

    repository_root = Path(__file__).resolve().parents[2]
    source_payload = json.loads(
        (repository_root / "configs/autodl/four_gpu_recovery.live_candidate.json")
        .read_text(encoding="utf-8")
    )
    source_manifest = root / "source-manifest.json"
    _write_json(source_manifest, source_payload)
    tasks = {task["id"]: task for task in source_payload["tasks"]}
    main_tasks = (
        "bace_b6_ppo_smoke",
        "bace_b7_ppo_full",
        "bace_b7_prep_gnn_before",
        "bace_b7_prep_shard_manifests",
        "bace_b7_prep_output_preflight",
        "bace_b10_pool_merged",
    )
    for task_id in (*main_tasks, "bace_b8_pool_base", "bace_b9_pool_hightemp"):
        task = tasks[task_id]
        instance_ids = (
            [f"shard-{index:03d}" for index in range(4)]
            if task_id in {"bace_b8_pool_base", "bace_b9_pool_hightemp"}
            else ["main"]
        )
        instances: dict[str, Any] = {}
        for index, instance_id in enumerate(instance_ids):
            run_id = f"fixture-{task_id}-{instance_id}"
            output = root / "source-outputs" / task_id / instance_id
            _materialize_required_outputs(output, task)
            gpu_index = index if task["resource"] == "gpu" else None
            spec = _run_spec_for_source_instance(
                root,
                task=task,
                instance_id=instance_id,
                run_id=run_id,
                output=output,
                gpu_index=gpu_index,
            )
            _write_json(runs_root / run_id / "launch_spec.json", spec)
            _write_json(
                runs_root / run_id / "state.json",
                {
                    "run_id": run_id,
                    "dataset": spec["dataset"],
                    "stage": spec["stage"],
                    "state": "PASS",
                    "gpu_index": spec["gpu_index"],
                    "gpu_uuid": spec["gpu_uuid"],
                    "log_path": spec["log_path"],
                },
            )
            instances[instance_id] = {
                "instance_id": instance_id,
                "state": "PASS",
                "run_id": run_id,
                "expected_output": str(output),
            }
        _write_json(
            source_controller_root / "tasks" / task_id / "state.json",
            {"task_id": task_id, "state": "PASS", "instances": instances},
        )
        _write_json(
            source_controller_root / "tasks" / task_id / "gate.json",
            {"task_id": task_id, "status": "PASS"},
        )

    repair_run_id = "fixture-bace-molclr-repair"
    repair_output = root / "repair-output"
    repair_output.mkdir()
    node_cache = root / "repair-node-cache"
    node_cache.mkdir()
    (node_cache / "one.npz").write_bytes(b"npz")
    _write_json(
        repair_output / "prep_manifest.json",
        {
            "status": "PASS",
            "dataset": "bace",
            "action": "CALIBRATION_MOLCLR_PARENT_CACHE",
            "rf_oracle_used": False,
            "calibration_loaded": True,
            "test_loaded": False,
            "policy_checkpoint_loaded": False,
            "candidate_generation_performed": False,
            "selector_fitted": False,
            "parent_count": 1,
        },
    )
    (repair_output / "calibration_parent_molclr_cache.jsonl").write_text(
        json.dumps(
            {
                "parent_id": "parent-1",
                "canonical_smiles": "CC",
                "cache_path": str(node_cache / "one.npz"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (repair_output / "PASS").write_text("PASS\n", encoding="utf-8")
    repair_task = {
        "id": "repair",
        "dataset": "bace-evidence",
        "stage": "B7_PREP_MOLCLR_PARENT",
        "runner_dataset": "bace-gnn-clean-prep",
        "runner_stage": "B7_PREP_MOLCLR_PARENT",
        "resource": "gpu",
        "required_output_files": [
            "calibration_parent_molclr_cache.jsonl",
            "prep_manifest.json",
            "PASS",
        ],
        "required_log_marker": "[BACE_B7_PARALLEL_PREP_PASS]",
    }
    repair_spec = _run_spec_for_source_instance(
        root,
        task=repair_task,
        instance_id="main",
        run_id=repair_run_id,
        output=repair_output,
        gpu_index=3,
    )
    repair_spec["command"] = [
        "bash",
        "/opt/counterfactual-subgraph/run-stage.sh",
        "--node-embedding-cache-dir",
        str(node_cache),
    ]
    _write_json(runs_root / repair_run_id / "launch_spec.json", repair_spec)
    _write_json(
        runs_root / repair_run_id / "state.json",
        {
            "run_id": repair_run_id,
            "dataset": repair_spec["dataset"],
            "stage": repair_spec["stage"],
            "state": "PASS",
            "gpu_index": 3,
            "gpu_uuid": "GPU-fixture-3",
            "log_path": repair_spec["log_path"],
        },
    )
    return root, {
        "source_manifest": source_manifest,
        "source_controller_root": source_controller_root,
        "runs_root": runs_root,
        "repair_run_id": repair_run_id,
        "node_cache": node_cache,
    }


def test_builder_flattens_all_pool_shards_and_releases_fresh_b11_b14() -> None:
    _root, fixture = _fixture()
    output_root = Path(
        "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/"
        "four_gpu_recovery/continuation-fixture/bace/frozen_gnn_downstream"
    )
    wnode = Path(
        "/autodl-fs/data/counterfactual-subgraph-runtime/cache/bace/"
        "frozen_gnn_downstream/continuation-fixture/wnode/wnode_cache.sqlite3"
    )
    payload = build_bace_continuation_payload(
        source_manifest=fixture["source_manifest"],
        source_controller_root=fixture["source_controller_root"],
        runs_root=fixture["runs_root"],
        molclr_repair_run_id=fixture["repair_run_id"],
        controller_id="continuation-fixture",
        output_root=output_root,
        wnode_cache_db=wnode,
    )
    tasks = {task["id"]: task for task in payload["tasks"]}
    flattened = [
        task
        for task_id, task in tasks.items()
        if task_id.startswith(("bace_b8_pool_base_shard_", "bace_b9_pool_hightemp_shard_"))
    ]
    assert len(flattened) == 8
    assert all("shards" not in task for task in flattened)
    assert all(task["adopt_existing_run_id"] for task in flattened)
    assert len(payload["continuation"]["adopted_run_ids"]) == 15
    assert tasks["bace_b10_pool_merged"]["depends_on"] == [
        *(f"bace_b8_pool_base_shard_{index:03d}" for index in range(4)),
        *(f"bace_b9_pool_hightemp_shard_{index:03d}" for index in range(4)),
    ]
    b11 = tasks["bace_b11_verification_shards"]
    assert "bace_b7_prep_molclr_parent" in b11["depends_on"]
    assert "bace_continuation_output_preflight" in b11["depends_on"]
    assert str(fixture["node_cache"]) in b11["command"]
    assert str(wnode) in b11["command"]
    assert b11["expected_output"].startswith(str(output_root))
    assert tasks["bace_b7_prep_molclr_parent"]["stage"] == (
        "B7_PREP_MOLCLR_PARENT_REPAIRED"
    )

    manifest_path = fixture["source_manifest"].parent / "continuation.json"
    _write_json(manifest_path, payload)
    manifest = load_controller_manifest(manifest_path)
    assert manifest.controller_id == "continuation-fixture"
    assert manifest.runtime["keep_alive_when_blocked"] is True
    assert manifest.by_id["bace_b11_verification_shards"].shards is not None
    assert manifest.by_id["bace_b14_frozen"].manifest_only is True


def test_continuation_guard_rejects_live_source_controller() -> None:
    _root, fixture = _fixture()
    payload = build_bace_continuation_payload(
        source_manifest=fixture["source_manifest"],
        source_controller_root=fixture["source_controller_root"],
        runs_root=fixture["runs_root"],
        molclr_repair_run_id=fixture["repair_run_id"],
        controller_id="continuation-lock-fixture",
        output_root=Path("/autodl-fs/data/continuation-lock-output"),
        wnode_cache_db=Path("/autodl-fs/data/continuation-lock-cache.sqlite3"),
    )
    lock_path = fixture["source_controller_root"] / "controller.lock"
    with lock_path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BaceContinuationError, match="still owns"):
            assert_continuation_predecessor_quiescent(
                payload["continuation"], require_fresh_targets=True
            )


def test_controller_manifest_allows_only_exact_tokenizer_scheduling_key(
    tmp_path: Path,
) -> None:
    template = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/autodl/four_gpu_recovery.template.json"
        ).read_text(encoding="utf-8")
    )
    template["tasks"][0]["environment"]["TOKENIZERS_PARALLELISM"] = "false"
    allowed = tmp_path / "allowed.json"
    _write_json(allowed, template)
    load_controller_manifest(allowed)

    template["tasks"][0]["environment"]["API_TOKEN"] = "forbidden"
    forbidden = tmp_path / "forbidden.json"
    _write_json(forbidden, template)
    with pytest.raises(Exception, match="credential-like environment key"):
        load_controller_manifest(forbidden)


def test_keep_alive_when_blocked_is_strict_boolean_and_needs_no_dummy_task(
    tmp_path: Path,
) -> None:
    template = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/autodl/four_gpu_recovery.template.json"
        ).read_text(encoding="utf-8")
    )
    template["runtime"]["keep_alive_when_blocked"] = "true"
    invalid = tmp_path / "invalid-keep-alive.json"
    _write_json(invalid, template)
    with pytest.raises(Exception, match="keep_alive_when_blocked must be boolean"):
        load_controller_manifest(invalid)

    states = [
        {
            "state": "BLOCKED",
            "instances": {"main": {"state": "BLOCKED"}},
        },
        {"state": "PASS", "instances": {"main": {"state": "PASS"}}},
    ]
    assert keep_alive_after_all_terminal(
        {"keep_alive_when_blocked": True}, states
    )
    assert not keep_alive_after_all_terminal({}, states)
    states[0]["instances"]["main"]["state"] = "RUNNING"
    assert not keep_alive_after_all_terminal(
        {"keep_alive_when_blocked": True}, states
    )
