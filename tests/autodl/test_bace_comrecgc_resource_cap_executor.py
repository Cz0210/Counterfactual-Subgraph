from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256
from src.utils import autodl_bace_comrecgc_resource_cap_executor as executor


def _observed(second: float, *, cpu: int = 10, output: int = 100) -> dict[str, object]:
    return {
        "observed_monotonic": second,
        "pid": 123,
        "start_ticks": 456,
        "cpu_ticks": cpu,
        "progress_step": 17_975,
        "checkpoint_step": 17_500,
        "output_bytes": output,
        "checkpoint_write_in_progress": False,
    }


def test_liveness_cpu_growth_is_running_slow_not_stalled() -> None:
    first = _observed(0.0, cpu=100)
    second = _observed(30.0, cpu=3_100)
    result = executor.classify_liveness([first, second])
    assert result["state"] == "RUNNING_SLOW"
    assert result["deltas"]["cpu_ticks"] == 3_000
    assert result["signal_allowed"] is False


def test_liveness_requires_full_hour_and_no_checkpoint_write_for_stall() -> None:
    short = executor.classify_liveness([_observed(0), _observed(3599)])
    assert short["state"] == "STALL_NOT_PROVEN"
    assert short["signal_allowed"] is False
    last = _observed(3600)
    last["checkpoint_write_in_progress"] = True
    writing = executor.classify_liveness([_observed(0), last])
    assert writing["state"] == "RUNNING_SLOW"
    stalled = executor.classify_liveness([_observed(0), _observed(3600)])
    assert stalled["state"] == "STALLED"
    assert stalled["signal_allowed"] is True


def _proc_contract(tmp_path: Path) -> tuple[executor.ProcessContract, Path]:
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    output = tmp_path / "generation"
    output.mkdir()
    cwd = output / "official_runtime"
    cwd.mkdir()
    argv = [b"python", b"run_generation.py", b"--output-dir", os.fsencode(output)]
    cmdline = b"\0".join(argv) + b"\0"
    (process / "cmdline").write_bytes(cmdline)
    # fields after the closing comm: state, ppid, then enough values for
    # utime[11], stime[12], and start_ticks[19].
    fields = ["S", "77", *(["0"] * 17), "456", "0", "0"]
    (process / "stat").write_text(
        "123 (science worker) " + " ".join(fields) + "\n", encoding="utf-8"
    )
    (process / "cwd").symlink_to(cwd, target_is_directory=True)
    controller = tmp_path / "controller.json"
    controller.write_text(
        json.dumps(
            {
                "controller_id": "controller-1",
                "science": {"pid": 123, "start_ticks": 456},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        executor.ProcessContract(
            pid=123,
            start_ticks=456,
            cmdline_sha256=hashlib.sha256(cmdline).hexdigest(),
            cwd=str(cwd),
            output_root=str(output),
            controller_id="controller-1",
            controller_receipt=str(controller),
            controller_receipt_sha256=sha256_file(controller),
            expected_ppid=77,
        ),
        proc,
    )


def test_precise_process_verifier_binds_generation_command_cwd_root_and_owner(
    tmp_path: Path,
) -> None:
    contract, proc = _proc_contract(tmp_path)
    result = executor.verify_exact_process(contract, proc_root=proc)
    assert result["pid"] == 123
    assert result["start_ticks"] == 456
    assert result["cwd"] == contract.cwd
    assert result["output_root"] == contract.output_root
    assert result["controller_id"] == "controller-1"


def test_precise_process_verifier_rejects_pid_reuse(tmp_path: Path) -> None:
    contract, proc = _proc_contract(tmp_path)
    changed = replace(contract, start_ticks=457)
    with pytest.raises(executor.BaceComRecGCResourceCapExecutorError, match="start ticks"):
        executor.verify_exact_process(changed, proc_root=proc)


def _eligible_request(step: int = 20_000) -> dict[str, object]:
    return {
        "status": "HANDOVER_ELIGIBLE",
        "reason": "RESOURCE_CAP_20000",
        "m_effective": step,
        "valid_unique_count": 10,
        "lineage_error_count": 0,
        "checkpoint_digest": "a" * 64,
    }


def test_materializer_replays_exact_checkpoint_without_another_walk_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "step-000000020000"
    checkpoint.mkdir()
    sqlite = checkpoint / "authoritative_graph_store.sqlite3"
    sqlite.write_bytes(b"sqlite")
    source_trace = tmp_path / "source-trace"
    chunks = source_trace / "selected_action_trace_chunks"
    chunks.mkdir(parents=True)
    chunk = chunks / "part-000000.jsonl"
    chunk.write_text('{"event":"teleport"}\n', encoding="utf-8")
    trace_state = {
        "schema_version": "comrecgc_action_trace_checkpoint_v1",
        "chunk_size": 512,
        "compact_enumeration": True,
        "chunks": [
            {
                "index": 0,
                "path": "selected_action_trace_chunks/part-000000.jsonl",
                "row_count": 1,
                "bytes": chunk.stat().st_size,
                "sha256": sha256_file(chunk),
            }
        ],
        "pending_events": [],
        "move_index": 20_000,
    }
    provenance = {
        "generation_parent_ids_sha256": stable_json_sha256(
            [f"p-{index}" for index in range(360)]
        )
    }
    official = {
        "graph_map": {"g": [object(), 0]},
        "graph_index_map": {"g": 0},
        "counterfactual_candidates": [{"graph_hash": "g"}],
        "MAX_COUNTERFACTUAL_SIZE": 50_000,
        "traversed_hashes": [None] * 20_000,
        "input_graphs_covered": [1.0],
    }
    validation = SimpleNamespace(
        checkpoint_dir=checkpoint,
        checkpoint_digest="a" * 64,
        completed_step=20_000,
        total_steps=50_000,
        provenance_fingerprints=provenance,
    )
    loaded = SimpleNamespace(
        validation=validation,
        completed_step=20_000,
        algorithm_state={"official_state": official},
        trace_state=trace_state,
        sqlite_snapshot_path=sqlite,
    )
    monkeypatch.setattr(executor, "load_generation_checkpoint", lambda *a, **k: loaded)
    parents = [f"p-{index}" for index in range(360)]
    monkeypatch.setattr(
        executor,
        "load_bace_generation_bundle",
        lambda **_kwargs: SimpleNamespace(parent_ids=parents, graphs=[object()] * 360),
    )

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def restore_checkpoint_state(self, value: object) -> None:
            assert value is trace_state

        def write(self, _root: Path, _payload: object, **kwargs: object) -> dict[str, object]:
            audit = Path(str(kwargs["frozen_payload_audit_path"]))
            audit.write_text('{"closure_complete":true}\n', encoding="utf-8")
            return {"frozen_payload_closure": {"closure_complete": True}}

    monkeypatch.setattr(executor, "ActionTraceRecorder", Recorder)
    monkeypatch.setattr(
        executor,
        "_torch_save_atomic",
        lambda _payload, path: path.write_bytes(b"checkpoint-materialized"),
    )
    monkeypatch.setattr(
        executor,
        "payload_file_audit",
        lambda path: {
            "payload_path": str(path),
            "payload_bytes": path.stat().st_size,
            "payload_checksum": sha256_file(path),
        },
    )
    config = {
        "dataset": "bace",
        "mode": "full",
        "parent_limit": 360,
        "total_steps": 50_000,
        "checkpoint_provenance": provenance,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "eligible_for_bace_gnn_main_results": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    config["config_sha256"] = stable_json_sha256(config)
    config_path = tmp_path / "resolved_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "fresh-generation"
    manifest = executor.materialize_resource_cap_checkpoint(
        checkpoint_dir=checkpoint,
        expected_checkpoint_digest="a" * 64,
        source_trace_root=source_trace,
        source_resolved_config=config_path,
        dataset_dir=dataset,
        output_dir=output,
        resource_cap_receipt=_eligible_request(),
    )
    assert manifest["algorithm_rerun"] is False
    assert manifest["M_configured_max"] == 20_000
    assert manifest["M_effective"] == 20_000
    assert manifest["resource_cap_used"] is True
    assert manifest["test_loaded"] is False
    assert (output / "_RUN_COMPLETE.json").is_file()


def test_executor_waits_without_request_and_never_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cap"
    contract = executor.ProcessContract(
        pid=123,
        start_ticks=456,
        cmdline_sha256="a" * 64,
        cwd=str(tmp_path),
        output_root=str(tmp_path),
        controller_id="owner",
        controller_receipt=str(tmp_path / "missing-owner"),
        controller_receipt_sha256="b" * 64,
    )
    inputs = executor.ResourceCapExecutorInputs(
        handover_request=str(tmp_path / "missing-request.json"),
        checkpoint_dir=str(tmp_path / "future-checkpoint"),
        source_trace_root=str(tmp_path / "future-trace"),
        source_resolved_config=str(tmp_path / "future-config"),
        dataset_dir=str(tmp_path / "future-dataset"),
        output_root=str(output),
        python="/python",
        project_root="/project",
        gnn_checkpoint="/gnn",
        calibration_split="/cal",
        test_split="/test",
        molclr_root="/molclr",
        molclr_checkpoint="/molclr/model",
        neurosed_checkpoint="/neurosed",
        official_root="/official",
        process=contract,
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, int(sig))))
    state = executor.ResourceCapExecutor(inputs).tick()
    assert state["state"] == "WAITING_RESOURCE_CAP_REQUEST"
    assert killed == []


def test_cap_executor_uses_only_exact_sigterm_and_persists_postprocess_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_eligible_request()), encoding="utf-8")
    checkpoint_dir = tmp_path / "step-000000020000"
    checkpoint_dir.mkdir()
    generation_state = checkpoint_dir / "generation_state.pt"
    generation_state.write_bytes(b"committed-20k-rng-and-science-state")
    generation_state_sha = sha256_file(generation_state)
    checkpoint_manifest = {
        "files": {"generation_state.pt": {"sha256": generation_state_sha}}
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    checkpoint = SimpleNamespace(
        validation=SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            checkpoint_digest="a" * 64,
            manifest=checkpoint_manifest,
        ),
        completed_step=20_000,
    )
    monkeypatch.setattr(
        executor,
        "load_generation_checkpoint",
        lambda *_a, **_k: checkpoint,
    )
    monkeypatch.setattr(
        executor,
        "verify_exact_process",
        lambda contract: {"pid": contract.pid, "start_ticks": contract.start_ticks},
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, int(sig))))
    monkeypatch.setattr(
        executor,
        "_parse_proc_stat",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
    )

    def materialize(**kwargs: object) -> dict[str, object]:
        root = Path(str(kwargs["output_dir"]))
        root.mkdir(parents=True)
        (root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
        candidate_path = root / "counterfactuals.pt"
        candidate_path.write_bytes(b"candidate-universe")
        candidate_sha = sha256_file(candidate_path)
        return {
            "M_configured_max": 20_000,
            "M_effective": 20_000,
            "stop_reason": "RESOURCE_CAP_20000",
            "resource_cap_used": True,
            "early_stop_used": False,
            "algorithm_rerun": False,
            "counterfactuals_path": str(candidate_path),
            "counterfactuals_sha256": candidate_sha,
        }

    monkeypatch.setattr(executor, "materialize_resource_cap_checkpoint", materialize)
    monkeypatch.setattr(
        executor,
        "build_postprocess_fragment",
        lambda **_kwargs: {"tasks": []},
    )
    output = tmp_path / "cap"
    contract = executor.ProcessContract(
        pid=123,
        start_ticks=456,
        cmdline_sha256="a" * 64,
        cwd=str(tmp_path),
        output_root=str(tmp_path),
        controller_id="owner",
        controller_receipt=str(tmp_path / "owner.json"),
        controller_receipt_sha256="b" * 64,
    )
    values = {field: str(tmp_path / field) for field in (
        "checkpoint_dir",
        "source_trace_root",
        "source_resolved_config",
        "dataset_dir",
        "project_root",
        "gnn_checkpoint",
        "calibration_split",
        "test_split",
        "molclr_root",
        "molclr_checkpoint",
        "neurosed_checkpoint",
        "official_root",
    )}
    inputs = executor.ResourceCapExecutorInputs(
        handover_request=str(request),
        output_root=str(output),
        python="/python",
        process=contract,
        exit_wait_seconds=1,
        **values,
    )
    state = executor.ResourceCapExecutor(inputs).tick()
    assert killed == [(123, 15)]
    assert state["state"] == "POSTPROCESS_QUEUE_READY"
    assert state["signals_sent"] == ["SIGTERM"]
    assert state["sigkill_used"] is False
    assert (output / "executor/resource_cap_receipt.json").is_file()
    formal = json.loads(
        (output / "executor/bace_comrecgc_20k_resource_cap_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = json.loads(
        (output / "executor/excluded_after_20k.json").read_text(encoding="utf-8")
    )
    assert formal["schema_version"] == "bace_comrecgc_20k_resource_cap_receipt_v1"
    assert excluded["schema_version"] == "bace_comrecgc_excluded_after_20k_v1"
    assert formal["RNG_state_SHA"] == generation_state_sha
    assert formal["candidate_universe_SHA"] == sha256_file(
        output / "train_generation/counterfactuals.pt"
    )
    assert formal["post_20k_uncommitted_outputs_excluded"] is True
    assert formal["scientific_result_adopted_through_step"] == 20_000
    assert formal["handover_graceful_exit"] is True
    assert formal["handover_exact_pid"] == 123
    assert formal["handover_exact_start_ticks"] == 456
    assert formal["sigkill_used"] is False
    assert excluded["later_partial_rows_adopted"] is False
    assert excluded["later_temporary_outputs_deleted"] is False
    assert sha256_file(
        output / "executor/bace_comrecgc_20k_resource_cap_receipt.json"
    ) == state["formal_resource_cap_receipt_sha256"]
    assert sha256_file(output / "executor/excluded_after_20k.json") == state[
        "excluded_after_20k_sha256"
    ]
    assert (output / "executor/postprocess.tasks.json").is_file()


def _post20k_receipt_kwargs(tmp_path: Path) -> dict[str, object]:
    checkpoint_dir = tmp_path / "step-000000020000"
    checkpoint_dir.mkdir()
    generation_state = checkpoint_dir / "generation_state.pt"
    generation_state.write_bytes(b"committed-state")
    checkpoint_manifest = {
        "files": {
            "generation_state.pt": {"sha256": sha256_file(generation_state)}
        }
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    source = tmp_path / "source-generation"
    source.mkdir()
    candidate_root = tmp_path / "formal-generation"
    candidate_root.mkdir()
    candidate = candidate_root / "counterfactuals.pt"
    candidate.write_bytes(b"formal-20k-candidates")
    state_root = tmp_path / "state"
    state_root.mkdir()
    resource_cap_receipt = {
        **_eligible_request(),
        "M_configured_max": 20_000,
        "M_effective": 20_000,
        "stop_reason": "RESOURCE_CAP_20000",
        "process_before_signal": {"pid": 123, "start_ticks": 456},
    }
    (state_root / "resource_cap_receipt.json").write_text(
        json.dumps(resource_cap_receipt), encoding="utf-8"
    )
    (state_root / "signal_receipt.json").write_text(
        json.dumps(
            {
                "signal": "SIGTERM",
                "signal_number": 15,
                "exact_pid": 123,
                "exact_start_ticks": 456,
                "exited_within_wait": True,
                "sigkill_used": False,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = SimpleNamespace(
        completed_step=20_000,
        validation=SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            checkpoint_digest="a" * 64,
            manifest=checkpoint_manifest,
        ),
    )
    return {
        "state_root": state_root,
        "source_generation_root": source,
        "checkpoint": checkpoint,
        "materialized_manifest": {
            "M_configured_max": 20_000,
            "M_effective": 20_000,
            "stop_reason": "RESOURCE_CAP_20000",
            "resource_cap_used": True,
            "early_stop_used": False,
            "algorithm_rerun": False,
            "counterfactuals_path": str(candidate),
            "counterfactuals_sha256": sha256_file(candidate),
        },
        "resource_cap_receipt": resource_cap_receipt,
    }


def test_post20k_formal_receipt_reopens_candidate_and_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    kwargs = _post20k_receipt_kwargs(tmp_path)
    manifest = dict(kwargs["materialized_manifest"])
    manifest["counterfactuals_sha256"] = "d" * 64
    kwargs["materialized_manifest"] = manifest

    with pytest.raises(
        executor.BaceComRecGCResourceCapExecutorError,
        match="candidate-universe SHA256 changed",
    ):
        executor._write_post20k_exclusion_receipts(**kwargs)
    state_root = Path(str(kwargs["state_root"]))
    assert not (state_root / "bace_comrecgc_20k_resource_cap_receipt.json").exists()
    assert not (state_root / "excluded_after_20k.json").exists()


def test_post20k_formal_receipt_rejects_non_20k_policy(
    tmp_path: Path,
) -> None:
    kwargs = _post20k_receipt_kwargs(tmp_path)
    receipt = dict(kwargs["resource_cap_receipt"])
    receipt["M_effective"] = 17_500
    receipt["stop_reason"] = "PREREGISTERED_CONVERGENCE_PASS"
    kwargs["resource_cap_receipt"] = receipt

    with pytest.raises(
        executor.BaceComRecGCResourceCapExecutorError,
        match="exact clean committed-20k handover",
    ):
        executor._write_post20k_exclusion_receipts(**kwargs)


def test_post20k_formal_receipt_rejects_signal_identity_drift(
    tmp_path: Path,
) -> None:
    kwargs = _post20k_receipt_kwargs(tmp_path)
    state_root = Path(str(kwargs["state_root"]))
    signal_receipt = json.loads(
        (state_root / "signal_receipt.json").read_text(encoding="utf-8")
    )
    signal_receipt["exact_start_ticks"] = 457
    (state_root / "signal_receipt.json").write_text(
        json.dumps(signal_receipt), encoding="utf-8"
    )

    with pytest.raises(
        executor.BaceComRecGCResourceCapExecutorError,
        match="does not prove exact graceful exit",
    ):
        executor._write_post20k_exclusion_receipts(**kwargs)


def test_absolute_cap_failure_stops_exact_worker_without_materializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_payload = {
        **_eligible_request(step=25_000),
        "status": "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP",
        "reason": "ABSOLUTE_CAP_INSUFFICIENT_RULES_OR_LINEAGE_ERRORS",
        "valid_unique_count": 9,
    }
    request = tmp_path / "request.json"
    request.write_text(json.dumps(request_payload), encoding="utf-8")
    checkpoint = SimpleNamespace(
        validation=SimpleNamespace(
            checkpoint_dir=tmp_path / "step-000000025000",
            checkpoint_digest="a" * 64,
        ),
        completed_step=25_000,
    )
    monkeypatch.setattr(executor, "load_generation_checkpoint", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr(
        executor,
        "verify_exact_process",
        lambda contract: {"pid": contract.pid, "start_ticks": contract.start_ticks},
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, int(sig))))
    monkeypatch.setattr(
        executor,
        "_parse_proc_stat",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
    )
    materialized: list[object] = []
    monkeypatch.setattr(
        executor,
        "materialize_resource_cap_checkpoint",
        lambda **kwargs: materialized.append(kwargs),
    )
    output = tmp_path / "cap"
    contract = executor.ProcessContract(
        pid=123,
        start_ticks=456,
        cmdline_sha256="a" * 64,
        cwd=str(tmp_path),
        output_root=str(tmp_path),
        controller_id="owner",
        controller_receipt=str(tmp_path / "owner.json"),
        controller_receipt_sha256="b" * 64,
    )
    values = {
        field: str(tmp_path / field)
        for field in (
            "checkpoint_dir",
            "source_trace_root",
            "source_resolved_config",
            "dataset_dir",
            "project_root",
            "gnn_checkpoint",
            "calibration_split",
            "test_split",
            "molclr_root",
            "molclr_checkpoint",
            "neurosed_checkpoint",
            "official_root",
        )
    }
    inputs = executor.ResourceCapExecutorInputs(
        handover_request=str(request),
        output_root=str(output),
        python="/python",
        process=contract,
        exit_wait_seconds=1,
        **values,
    )
    state = executor.ResourceCapExecutor(inputs).tick()
    assert killed == [(123, 15)]
    assert materialized == []
    assert state["state"] == "SCIENTIFIC_FAILED_AT_ABSOLUTE_CAP"
    assert state["M_effective"] == 25_000
    assert state["postprocess_started"] is False
    assert not (output / "train_generation").exists()


def test_request_validation_rejects_false_failure_and_unregistered_reason() -> None:
    false_failure = {
        **_eligible_request(step=25_000),
        "status": "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP",
        "reason": "ABSOLUTE_CAP_INSUFFICIENT_RULES_OR_LINEAGE_ERRORS",
    }
    with pytest.raises(
        executor.BaceComRecGCResourceCapExecutorError,
        match="contradicts",
    ):
        executor._validate_request(
            false_failure,
            checkpoint_digest="a" * 64,
            completed_step=25_000,
        )
    early = {
        **_eligible_request(step=17_500),
        "reason": "RESOURCE_CAP_20000",
    }
    with pytest.raises(
        executor.BaceComRecGCResourceCapExecutorError,
        match="registered policy",
    ):
        executor._validate_request(
            early,
            checkpoint_digest="a" * 64,
            completed_step=17_500,
        )
