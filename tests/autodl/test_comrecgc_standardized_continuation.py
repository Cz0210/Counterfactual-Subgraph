from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from scripts.autodl import run_comrecgc_standardized_continuation as continuation


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _file(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fake_procfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(continuation, "_PROC_ROOT", proc)
    return proc


def _inputs(tmp_path: Path, dataset: str = "mutagenicity") -> continuation.ContinuationInputs:
    source = tmp_path / "source"
    source.mkdir()
    payload = _file(source / "counterfactuals.pt")
    payload_sha256 = continuation.sha256_file(payload)
    parent_limit = 1283 if dataset == "aids" else 1448
    _json(
        source / "run_manifest.json",
        {
            "dataset": dataset,
            "mode": "full",
            "parent_limit": parent_limit,
            "run_complete": True,
            "freeze_only_recovery": True,
            "algorithm_rerun": False,
            "upstream_commit": continuation.UPSTREAM_COMMIT,
            "generation_mode": "adopted_read_only_cache",
            "counterfactuals_path": str(payload),
            "counterfactuals_sha256": payload_sha256,
            "counterfactual_candidate_count": 100,
            "project_commit": "b" * 40,
        },
    )
    _json(
        source / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "freeze_only_recovery": True,
            "counterfactuals_sha256": payload_sha256,
        },
    )
    _json(
        source / "freeze_only_recovery.json",
        {
            "recovery_completed": True,
            "completed_steps": 50_000,
            "algorithm_rerun": False,
            "counterfactuals_sha256": payload_sha256,
        },
    )
    _json(
        source / "frozen_payload_closure_audit.json",
        {"closure_complete": True, "post_write_reload_verified": True},
    )
    _json(
        source / "adoption_manifest.json",
        {
            "generation_mode": "adopted_read_only_cache",
            "source_checksums": {"resolved_config.json": "c" * 64},
        },
    )
    _file(source / "trace/candidate_action_lineage.json")
    _file(source / "trace/trace_summary.json")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _file(dataset_dir / "dataset_summary.json", "{}\n")
    _file(
        dataset_dir
        / ("graphs.pt" if dataset == "aids" else "generation_source_graphs.pt")
    )
    molclr_root = tmp_path / "molclr"
    molclr_root.mkdir()
    return continuation.ContinuationInputs(
        dataset=dataset,
        source_generation_root=source,
        upstream_root=upstream,
        dataset_dir=dataset_dir,
        source_csv=_file(tmp_path / "source.csv") if dataset == "aids" else None,
        distance_checkpoint=_file(tmp_path / "distance.pt"),
        dataset_csv=_file(tmp_path / "test.csv"),
        teacher_path=_file(tmp_path / "teacher.pkl"),
        molclr_root=molclr_root,
        molclr_checkpoint=_file(molclr_root / "model.pth"),
        thresholds_path=_file(tmp_path / "thresholds.json", "{}"),
        output_root=tmp_path / "fresh-output",
        device="cuda:0",
        theta_star=0.05 if dataset == "aids" else None,
        cost_cap=0.0535 if dataset == "aids" else None,
    )


def test_external_common_recovery_bootstrap_is_idempotent_and_starts_no_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _inputs(tmp_path, dataset="aids")
    pair_owner = tmp_path / "pair-owner"
    pair_owner.mkdir()
    pair_manifest = _file(pair_owner / "run_manifest.json", "{}\n")
    close_manifest = _file(tmp_path / "close-view.json", "{}\n")
    values = replace(
        base,
        device="cpu",
        common_recourse_engine="external_memory_exact_v1",
        external_dbscan_shortcut_mode="all_core_one_component_adaptive_anchor_v1",
        external_pair_store_source_manifest=pair_manifest,
        external_pair_store_source_owner_root=pair_owner,
        external_close_pair_view_manifest=close_manifest,
        common_recourse_resume=True,
    )
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *args, **kwargs: {
            "root": str(values.upstream_root),
            "actual_commit": continuation.UPSTREAM_COMMIT,
            "passed": True,
        },
    )
    first = continuation.bootstrap_external_common_recovery_continuation(values)
    second = continuation.bootstrap_external_common_recovery_continuation(values)
    assert first == second
    assert first["common_recourse_started"] is False
    assert first["downstream_started"] is False
    assert not (values.output_root / "common_recourse/_RUN_COMPLETE.json").exists()
    assert not (values.output_root / "PASS").exists()
    for name in (
        "generation_adoption_manifest.json",
        "upstream_checkout_audit.json",
        "continuation_resume_contract.json",
        "exact_recovery_continuation_bootstrap.json",
    ):
        assert (values.output_root / name).is_file()


def test_stage_commands_bind_physical_python_across_launcher_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    physical_python = _file(tmp_path / "bin/python3.10", "python\n")
    launcher_python = physical_python.with_name("python")
    launcher_python.symlink_to(physical_python.name)
    kwargs = {
        "project_commit": "b" * 40,
        "candidate_count": 100,
        "teacher_sha256": "d" * 64,
    }

    monkeypatch.setattr(continuation.sys, "executable", str(launcher_python))
    through_symlink = continuation.build_stage_commands(inputs, **kwargs)
    monkeypatch.setattr(continuation.sys, "executable", str(physical_python))
    through_physical_path = continuation.build_stage_commands(inputs, **kwargs)

    assert through_symlink == through_physical_path
    for _, argv, _, _ in through_symlink:
        assert argv[0] == str(physical_python.resolve(strict=True))
    assert [
        continuation.stable_json_sha256(argv)
        for _, argv, _, _ in through_symlink
    ] == [
        continuation.stable_json_sha256(argv)
        for _, argv, _, _ in through_physical_path
    ]


def test_adopts_recovered_generation_hashes_large_payload_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    original = continuation.sha256_file
    hashed: list[Path] = []

    def recording_sha(path: str | Path) -> str:
        resolved = Path(path).resolve()
        hashed.append(resolved)
        return original(resolved)

    monkeypatch.setattr(continuation, "sha256_file", recording_sha)
    result = continuation.validate_adopted_generation(inputs)
    assert result["status"] == "PASS"
    assert result["generation_adopted"] is True
    assert result["generation_rerun"] is False
    expected = original(inputs.source_generation_root / "counterfactuals.pt")
    assert result["counterfactuals_sha256_claimed"] == expected
    assert result["counterfactuals_sha256_actual"] == expected
    assert result["counterfactuals_sha256_computation_count"] == 1
    assert hashed.count((inputs.source_generation_root / "counterfactuals.pt").resolve()) == 1
    assert hashed


def test_rejects_tampered_counterfactual_payload(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (inputs.source_generation_root / "counterfactuals.pt").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="COUNTERFACTUALS_PAYLOAD_SHA256_MISMATCH"):
        continuation.validate_adopted_generation(inputs)


def test_rejects_live_procfs_writer(
    tmp_path: Path, _fake_procfs: Path
) -> None:
    inputs = _inputs(tmp_path)
    pid = _fake_procfs / "424242"
    (pid / "fd").mkdir(parents=True)
    (pid / "fdinfo").mkdir()
    (pid / "fd" / "7").symlink_to(
        inputs.source_generation_root / "counterfactuals.pt"
    )
    (pid / "fdinfo" / "7").write_text("flags:\t0100001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LIVE_WRITER_DETECTED"):
        continuation.validate_adopted_generation(inputs)


def test_rejects_closure_manifest_change_after_entry_gate(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    adoption = continuation.validate_adopted_generation(inputs)
    closure = inputs.source_generation_root / "frozen_payload_closure_audit.json"
    payload = json.loads(closure.read_text(encoding="utf-8"))
    payload["tampered"] = True
    _json(closure, payload)
    with pytest.raises(ValueError, match="SOURCE_CLOSURE_CHANGED"):
        continuation._verify_adopted_generation_integrity(adoption)


def test_parent_metadata_never_enters_generation_content_identity(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest_path = inputs.source_generation_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["representative_parent_id"] = "DIFFERENT_PROVENANCE_PARENT"
    _json(manifest_path, manifest)
    result = continuation.validate_adopted_generation(inputs)
    assert result["status"] == "PASS"
    assert "representative_parent_id" not in result


def test_rejects_payload_outside_frozen_recovery_root(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    outside = _file(tmp_path / "outside.pt")
    manifest_path = inputs.source_generation_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counterfactuals_path"] = str(outside)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not_inside_frozen_source"):
        continuation.validate_adopted_generation(inputs)


def test_commands_use_native_comrecgc_and_frozen_rf_contract(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, dataset="aids")
    commands = continuation.build_stage_commands(
        inputs,
        project_commit="d" * 40,
        candidate_count=100262,
        teacher_sha256="e" * 64,
    )
    names = [row[0] for row in commands]
    assert names == ["common_recourse", "chemistry", "unified_eval", "full_gate", "freeze"]
    common = commands[0][1]
    chemistry = commands[1][1]
    evaluation = commands[2][1]
    gate = commands[3][1]
    assert "run_common_recourse.py" in " ".join(common)
    assert common[common.index("--generation-dir") + 1] == str(
        inputs.source_generation_root
    )
    assert chemistry[chemistry.index("--expected-candidate-count") + 1] == "100262"
    assert evaluation[evaluation.index("--teacher-path") + 1] == str(
        inputs.teacher_path
    )
    assert evaluation[evaluation.index("--theta-star") + 1] == "0.050000000000000003"
    assert evaluation.count("--device") == 1
    assert evaluation[evaluation.index("--device") + 1] == inputs.device
    assert gate[gate.index("--expected-teacher-sha256") + 1] == "e" * 64


def test_aids_external_engine_is_explicit_cpu_bounded_and_resumable(
    tmp_path: Path,
) -> None:
    pair_source = _file(tmp_path / "old-pair-store/run_manifest.json", "{}\n")
    close_view = _file(tmp_path / "close-view/close_pair_contract.json", "{}\n")
    inputs = replace(
        _inputs(tmp_path, dataset="aids"),
        device="cpu",
        common_recourse_engine="external_memory_exact_v1",
        external_max_rss_gb=96.0,
        external_query_block_size=8,
        external_checkpoint_interval_blocks=1,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_shortcut_seed_count=3,
        external_shortcut_failure_cap=4096,
        external_shortcut_query_block_size=65536,
        external_exact_fallback_max_samples=0,
        external_summary_block_size=65536,
        external_pair_store_source_manifest=pair_source,
        external_pair_store_source_owner_root=pair_source.parent,
        external_close_pair_view_manifest=close_view,
        expected_sklearn_version="1.7.2",
        common_recourse_resume=True,
    )
    command = continuation.build_stage_commands(
        inputs,
        project_commit="d" * 40,
        candidate_count=100262,
        teacher_sha256="e" * 64,
    )[0][1]
    assert command[command.index("--engine") + 1] == "external_memory_exact_v1"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--external-max-rss-gb") + 1] == "96"
    assert command[command.index("--external-query-block-size") + 1] == "8"
    assert command[command.index("--external-dbscan-shortcut-mode") + 1] == (
        "all_core_one_component_adaptive_anchor_v1"
    )
    assert command[command.index("--external-shortcut-seed-count") + 1] == "3"
    assert command[command.index("--external-shortcut-failure-cap") + 1] == "4096"
    assert command[command.index("--external-exact-fallback-max-samples") + 1] == "0"
    assert command[command.index("--external-summary-block-size") + 1] == "65536"
    assert command[
        command.index("--external-pair-store-source-manifest") + 1
    ] == str(pair_source)
    assert command[
        command.index("--external-pair-store-source-owner-root") + 1
    ] == str(pair_source.parent)
    assert command[
        command.index("--external-close-pair-view-manifest") + 1
    ] == str(close_view)
    assert command[command.index("--expected-sklearn-version") + 1] == "1.7.2"
    assert "--resume" in command


def test_pair_store_adoption_route_fails_before_output_on_contract_drift(
    tmp_path: Path,
) -> None:
    source_manifest = _file(tmp_path / "pair-store/run_manifest.json", "{}\n")
    inputs = replace(
        _inputs(tmp_path, dataset="aids"),
        external_pair_store_source_manifest=source_manifest,
        external_pair_store_source_owner_root=source_manifest.parent,
        common_recourse_engine="external_memory_exact_v1",
        external_dbscan_shortcut_mode="disabled",
        device="cpu",
    )
    with pytest.raises(ValueError, match="PAIR_STORE_ADOPTION_ROUTE_CONTRACT"):
        continuation.run_continuation(inputs)
    assert not inputs.output_root.exists()


def test_mut_pair_store_adoption_requires_float64_multi_component_route(
    tmp_path: Path,
) -> None:
    source_manifest = _file(tmp_path / "mut-pair-store/run_manifest.json", "{}\n")
    inputs = replace(
        _inputs(tmp_path, dataset="mutagenicity"),
        external_pair_store_source_manifest=source_manifest,
        external_pair_store_source_owner_root=source_manifest.parent,
        common_recourse_engine="external_memory_exact_v1",
        external_dbscan_shortcut_mode=(
            "sklearn_float64_exact_multi_component_v1"
        ),
        device="cpu",
    )
    continuation._validate_route_inputs(inputs)

    forbidden = replace(
        inputs,
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
    )
    with pytest.raises(ValueError, match="PAIR_STORE_ADOPTION_ROUTE_CONTRACT"):
        continuation._validate_route_inputs(forbidden)

    chunk_source = replace(
        inputs,
        external_pair_store_source_manifest=None,
        external_pair_store_source_checkpoint=source_manifest,
        external_close_pair_view_manifest=_file(tmp_path / "close.json", "{}\n"),
        external_vector_cache_root=tmp_path / "cache",
        external_vector_cache_lock=tmp_path / "cache.lock",
        external_vector_cache_route_lock=tmp_path / "route.lock",
    )
    with pytest.raises(ValueError, match="PAIR_STORE_ADOPTION_ROUTE_CONTRACT"):
        continuation._validate_route_inputs(chunk_source)


def test_chunk_source_route_freezes_cache_and_owner_arguments(
    tmp_path: Path, _fake_procfs: Path
) -> None:
    checkpoint = _file(tmp_path / "old/pair_store/checkpoint.json", "{}\n")
    owner = tmp_path / "old"
    cache = tmp_path / "local-cache"
    lock = tmp_path / "local-cache.lock"
    route_lock = tmp_path / "local-route.lock"
    close_view = _file(tmp_path / "close-view/close_pair_contract.json", "{}\n")
    inputs = replace(
        _inputs(tmp_path, dataset="aids"),
        device="cpu",
        common_recourse_engine="external_memory_exact_v1",
        external_dbscan_shortcut_mode=(
            "all_core_one_component_adaptive_anchor_v1"
        ),
        external_exact_fallback_max_samples=0,
        external_pair_store_source_checkpoint=checkpoint,
        external_pair_store_source_owner_root=owner,
        external_close_pair_view_manifest=close_view,
        external_vector_cache_root=cache,
        external_vector_cache_lock=lock,
        external_vector_cache_route_lock=route_lock,
        external_vector_cache_min_free_gb=3.0,
        external_vector_cache_proc_root=_fake_procfs,
        common_recourse_resume=True,
    )
    command = continuation.build_stage_commands(
        inputs,
        project_commit="d" * 40,
        candidate_count=100262,
        teacher_sha256="e" * 64,
    )[0][1]
    expected = {
        "--external-pair-store-source-checkpoint": str(checkpoint),
        "--external-pair-store-source-owner-root": str(owner),
        "--external-close-pair-view-manifest": str(close_view),
        "--external-vector-cache-root": str(cache),
        "--external-vector-cache-lock": str(lock),
        "--external-vector-cache-route-lock": str(route_lock),
        "--external-vector-cache-min-free-gb": "3",
        "--external-vector-cache-proc-root": str(_fake_procfs),
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert "--external-pair-store-source-manifest" not in command


def _external_common_terminal(root: Path) -> Path:
    pair_root = root / "external_memory/pair_store"
    pairs = _file(pair_root / "pairs.npy", "pairs")
    vectors = _file(pair_root / "vectors.npy", "vectors")
    pair_manifest = pair_root / "run_manifest.json"
    pair_identity = {"dataset": "aids", "fixture": "terminal-closure"}
    pair_identity_sha = continuation.stable_json_sha256(pair_identity)
    _json(
        pair_manifest,
        {
            "schema_version": continuation.PAIR_STORE_SCHEMA,
            "run_complete": True,
            "scientific_identity": pair_identity,
            "scientific_identity_sha256": pair_identity_sha,
            "pairs_path": str(pairs),
            "pairs_sha256": continuation.sha256_file(pairs),
            "vectors_path": str(vectors),
            "vectors_sha256": continuation.sha256_file(vectors),
        },
    )
    selected_json = _file(root / "selected_common_recourses.json", "[]\n")
    selected_csv = _file(root / "selected_common_recourses.csv", "rank\n")
    representatives = _file(root / "representative_counterfactuals.pt", "pt")
    run_manifest = root / "run_manifest.json"
    _json(
        run_manifest,
        {
            "run_complete": True,
            "common_recourse_engine": "external_memory_exact_v1",
            "theta_eligible_pair_count": 0,
            "external_memory_artifacts": {
                "pair_store_manifest": str(pair_manifest),
                "pair_store_manifest_sha256": continuation.sha256_file(pair_manifest),
                "pair_store_scientific_identity_sha256": pair_identity_sha,
                "pair_store_adopted_read_only": False,
                "pair_store_adoption_manifest": None,
                "pair_store_adoption_manifest_sha256": None,
                "dbscan_manifest": None,
            },
        },
    )
    marker = root / "_RUN_COMPLETE.json"
    _json(
        marker,
        {
            "schema_version": "comrecgc_common_recourse_terminal_v2",
            "run_complete": True,
            "common_recourse_engine": "external_memory_exact_v1",
            "artifact_sha256": {
                "run_manifest.json": continuation.sha256_file(run_manifest),
                "selected_common_recourses.json": continuation.sha256_file(
                    selected_json
                ),
                "selected_common_recourses.csv": continuation.sha256_file(
                    selected_csv
                ),
                "representative_counterfactuals.pt": continuation.sha256_file(
                    representatives
                ),
                "external_memory/pair_store/run_manifest.json": (
                    continuation.sha256_file(pair_manifest)
                ),
            },
        },
    )
    return marker


@pytest.mark.parametrize("failure_kind", ("missing_closure", "tampered_artifact"))
def test_fresh_external_common_stage_requires_full_hash_closure_before_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    root = tmp_path / "common"
    root.mkdir()
    marker = _external_common_terminal(root)
    if failure_kind == "missing_closure":
        terminal = json.loads(marker.read_text(encoding="utf-8"))
        terminal.pop("artifact_sha256")
        _json(marker, terminal)
    else:
        _file(root / "selected_common_recourses.json", "[{\"tampered\":true}]\n")
    output_root = tmp_path / "continuation"
    output_root.mkdir()
    checkpoint = output_root / "common-recourse-checkpoint.json"
    popen_calls: list[dict[str, object]] = []
    observed_launcher: list[str] = []
    held_read_fds: list[int] = []

    class Process:
        pid = 4242

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            del timeout
            while held_read_fds:
                continuation.os.close(held_read_fds.pop())
            return 0

    def fake_popen(args: object, **kwargs: object) -> object:
        observed_launcher[:] = list(args)  # type: ignore[arg-type]
        held_read_fds.append(continuation.os.dup(kwargs["pass_fds"][0]))
        popen_calls.append(kwargs)
        return Process()

    monkeypatch.setattr(
        continuation.subprocess, "Popen", fake_popen
    )
    monkeypatch.setattr(
        continuation, "_read_proc_start_ticks", lambda *args, **kwargs: 99
    )
    monkeypatch.setattr(
        continuation, "_proc_argv", lambda *args, **kwargs: observed_launcher
    )

    with pytest.raises(ValueError, match="RESUME_COMMON_TERMINAL"):
        continuation._run_stage(
            stage="common_recourse",
            argv=[
                "python",
                "run_common_recourse.py",
                "--engine",
                "external_memory_exact_v1",
            ],
            marker=marker,
            required_field="run_complete",
            environment={},
            output_root=output_root,
            checkpoint_path=checkpoint,
        )
    stage = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert stage["status"] == "FAILED"
    assert stage["child_pid"] == 4242
    assert stage["process_group_id"] == 4242
    assert (
        stage["process_group_contract"]
        == "durable_barrier_dedicated_child_session_v2"
    )
    assert stage["child_start_ticks"] == 99
    assert stage["startup_barrier"]["phase"] == "BOUND"
    assert len(popen_calls) == 1
    assert popen_calls[0]["cwd"] == continuation.PROJECT_ROOT
    assert popen_calls[0]["env"] == {}
    assert popen_calls[0]["start_new_session"] is True
    assert popen_calls[0]["close_fds"] is True
    assert len(popen_calls[0]["pass_fds"]) == 2
    assert not (output_root / "PASS").exists()


def test_forwarded_sigterm_never_allows_the_wrapper_to_continue_after_child_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "continuation"
    output_root.mkdir()
    marker = tmp_path / "chemistry" / "_RUN_COMPLETE.json"
    _json(marker, {"run_complete": True})
    checkpoint = output_root / "chemistry-checkpoint.json"
    installed: dict[int, object] = {}
    forwarded: list[tuple[int, int]] = []
    observed_launcher: list[str] = []
    held_read_fds: list[int] = []

    def fake_signal(signum: int, handler: object) -> object:
        previous = installed.get(signum, continuation.signal.SIG_DFL)
        installed[signum] = handler
        return previous

    class Process:
        pid = 4242

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            del timeout
            while held_read_fds:
                continuation.os.close(held_read_fds.pop())
            handler = installed[continuation.signal.SIGTERM]
            assert callable(handler)
            handler(continuation.signal.SIGTERM, None)
            return 0

    monkeypatch.setattr(continuation.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(continuation.signal, "signal", fake_signal)
    def fake_popen(args: object, **kwargs: object) -> Process:
        observed_launcher[:] = list(args)  # type: ignore[arg-type]
        held_read_fds.append(continuation.os.dup(kwargs["pass_fds"][0]))
        return Process()

    monkeypatch.setattr(continuation.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        continuation, "_read_proc_start_ticks", lambda *args, **kwargs: 99
    )
    monkeypatch.setattr(
        continuation, "_proc_argv", lambda *args, **kwargs: observed_launcher
    )
    monkeypatch.setattr(
        continuation.os,
        "killpg",
        lambda pid, signum: forwarded.append((pid, signum)),
    )

    with pytest.raises(
        continuation._StageTerminationRequested,
        match="STAGE_TERMINATED_AFTER_CHILD_COMPLETION",
    ):
        continuation._run_stage(
            stage="chemistry",
            argv=["python", "chemistry.py"],
            marker=marker,
            required_field="run_complete",
            environment={},
            output_root=output_root,
            checkpoint_path=checkpoint,
        )
    failed = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"
    assert failed["error_class"] == "_StageTerminationRequested"
    assert forwarded == [(4242, continuation.signal.SIGTERM)]
    assert not (output_root / "PASS").exists()


@pytest.mark.parametrize("signal_phase", ("during_arm", "during_launch"))
def test_stop_before_durable_science_release_never_executes_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_phase: str,
) -> None:
    output_root = tmp_path / "continuation"
    output_root.mkdir()
    marker = tmp_path / "chemistry" / "_RUN_COMPLETE.json"
    checkpoint = output_root / "chemistry-checkpoint.json"
    installed: dict[int, object] = {}
    observed = {"launch": 0, "release": 0, "abort": 0}

    def fake_signal(signum: int, handler: object) -> object:
        previous = installed.get(signum, continuation.signal.SIG_DFL)
        installed[signum] = handler
        return previous

    def request_stop() -> None:
        handler = installed[continuation.signal.SIGTERM]
        assert callable(handler)
        handler(continuation.signal.SIGTERM, None)

    class Process:
        pid = 4242

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            del timeout
            return 0

    class FakeBarrier:
        launcher_argv = ["python", "-S", "startup-wrapper"]

        @staticmethod
        def abort() -> None:
            observed["abort"] += 1

        @staticmethod
        def release() -> None:
            observed["release"] += 1

        @staticmethod
        def launch(**_kwargs: object) -> Process:
            observed["launch"] += 1
            if signal_phase == "during_launch":
                request_stop()
            return Process()

    def fake_arm(**kwargs: object) -> FakeBarrier:
        record_path = Path(str(kwargs["record_path"]))
        _json(record_path, {"fake": "authenticated-record"})
        if signal_phase == "during_arm":
            request_stop()
        return FakeBarrier()

    monkeypatch.setattr(continuation.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(continuation.signal, "signal", fake_signal)
    monkeypatch.setattr(continuation, "arm_exec_startup_barrier", fake_arm)

    with pytest.raises(
        continuation._StageTerminationRequested,
        match="STAGE_TERMINATED_BEFORE_SCIENCE_RELEASE",
    ):
        continuation._run_stage(
            stage="chemistry",
            argv=["python", "chemistry.py"],
            marker=marker,
            required_field="run_complete",
            environment={},
            output_root=output_root,
            checkpoint_path=checkpoint,
        )
    failed = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"
    assert failed["error_class"] == "_StageTerminationRequested"
    assert observed["release"] == 0
    assert observed["launch"] == (0 if signal_phase == "during_arm" else 1)
    assert observed["abort"] >= 1
    assert not marker.exists()


def test_completed_marker_cannot_bypass_bound_process_group_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "continuation"
    checkpoint = output_root / "stage_checkpoints/chemistry.json"
    argv = ["python", "chemistry.py"]
    lock_path, record_path = continuation._stage_startup_barrier_paths(
        output_root, stage="chemistry", generation=0
    )
    barrier = continuation.arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=argv,
        record_policy="fresh",
    )
    barrier.abort()
    binding = {
        "schema_version": continuation._STARTUP_BARRIER_BINDING_SCHEMA,
        "stage": "chemistry",
        "generation": 0,
        "phase": "BOUND",
        "record_path": str(record_path),
        "lock_path": str(lock_path),
        "target_argv_sha256": continuation.stable_json_sha256(argv),
        "record_sha256": continuation.sha256_file(record_path),
        "launcher_argv_sha256": continuation.stable_json_sha256(
            continuation.validate_startup_barrier_record(record_path).launcher_argv
        ),
    }
    _json(
        checkpoint,
        {
            "schema_version": 2,
            "status": "RUNNING",
            "stage": "chemistry",
            "argv_sha256": continuation.stable_json_sha256(argv),
            "process_group_contract": "durable_barrier_dedicated_child_session_v2",
            "process_group_id": 4242,
            "startup_barrier": binding,
        },
    )
    observed: list[int] = []

    def reject_live_group(process_group_id: int, **_kwargs: object) -> None:
        observed.append(process_group_id)
        raise ValueError("STAGE_CHILD_PROCESS_GROUP_NOT_QUIESCENT")

    monkeypatch.setattr(
        continuation, "_wait_for_process_group_quiescence", reject_live_group
    )
    with pytest.raises(ValueError, match="NOT_QUIESCENT"):
        continuation._require_completed_stage_writer_quiescence(
            output_root=output_root,
            stage="chemistry",
            argv=argv,
            checkpoint_path=checkpoint,
        )
    assert observed == [4242]


@pytest.mark.parametrize("window", ("temp_only", "final_and_temp"))
def test_continuation_prearm_reconciles_barrier_record_publication_crash(
    tmp_path: Path, window: str
) -> None:
    output_root = tmp_path / "continuation"
    output_root.mkdir()
    argv = ["python", "chemistry.py"]
    lock_path, record_path = continuation._stage_startup_barrier_paths(
        output_root, stage="chemistry", generation=0
    )
    barrier = continuation.arm_exec_startup_barrier(
        lock_path=lock_path,
        record_path=record_path,
        target_argv=argv,
    )
    barrier.abort()
    temporary = Path(f"{record_path}.tmp.fixture")
    if window == "temp_only":
        record_path.rename(temporary)
    else:
        continuation.os.link(record_path, temporary, follow_symlinks=False)
    binding = {
        "schema_version": continuation._STARTUP_BARRIER_BINDING_SCHEMA,
        "stage": "chemistry",
        "generation": 0,
        "phase": "PRE_ARM",
        "record_path": str(record_path),
        "lock_path": str(lock_path),
        "target_argv_sha256": continuation.stable_json_sha256(argv),
    }
    record = continuation._validate_stage_startup_binding(
        output_root=output_root,
        stage="chemistry",
        argv=argv,
        binding=binding,
        allowed_phases={"PRE_ARM"},
    )
    assert (record is not None) is (window == "final_and_temp")
    assert not temporary.exists()


def test_pass_marker_is_published_only_after_all_frozen_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *_args, **_kwargs: {"passed": True, "actual_commit": continuation.UPSTREAM_COMMIT},
    )
    monkeypatch.setattr(continuation, "_git_head", lambda: "f" * 40)
    monkeypatch.setattr(
        continuation, "_validate_common_recourse_completion", lambda **_kwargs: None
    )
    observed: list[str] = []

    def fake_stage(**kwargs) -> None:
        observed.append(str(kwargs["stage"]))
        if kwargs["stage"] == "freeze":
            standardized = inputs.output_root / "standardized"
            _json(
                standardized / "run_manifest.json",
                {
                    "dataset_key": "mutagenicity",
                    "cf_mode": continuation.CF_MODE,
                    "distance_line": continuation.DISTANCE_LINE,
                    "teacher_sha256": continuation.sha256_file(inputs.teacher_path),
                    "molclr_checkpoint_sha256": "1" * 64,
                    "dataset_csv_sha256": "2" * 64,
                },
            )
            _json(
                standardized / "freeze_manifest.json",
                {"dataset_key": "mutagenicity"},
            )

    monkeypatch.setattr(continuation, "_run_stage", fake_stage)
    result = continuation.run_continuation(inputs)
    assert observed == ["common_recourse", "chemistry", "unified_eval", "full_gate", "freeze"]
    assert result["status"] == "PASS"
    assert (inputs.output_root / "PASS").read_text(encoding="utf-8") == "PASS\n"
    assert json.loads((inputs.output_root / "_RUN_COMPLETE.json").read_text())["run_complete"] is True


def test_failure_keeps_evidence_and_never_publishes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(continuation, "_git_head", lambda: "f" * 40)

    def fail_stage(**_kwargs) -> None:
        raise RuntimeError("semantic lineage failure")

    monkeypatch.setattr(continuation, "_run_stage", fail_stage)
    with pytest.raises(RuntimeError, match="semantic lineage failure"):
        continuation.run_continuation(inputs)
    failure = json.loads((inputs.output_root / "FAILED.json").read_text())
    assert failure["status"] == "FAILED"
    assert not (inputs.output_root / "PASS").exists()


def test_exact_resume_adopts_only_hash_bound_completed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = replace(
        _inputs(tmp_path, dataset="aids"),
        device="cpu",
        common_recourse_engine="external_memory_exact_v1",
        common_recourse_resume=True,
    )
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *_args, **_kwargs: {
            "passed": True,
            "actual_commit": continuation.UPSTREAM_COMMIT,
        },
    )
    monkeypatch.setattr(continuation, "_git_head", lambda: "f" * 40)
    monkeypatch.setattr(
        continuation, "_validate_common_recourse_completion", lambda **_kwargs: None
    )

    def checkpoint_pass(**kwargs) -> None:
        marker = Path(kwargs["marker"])
        _json(marker, {kwargs["required_field"]: True})
        _json(
            kwargs["checkpoint_path"],
                {
                    "schema_version": 2,
                    "status": "PASS",
                    "stage": kwargs["stage"],
                    "runner_pid": 4242,
                    "child_pid": 4242,
                    "child_start_ticks": 99,
                    "process_group_id": 4242,
                    "process_group_contract": "dedicated_child_session_v1",
                    "argv_sha256": continuation.stable_json_sha256(
                    list(kwargs["argv"])
                ),
                "marker": str(marker),
                "required_field": kwargs["required_field"],
                "marker_sha256": continuation.sha256_file(marker),
            },
        )

    first_observed: list[str] = []
    common_stage_call: dict[str, object] = {}

    def interrupt_after_common(**kwargs) -> None:
        first_observed.append(kwargs["stage"])
        if kwargs["stage"] == "common_recourse":
            common_stage_call.update(kwargs)
            checkpoint_pass(**kwargs)
            checkpoint = json.loads(kwargs["checkpoint_path"].read_text())
            checkpoint["status"] = "RUNNING"
            checkpoint.pop("marker_sha256")
            _json(kwargs["checkpoint_path"], checkpoint)
            return
        marker = Path(kwargs["marker"])
        _file(marker.parent / "partial-child-output.json", "interrupted\n")
        _json(
            kwargs["checkpoint_path"],
            {
                "schema_version": 2,
                "status": "RUNNING",
                "stage": kwargs["stage"],
                "runner_pid": 4242,
                "process_group_id": 4242,
                "process_group_contract": "dedicated_child_session_v1",
                "argv_sha256": continuation.stable_json_sha256(
                    list(kwargs["argv"])
                ),
                "marker": str(marker),
                "required_field": kwargs["required_field"],
                "started_at": "before-sigterm",
            },
        )
        raise RuntimeError("diagnostic interruption")

    monkeypatch.setattr(continuation, "_run_stage", interrupt_after_common)
    with pytest.raises(RuntimeError, match="diagnostic interruption"):
        continuation.run_continuation(inputs)
    assert first_observed == ["common_recourse", "chemistry"]
    assert (inputs.output_root / "FAILED.json").is_file()

    resumed_observed: list[str] = []

    def finish_after_resume(**kwargs) -> None:
        resumed_observed.append(kwargs["stage"])
        checkpoint_pass(**kwargs)
        if kwargs["stage"] == "freeze":
            standardized = inputs.output_root / "standardized"
            _json(
                standardized / "run_manifest.json",
                {
                    "dataset_key": "aids",
                    "cf_mode": continuation.CF_MODE,
                    "distance_line": continuation.DISTANCE_LINE,
                    "teacher_sha256": continuation.sha256_file(inputs.teacher_path),
                    "molclr_checkpoint_sha256": "1" * 64,
                    "dataset_csv_sha256": "2" * 64,
                },
            )
            _json(standardized / "freeze_manifest.json", {"dataset_key": "aids"})

    monkeypatch.setattr(continuation, "_run_stage", finish_after_resume)
    result = continuation.run_continuation(inputs)
    assert resumed_observed == ["chemistry", "unified_eval", "full_gate", "freeze"]
    assert result["status"] == "PASS"
    common_checkpoint = json.loads(
        (inputs.output_root / "stage_checkpoints/common_recourse.json").read_text()
    )
    assert common_checkpoint["reconciled_after_child_completion"] is True
    assert list((inputs.output_root / "failure_history").glob("FAILED.*.json"))
    archived = list(
        (inputs.output_root / "partial_stage_history").glob("chemistry.*")
    )
    archived_payloads = [path for path in archived if path.is_dir()]
    archived_authorities = [path for path in archived if path.suffix == ".json"]
    assert len(archived_payloads) == 1
    assert len(archived_authorities) == 1
    assert (archived_payloads[0] / "partial-child-output.json").read_text() == (
        "interrupted\n"
    )
    continuation._require_completed_stage_writer_quiescence(
        output_root=inputs.output_root,
        stage="common_recourse",
        argv=common_stage_call["argv"],
        checkpoint_path=common_stage_call["checkpoint_path"],
    )
    assert (
        continuation._validate_completed_stage(
            stage="common_recourse",
            argv=common_stage_call["argv"],
            marker=common_stage_call["marker"],
            required_field=common_stage_call["required_field"],
            checkpoint_path=common_stage_call["checkpoint_path"],
        )
        is False
    )


def test_partial_stage_archive_refuses_a_live_old_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    marker = root / "chemistry/_RUN_COMPLETE.json"
    _file(marker.parent / "partial.json")
    argv = ["python", "chemistry.py"]
    checkpoint = root / "stage_checkpoints/chemistry.json"
    _json(
        checkpoint,
        {
            "schema_version": 2,
            "status": "RUNNING",
            "stage": "chemistry",
            "runner_pid": 4242,
            "process_group_id": 4242,
            "process_group_contract": "dedicated_child_session_v1",
            "argv_sha256": continuation.stable_json_sha256(argv),
            "marker": str(marker),
            "required_field": "run_complete",
        },
    )
    monkeypatch.setattr(
        continuation,
        "_process_group_member_pids",
        lambda *args, **kwargs: (4243,),
    )
    with pytest.raises(ValueError, match="PROCESS_GROUP_STILL_LIVE"):
        continuation._archive_noncheckpointed_partial_stage(
            stage="chemistry",
            argv=argv,
            marker=marker,
            required_field="run_complete",
            checkpoint_path=checkpoint,
            output_root=root,
        )
    assert (marker.parent / "partial.json").is_file()
    assert not (root / "partial_stage_history").exists()


def test_partial_stage_archive_authority_replays_after_proc_audit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    marker = root / "chemistry/_RUN_COMPLETE.json"
    _file(marker.parent / "partial.json", "preserve-me\n")
    argv = ["python", "chemistry.py"]
    checkpoint = root / "stage_checkpoints/chemistry.json"
    _json(
        checkpoint,
        {
            "schema_version": 2,
            "status": "FAILED",
            "stage": "chemistry",
            "runner_pid": 4242,
            "process_group_id": 4242,
            "process_group_contract": "dedicated_child_session_v1",
            "argv_sha256": continuation.stable_json_sha256(argv),
            "marker": str(marker),
            "required_field": "run_complete",
        },
    )
    monkeypatch.setattr(
        continuation, "_process_group_member_pids", lambda *args, **kwargs: ()
    )
    scan_count = 0

    def drifting_proc_audit(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal scan_count
        scan_count += 1
        return {"all_writers_quiescent": True, "scanned_process_count": scan_count}

    monkeypatch.setattr(continuation, "_scan_live_source_writers", drifting_proc_audit)
    real_rename = continuation.os.rename
    monkeypatch.setattr(
        continuation.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("crash-after-authority-before-rename")
        ),
    )
    with pytest.raises(RuntimeError, match="crash-after-authority"):
        continuation._archive_noncheckpointed_partial_stage(
            stage="chemistry",
            argv=argv,
            marker=marker,
            required_field="run_complete",
            checkpoint_path=checkpoint,
            output_root=root,
        )
    authority = next((root / "partial_stage_history").glob("*.archive.json"))
    first = json.loads(authority.read_text(encoding="utf-8"))
    assert first["publication_audit"]["live_writer_audit_before"][
        "scanned_process_count"
    ] == 1
    monkeypatch.setattr(continuation.os, "rename", real_rename)
    archived = continuation._archive_noncheckpointed_partial_stage(
        stage="chemistry",
        argv=argv,
        marker=marker,
        required_field="run_complete",
        checkpoint_path=checkpoint,
        output_root=root,
    )
    assert (archived / "partial.json").read_text(encoding="utf-8") == "preserve-me\n"
    assert scan_count >= 4


def test_partial_stage_archive_enforces_the_budgeted_one_gibibyte_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    marker = root / "chemistry/_RUN_COMPLETE.json"
    oversized = _file(marker.parent / "oversized.bin", "")
    oversized.write_bytes(b"")
    with oversized.open("r+b") as stream:
        stream.truncate(continuation._PARTIAL_STAGE_ARCHIVE_MAX_BYTES + 1)
    argv = ["python", "chemistry.py"]
    checkpoint = root / "stage_checkpoints/chemistry.json"
    _json(
        checkpoint,
        {
            "schema_version": 2,
            "status": "FAILED",
            "stage": "chemistry",
            "runner_pid": 4242,
            "process_group_id": 4242,
            "process_group_contract": "dedicated_child_session_v1",
            "argv_sha256": continuation.stable_json_sha256(argv),
            "marker": str(marker),
            "required_field": "run_complete",
        },
    )
    monkeypatch.setattr(
        continuation, "_process_group_member_pids", lambda *args, **kwargs: ()
    )
    monkeypatch.setattr(
        continuation,
        "_scan_live_source_writers",
        lambda *_args, **_kwargs: {"all_writers_quiescent": True},
    )
    with pytest.raises(ValueError, match="ARCHIVE_BYTE_LIMIT_EXCEEDED"):
        continuation._archive_noncheckpointed_partial_stage(
            stage="chemistry",
            argv=argv,
            marker=marker,
            required_field="run_complete",
            checkpoint_path=checkpoint,
            output_root=root,
        )
    assert oversized.exists()
    assert not list((root / "partial_stage_history").glob("*.archive.json"))


def test_recovery_budget_and_continuation_archive_caps_are_identical() -> None:
    from src.utils import (
        autodl_aids_comrecgc_exact_recovery_controller_v1 as recovery_controller,
    )

    assert continuation._PARTIAL_STAGE_ARCHIVE_MAX_BYTES == (
        recovery_controller.PARTIAL_STAGE_ARCHIVE_MAX_BYTES
    )


def test_immutable_json_publisher_recovers_a_partial_temporary_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    payload = {"schema_version": "fixture_v1", "status": "READY"}
    real_write = continuation.os.write
    crashed = False

    def partial_write_then_crash(descriptor: int, data: bytes) -> int:
        nonlocal crashed
        if not crashed:
            crashed = True
            real_write(descriptor, data[: max(1, len(data) // 2)])
            raise OSError("injected-mid-publication-crash")
        return real_write(descriptor, data)

    monkeypatch.setattr(continuation.os, "write", partial_write_then_crash)
    with pytest.raises(OSError, match="injected-mid-publication-crash"):
        continuation._write_new_json(receipt, payload)
    assert not receipt.exists()
    temporary = tmp_path / ".receipt.json.partial"
    assert temporary.is_file()
    monkeypatch.setattr(continuation.os, "write", real_write)
    changed_audit_payload = {
        **payload,
        "publication_audit": {"scanned_process_count": 17},
    }
    continuation._write_new_json(receipt, changed_audit_payload)
    assert json.loads(receipt.read_text(encoding="utf-8")) == changed_audit_payload
    assert not temporary.exists()


def test_exact_resume_fails_closed_on_same_path_input_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = replace(
        _inputs(tmp_path, dataset="aids"),
        device="cpu",
        common_recourse_engine="external_memory_exact_v1",
        common_recourse_resume=True,
    )
    monkeypatch.setattr(
        continuation,
        "verify_checkout",
        lambda *_args, **_kwargs: {
            "passed": True,
            "actual_commit": continuation.UPSTREAM_COMMIT,
        },
    )
    monkeypatch.setattr(continuation, "_git_head", lambda: "f" * 40)
    monkeypatch.setattr(
        continuation,
        "_run_stage",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupt")),
    )
    with pytest.raises(RuntimeError, match="interrupt"):
        continuation.run_continuation(inputs)
    inputs.distance_checkpoint.write_text("changed-distance", encoding="utf-8")
    with pytest.raises(ValueError, match="RESUME_SCIENTIFIC_CONTRACT_MISMATCH"):
        continuation.run_continuation(inputs)
