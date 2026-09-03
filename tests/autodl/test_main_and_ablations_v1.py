from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/autodl/run_main_and_ablations_v1.py"
SPEC = importlib.util.spec_from_file_location("run_main_and_ablations_v1", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "RUN_MAIN_TABLE": "1",
        "RUN_LLM_ABLATION": "1",
        "RUN_GNN_ABLATION": "1",
        "MAIN_TABLE_PRIORITY": "100",
        "LLM_ABLATION_PRIORITY": "50",
        "GNN_ABLATION_PRIORITY": "20",
        "ALLOW_MUT_CONTINUATION_RECOVERY": "1",
        "MUT_CPU_WORKERS": "2",
        "MUT_BASELINE_WINDOW_SECONDS": "1800",
        "MUT_SLOWDOWN_THRESHOLD": "0.15",
        "MUT_SLOWDOWN_SUSTAIN_SECONDS": "1200",
        "ALLOW_T14_EXTERNAL_CONVERGENCE_AUDITOR": "1",
        "ALLOW_TASTE_GLOBALGCE_VALID_ZERO_RULE_RESULT": "1",
        "T8_RECOVERY_MAX_ATTEMPTS": "1",
        "LLM_EARLY_START_MIN_MATRIX": "13",
        "LLM_EARLY_START_IDLE_SECONDS": "1200",
        "LLM_MAX_EARLY_GPUS": "1",
        "RUN_MATCHED_SFT_AUXILIARY_STUDY": "0",
        "GNN_START_AFTER_MATRIX": "16",
        "GNN_PRIMARY_SEEDS": "7",
        "GNN_MAX_CONCURRENT_GPUS": "2",
        "RUN_GRAPH_MAMBA": "0",
        "LLM_CORE_VARIANTS": (
            "BRICS_FIXED,CHEMLLM_7B_OFF_THE_SHELF,"
            "CHEMLLM_7B_PPO_LORA_MAIN,CHEMLLM_2B_OFF_THE_SHELF"
        ),
        "GNN_BACKBONES": "gine,gin,gcn,gatv2,gatedgcn_plus",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _matrix(path: Path, cells: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "fast16_matrix_authority_pointer_v1",
                "latest_count": len(cells),
                "applied_cells": cells,
                "latest_authority_root": "/runtime/authority",
                "latest_combined_audit_sha256": "a" * 64,
                "latest_matrix_status_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )


def _pass(path: Path) -> None:
    path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")


def _task_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    component: str = "mut_continuation",
    output_mode: str = "fresh",
) -> tuple[Path, dict[str, object]]:
    fixture = tmp_path / component
    fixture.mkdir(parents=True, exist_ok=True)
    manifest = fixture / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    input_root = fixture / "input"
    input_root.mkdir()
    output_root = fixture / "output-{attempt_uuid}"
    if output_mode == "resume":
        output_root = fixture / "resume-output"
        output_root.mkdir()
    owner_heartbeat = fixture / "owner.json"
    terminal = fixture / "terminal.json"
    required_environment: dict[str, str] = {
        "AUTODL_PYTHON": sys.executable,
        "RUN_GNN_ABLATION": "0",
    }
    for name in module.COMPONENT_REQUIRED_ENV.get(component, frozenset()):
        required_environment.setdefault(name, str(manifest))
    required_environment.update(
        {
            "AUTODL_DATA_ROOT": str(fixture),
            "AUTODL_RUNTIME_ROOT": str(fixture),
            "AUTODL_CONTROL_ROOT": str(fixture),
            "AUTODL_PYTHON": sys.executable,
            "RUN_GNN_ABLATION": "0",
        }
    )
    if component == "mut_continuation":
        required_environment["MUT_TRACE_CONTROLLER_PID"] = "123"
        required_environment["MUT_TRACE_CONTROLLER_START_TICKS"] = "456"
        required_environment["MUT_TRACE_OUTPUT_ROOT"] = str(output_root)
    if component == "t14_resume":
        resume_spec = fixture / "resume-spec.json"
        resume_spec.write_text("{}\n", encoding="utf-8")
        required_environment.update(
            {
                "T14_AUDITOR_REPO_ROOT": str(PROJECT_ROOT),
                "T14_CHECKPOINT_ROOT": str(input_root),
                "T14_RESUME_SPEC": str(resume_spec),
                "TASTEMOLNET_T14_RESUME": "1",
                "TASTEMOLNET_T14_OUTPUT": str(output_root),
                "TASTEMOLNET_T14_GPU_INDEX": "2",
                "TASTEMOLNET_T14_RUN_ID": "fixture-{attempt_uuid}",
                "RUN_TASTEMOLNET": "1",
                "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
                "TASTE_PAPER_RESULTS_ALLOWED": "1",
                "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_request: dict[str, object] = {"required": False}
    memory_request: dict[str, object] = {"required_headroom_bytes": 0}
    if component == "mut_continuation":
        gpu_request = {
            "required": True,
            "index": 0,
            "lease_path": str(fixture / "gpu0.lock"),
        }
    if component == "t14_resume":
        gpu_request = {
            "required": True,
            "index": 2,
            "lease_path": str(fixture / "gpu2.lock"),
        }
        memory_limit = fixture / "memory.limit"
        memory_current = fixture / "memory.current"
        memory_limit.write_text("1000\n", encoding="utf-8")
        memory_current.write_text("0\n", encoding="utf-8")
        memory_request = {
            "required_headroom_bytes": 1,
            "limit_path": str(memory_limit),
            "current_path": str(memory_current),
        }
    spec: dict[str, object] = {
        "schema_version": module.TASK_SPEC_SCHEMA,
        "task_id": f"fixture-{component}",
        "task_type": component,
        "repo_root": str(PROJECT_ROOT),
        "execution_commit": commit,
        "python": sys.executable,
        "config": str(PROJECT_ROOT / "configs/hpc.yaml"),
        "manifest": str(manifest),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "output_mode": output_mode,
        "gpu_request": gpu_request,
        "cpu_request": {"workers": 2},
        "memory_request": memory_request,
        "required_environment": required_environment,
        "launcher": module.DEFAULT_LAUNCHERS[component],
        "owner": {
            "heartbeat_path": str(owner_heartbeat),
            "heartbeat_schema": "fixture_owner_v1",
            "pid_field": "science_pid",
            "start_ticks_field": "science_start_ticks",
            "output_root_field": "output_root",
            "timestamp_field": "written_at",
            "max_age_seconds": 120,
            "command_sha256": "a" * 64,
            "command_contains": ["science.py"],
            "cwd": str(PROJECT_ROOT),
        },
        "terminal": {
            "receipt_path": str(terminal),
            "schema_version": "fixture_terminal_v1",
            "state_field": "state",
            "terminal_states": ["PASS", "FAILED"],
            "output_root_field": "output_root",
        },
    }
    if component == "t14_resume":
        spec["serial_auditor"] = {"active": False}
    path = fixture / "task-spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setenv(module.TASK_SPEC_ENV[component], str(path))
    return path, spec


def test_policy_is_strict_and_keeps_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _policy_env(monkeypatch)
    policy = module.validate_policy()
    assert policy["main_priority"] > policy["llm_priority"] > policy["gnn_priority"]
    assert policy["graph_mamba_run_enabled"] is False
    monkeypatch.setenv("MUT_CPU_WORKERS", "4")
    with pytest.raises(ValueError, match="MUT_CPU_WORKERS"):
        module.validate_policy()


def test_at_12_dispatches_only_missing_main_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _policy_env(monkeypatch)
    matrix = tmp_path / "matrix.json"
    cells = [
        "AIDS/Ours",
        "AIDS/GCFExplainer",
        "AIDS/GlobalGCE",
        "AIDS/ComRecGC",
        "Mutagenicity/Ours",
        "Mutagenicity/GCFExplainer",
        "Mutagenicity/GlobalGCE",
        "BACE/Ours",
        "BACE/GCFExplainer",
        "BACE/GlobalGCE",
        "BACE/ComRecGC",
        "TasteMolNet/Ours",
    ]
    _matrix(matrix, cells)
    attempt_receipt = tmp_path / "t8-attempt.json"
    attempt_receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TASTE_GLOBALGCE_ATTEMPT_RECEIPT", str(attempt_receipt))
    called: list[str] = []

    def fake_dispatch(_state, _project, component: str, **_kwargs):
        called.append(component)
        return {"state": "WOULD_DISPATCH"}

    monkeypatch.setattr(module, "_dispatch", fake_dispatch)
    result = module.observe_and_dispatch(
        state_root=tmp_path / "state",
        project_root=PROJECT_ROOT,
        matrix_path=matrix,
        policy=module.validate_policy(),
        dry_run=True,
    )
    assert called == [
        "mut_continuation",
        "t14_resume",
        "t8_valid_zero_finalizer",
    ]
    assert result["components"]["llm_ablation"]["state"] == "BLOCKED_MAIN_PRIORITY"
    assert result["components"]["gnn_ablation"]["state"] == "BLOCKED_WAITING_FINAL_MAIN"


def test_live_mut_owner_is_adopted_without_duplicate_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _policy_env(monkeypatch)
    matrix = tmp_path / "matrix.json"
    _matrix(matrix, ["Mutagenicity/Ours"])
    heartbeat = tmp_path / "mut-heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "state": "PROTECTED_BASELINE_RUNNING",
                "worker_pid": 4321,
                "worker_start_ticks": 999,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MUT_CONTINUATION_HEARTBEAT", str(heartbeat))
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: {"pid": pid, "alive": True, "start_ticks": 999},
    )
    called: list[str] = []

    def fake_dispatch(_state, _project, component: str, **_kwargs):
        called.append(component)
        return {"state": "WOULD_DISPATCH"}

    monkeypatch.setattr(module, "_dispatch", fake_dispatch)
    result = module.observe_and_dispatch(
        state_root=tmp_path / "state",
        project_root=PROJECT_ROOT,
        matrix_path=matrix,
        policy=module.validate_policy(),
        dry_run=True,
    )
    assert result["components"]["mut_continuation"]["state"] == "ADOPTED_LIVE_OWNER"
    assert "mut_continuation" not in called


def test_live_t14_relay_is_adopted_without_duplicate_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _policy_env(monkeypatch)
    matrix = tmp_path / "matrix.json"
    _matrix(matrix, ["TasteMolNet/Ours"])
    heartbeat = tmp_path / "t14-heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "schema_version": (
                    "tastemolnet_t14_external_convergence_relay_heartbeat_v1"
                ),
                "written_at": datetime.now(timezone.utc).isoformat(),
                "controller_pid": 7654,
                "phase": "WAITING_FOR_12500",
                "audited_through_step": 0,
                "converged": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("T14_AUDITOR_RELAY_HEARTBEAT", str(heartbeat))
    monkeypatch.setenv("T14_AUDITOR_RELAY_START_TICKS", "8765")
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: {"pid": pid, "alive": True, "start_ticks": 8765},
    )
    called: list[str] = []

    def fake_dispatch(_state, _project, component: str, **_kwargs):
        called.append(component)
        return {"state": "WOULD_DISPATCH"}

    monkeypatch.setattr(module, "_dispatch", fake_dispatch)
    result = module.observe_and_dispatch(
        state_root=tmp_path / "state",
        project_root=PROJECT_ROOT,
        matrix_path=matrix,
        policy=module.validate_policy(),
        dry_run=True,
    )
    assert (
        result["components"]["t14_convergence_auditor"]["state"]
        == "ADOPTED_LIVE_RELAY"
    )
    assert (
        result["components"]["t14_resume"]["state"]
        == "BLOCKED_SERIAL_AUDITOR_ACTIVE"
    )
    assert "t14_resume" not in called


def test_t8_zero_relay_fails_closed_without_authoritative_attempt_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _policy_env(monkeypatch)
    matrix = tmp_path / "matrix.json"
    _matrix(matrix, ["TasteMolNet/Ours"])
    monkeypatch.delenv("TASTE_GLOBALGCE_ATTEMPT_RECEIPT", raising=False)
    called: list[str] = []

    def fake_dispatch(_state, _project, component: str, **_kwargs):
        called.append(component)
        return {"state": "WOULD_DISPATCH"}

    monkeypatch.setattr(module, "_dispatch", fake_dispatch)
    result = module.observe_and_dispatch(
        state_root=tmp_path / "state",
        project_root=PROJECT_ROOT,
        matrix_path=matrix,
        policy=module.validate_policy(),
        dry_run=True,
    )
    assert result["components"]["t8_valid_zero_finalizer"] == {
        "state": "BLOCKED_MISSING_AUTHORITATIVE_ATTEMPT_RECEIPT",
        "required_env": "TASTE_GLOBALGCE_ATTEMPT_RECEIPT",
    }
    assert "t8_valid_zero_finalizer" not in called


def test_main_owner_manifest_is_conservative_ready_queue_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = tmp_path / "matrix.json"
    cells = [
        "Mutagenicity/ComRecGC",
        "TasteMolNet/Ours",
    ]
    _matrix(matrix, cells)
    owner_manifest = tmp_path / "owners.json"
    owner_manifest.write_text(
        json.dumps(
            {
                "schema_version": "main_live_owner_manifest_v1",
                "owners": [
                    {"cell": "TasteMolNet/GCFExplainer", "pid": 12, "start_ticks": 120},
                    {"cell": "TasteMolNet/GlobalGCE", "pid": 13, "start_ticks": 130},
                    {"cell": "TasteMolNet/ComRecGC", "pid": 14, "start_ticks": 140},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MAIN_READY_QUEUE", raising=False)
    monkeypatch.setenv("MAIN_OWNER_MANIFEST", str(owner_manifest))
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: {"pid": pid, "alive": True, "start_ticks": pid * 10},
    )
    ready, reason = module._main_ready_waiting(module._matrix(matrix))
    assert ready is False
    assert reason.endswith("=0")
    payload = json.loads(owner_manifest.read_text(encoding="utf-8"))
    payload["owners"][0]["start_ticks"] = 999
    owner_manifest.write_text(json.dumps(payload), encoding="utf-8")
    ready, reason = module._main_ready_waiting(module._matrix(matrix))
    assert ready is True
    assert "TasteMolNet/GCFExplainer" in reason


def test_gnn_dispatch_requires_16_and_all_final_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _policy_env(monkeypatch)
    matrix = tmp_path / "matrix.json"
    datasets = ("AIDS", "Mutagenicity", "BACE", "TasteMolNet")
    methods = ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC")
    _matrix(matrix, [f"{dataset}/{method}" for dataset in datasets for method in methods])
    ready = tmp_path / "ready.json"
    ready.write_text(json.dumps({"ready_waiting_gpu": []}), encoding="utf-8")
    monkeypatch.setenv("MAIN_READY_QUEUE", str(ready))
    llm = tmp_path / "llm.json"
    _pass(llm)
    monkeypatch.setenv("LLM_EARLY_GATE_RECEIPT", str(llm))
    for variable in (
        "FINAL_MATRIX_AUDIT_RECEIPT",
        "FINAL_FIGURE3_RECEIPT",
        "FINAL_FIGURE4_RECEIPT",
        "FINAL_TABLE2_RECEIPT",
    ):
        receipt = tmp_path / f"{variable}.json"
        _pass(receipt)
        monkeypatch.setenv(variable, str(receipt))
    called: list[str] = []

    def fake_dispatch(_state, _project, component: str, **_kwargs):
        called.append(component)
        return {"state": "WOULD_DISPATCH"}

    monkeypatch.setattr(module, "_dispatch", fake_dispatch)
    result = module.observe_and_dispatch(
        state_root=tmp_path / "state",
        project_root=PROJECT_ROOT,
        matrix_path=matrix,
        policy=module.validate_policy(),
        dry_run=True,
    )
    assert called == ["llm_ablation", "gnn_ablation"]
    assert result["matrix_complete_cells"] == 16


def test_launch_and_slurm_contracts() -> None:
    launcher = (PROJECT_ROOT / "scripts/autodl/launch_main_and_ablations_v1.sh").read_text()
    assert "nohup" in launcher
    assert "--set inference.fallback_to_heuristic=false" in launcher
    assert "SIGKILL" not in launcher and "pkill" not in launcher
    for name in ("run_main_and_ablations_v1.sh", "status_main_and_ablations_v1.sh"):
        text = (PROJECT_ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "source ~/.bashrc" in text
        assert "conda activate smiles_pip118" in text
        assert "cd /share/home/u20526/czx/counterfactual-subgraph" in text
        assert "export PYTHONPATH=$PWD" in text
        assert "--config configs/hpc.yaml" in text
        assert "--set inference.fallback_to_heuristic=false" in text


def test_sidecar_uses_t14_science_resume_not_auditor_as_owner() -> None:
    assert module.DEFAULT_LAUNCHERS["t14_resume"].endswith(
        "run_tastemolnet_t14_comrecgc_full.sh"
    )
    assert "t14_convergence_auditor" not in module.DEFAULT_LAUNCHERS


def test_sidecar_uses_waiting_globalgce_zero_relay() -> None:
    assert module.DEFAULT_LAUNCHERS["t8_valid_zero_finalizer"].endswith(
        "launch_tastemolnet_globalgce_valid_zero_relay_v1.sh"
    )


def test_missing_task_spec_is_dispatch_config_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MUT_CONTINUATION_TASK_SPEC", raising=False)
    result = module._dispatch(
        tmp_path / "state",
        PROJECT_ROOT,
        "mut_continuation",
        reason="fixture",
        dry_run=True,
    )
    assert result["state"] == "DISPATCH_CONFIG_INVALID"
    assert "MUT_CONTINUATION_TASK_SPEC" in result["error"]


def test_component_required_environment_is_loaded_from_task_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, spec = _task_spec(tmp_path, monkeypatch)
    loaded_path, loaded, digest = module._load_task_spec(
        PROJECT_ROOT, "mut_continuation"
    )
    assert loaded_path == path
    assert loaded["required_environment"] == spec["required_environment"]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["required_environment"]["MUT_FAST_SPEC"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="MUT_FAST_SPEC"):
        module._load_task_spec(PROJECT_ROOT, "mut_continuation")


def test_attempt_template_freezes_exact_owner_command_hash() -> None:
    spec = {
        "owner": {
            "command_argv": [
                "/python",
                "science.py",
                "--run-id",
                "run-{attempt_uuid}",
                "--output",
                "{output_root}",
            ]
        }
    }
    observed = module._resolved_owner_command_sha256(
        spec,
        None,
        attempt_uuid="abc",
        attempt=2,
        output_root="/runtime/output-abc",
    )
    expected = module._command_sha256_from_argv(
        [
            "/python",
            "science.py",
            "--run-id",
            "run-abc",
            "--output",
            "/runtime/output-abc",
        ]
    )
    assert observed == expected


def test_t14_task_spec_requires_resume_and_auditor_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _task_spec(
        tmp_path, monkeypatch, component="t14_resume", output_mode="resume"
    )
    _, loaded, _ = module._load_task_spec(PROJECT_ROOT, "t14_resume")
    environment = loaded["required_environment"]
    assert environment["TASTEMOLNET_T14_RESUME"] == "1"
    assert environment["T14_AUDITOR_REPO_ROOT"] == str(PROJECT_ROOT)

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["required_environment"]["T14_RESUME_SPEC"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="T14_RESUME_SPEC"):
        module._load_task_spec(PROJECT_ROOT, "t14_resume")


def test_owner_confirmation_requires_all_five_identity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, spec = _task_spec(tmp_path, monkeypatch)
    expected_output = str(tmp_path / "resolved-output")
    owner = spec["owner"]
    assert isinstance(owner, dict)
    heartbeat_path = Path(str(owner["heartbeat_path"]))
    heartbeat_path.write_text(
        json.dumps(
            {
                "schema_version": "fixture_owner_v1",
                "science_pid": 4321,
                "science_start_ticks": 9876,
                "output_root": expected_output,
                "written_at": datetime.fromtimestamp(1_000, tz=timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    identity = {
        "pid": 4321,
        "alive": True,
        "start_ticks": 9876,
        "command": "python science.py --output resolved-output",
        "command_sha256": "a" * 64,
        "cwd": str(PROJECT_ROOT),
    }
    monkeypatch.setattr(module, "_process_identity", lambda _pid: dict(identity))
    confirmed = module._owner_probe(
        spec,
        expected_output=expected_output,
        expected_command_sha256="a" * 64,
        expected_start_ticks=None,
        now_epoch=1_030,
    )
    assert confirmed["state"] == "OWNER_CONFIRMED"

    for changed in (
        {"alive": False},
        {"start_ticks": 9877},
        {"command_sha256": "b" * 64},
        {"cwd": str(tmp_path)},
        {"command": "python another_program.py"},
    ):
        observed = dict(identity)
        observed.update(changed)
        monkeypatch.setattr(module, "_process_identity", lambda _pid, row=observed: row)
        result = module._owner_probe(
            spec,
            expected_output=expected_output,
            expected_command_sha256="a" * 64,
            expected_start_ticks=None,
            now_epoch=1_030,
        )
        if changed == {"alive": False}:
            assert result["state"] == "ABSENT"
        else:
            assert result["state"] == "INVALID"


def test_t14_progress_can_bind_true_science_start_ticks_without_runner_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, spec = _task_spec(
        tmp_path, monkeypatch, component="t14_resume", output_mode="resume"
    )
    output = Path(str(spec["output_root"]))
    owner = spec["owner"]
    assert isinstance(owner, dict)
    owner.update(
        {
            "heartbeat_path": str(output / "progress.json"),
            "heartbeat_schema": "tastemolnet_t14_progress_v1",
            "pid_field": "pid",
            "start_ticks_field": None,
            "output_root_field": None,
            "output_root_from_heartbeat_parent": True,
            "timestamp_field": "updated_at",
            "command_sha256": "c" * 64,
            "command_contains": ["run_tastemolnet_comrecgc_full.py"],
        }
    )
    (output / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": "tastemolnet_t14_progress_v1",
                "pid": 7654,
                "updated_at": datetime.fromtimestamp(
                    2_000, tz=timezone.utc
                ).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda _pid: {
            "alive": True,
            "start_ticks": 555,
            "command": "python run_tastemolnet_comrecgc_full.py",
            "command_sha256": "c" * 64,
            "cwd": str(PROJECT_ROOT),
        },
    )
    candidate = module._owner_probe(
        spec,
        expected_output=str(output),
        expected_command_sha256="c" * 64,
        expected_start_ticks=None,
        now_epoch=2_010,
    )
    assert candidate["state"] == "OWNER_CANDIDATE"
    confirmed = module._owner_probe(
        spec,
        expected_output=str(output),
        expected_command_sha256="c" * 64,
        expected_start_ticks=555,
        now_epoch=2_010,
    )
    assert confirmed["state"] == "OWNER_CONFIRMED"


def test_terminal_receipt_prevents_duplicate_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, spec = _task_spec(tmp_path, monkeypatch)
    output = tmp_path / "terminal-output"
    spec["output_root"] = str(output)
    environment = spec["required_environment"]
    assert isinstance(environment, dict)
    environment["MUT_TRACE_OUTPUT_ROOT"] = str(output)
    owner = spec["owner"]
    assert isinstance(owner, dict)
    owner["expected_output_root"] = str(output)
    path.write_text(json.dumps(spec), encoding="utf-8")
    terminal = spec["terminal"]
    assert isinstance(terminal, dict)
    Path(str(terminal["receipt_path"])).write_text(
        json.dumps(
            {
                "schema_version": "fixture_terminal_v1",
                "state": "PASS",
                "output_root": str(output),
            }
        ),
        encoding="utf-8",
    )
    result = module._dispatch(
        tmp_path / "state",
        PROJECT_ROOT,
        "mut_continuation",
        reason="fixture",
        dry_run=False,
    )
    assert result["state"] == "TERMINAL"
    assert result["terminal_state"] == "PASS"


@pytest.mark.parametrize(
    ("attempt_count", "expected_backoff"), [(1, 60), (2, 120), (3, 300)]
)
def test_dead_launcher_without_owner_enters_bounded_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_count: int,
    expected_backoff: int,
) -> None:
    path, spec = _task_spec(tmp_path, monkeypatch)
    component_root = tmp_path / "state/components/mut_continuation"
    component_root.mkdir(parents=True)
    output = tmp_path / f"attempt-{attempt_count}-output"
    exit_receipt = component_root / f"exit-{attempt_count}.json"
    exit_receipt.write_text(
        json.dumps(
            {
                "schema_version": module.LAUNCH_EXIT_SCHEMA,
                "returncode": 64,
                "written_at": datetime.fromtimestamp(10, tz=timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    state = {
        "schema_version": module.DISPATCH_SCHEMA,
        "component": "mut_continuation",
        "task_id": spec["task_id"],
        "task_spec_path": str(path),
        "task_spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state": "LAUNCHING",
        "attempt_count": attempt_count,
        "attempts": [
            {
                "attempt": attempt_count,
                "attempt_uuid": f"attempt-{attempt_count}",
                "launcher_pid": 9000 + attempt_count,
                "launcher_start_ticks": 10,
                "started_at": datetime.fromtimestamp(0, tz=timezone.utc).isoformat(),
                "output_root": str(output),
                "exit_receipt": str(exit_receipt),
            }
        ],
    }
    module._atomic_json(component_root / "dispatch_state.json", state)
    monkeypatch.setattr(
        module, "_process_identity", lambda _pid: {"alive": False}
    )
    monkeypatch.setattr(module, "_writer_pids", lambda _root: [])
    result = module._dispatch(
        tmp_path / "state",
        PROJECT_ROOT,
        "mut_continuation",
        reason="fixture",
        dry_run=False,
        now_epoch=100,
    )
    assert result["state"] == "FAILED_TO_START"
    assert result["retry_after_seconds"] == expected_backoff
    assert "ALREADY_DISPATCHED" not in json.dumps(result)
    persisted = json.loads(
        (component_root / "dispatch_state.json").read_text(encoding="utf-8")
    )
    assert persisted["state"] == "FAILED_TO_START"
    assert persisted["next_retry_at_unix"] == 100 + expected_backoff


def test_third_failed_launch_becomes_terminal_after_final_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, spec = _task_spec(tmp_path, monkeypatch)
    component_root = tmp_path / "state/components/mut_continuation"
    component_root.mkdir(parents=True)
    module._atomic_json(
        component_root / "dispatch_state.json",
        {
            "schema_version": module.DISPATCH_SCHEMA,
            "component": "mut_continuation",
            "task_id": spec["task_id"],
            "task_spec_path": str(path),
            "task_spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "state": "BACKOFF",
            "attempt_count": 3,
            "attempts": [],
            "next_retry_at_unix": 300,
            "failure_reason": "fixture",
        },
    )
    monkeypatch.setattr(module, "_writer_pids", lambda _root: [])
    result = module._dispatch(
        tmp_path / "state",
        PROJECT_ROOT,
        "mut_continuation",
        reason="fixture",
        dry_run=False,
        now_epoch=301,
    )
    assert result == {
        "state": "BLOCKED_LAUNCHER_RETRY_EXHAUSTED",
        "attempt_count": 3,
    }


def test_duplicate_output_writer_blocks_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _task_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_writer_pids", lambda _root: [111, 222])
    result = module._dispatch(
        tmp_path / "state",
        PROJECT_ROOT,
        "mut_continuation",
        reason="fixture",
        dry_run=True,
    )
    assert result["state"] == "BLOCKED_DUPLICATE_OUTPUT_WRITER"
    assert result["writer_pids"] == [111, 222]


def test_retry_refuses_live_writer_from_prior_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, spec = _task_spec(tmp_path, monkeypatch)
    component_root = tmp_path / "state/components/mut_continuation"
    component_root.mkdir(parents=True)
    prior_output = str(tmp_path / "old-attempt")
    module._atomic_json(
        component_root / "dispatch_state.json",
        {
            "schema_version": module.DISPATCH_SCHEMA,
            "component": "mut_continuation",
            "task_id": spec["task_id"],
            "task_spec_path": str(path),
            "task_spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "state": "READY",
            "attempt_count": 1,
            "attempts": [{"attempt": 1, "output_root": prior_output}],
        },
    )
    monkeypatch.setattr(
        module,
        "_writer_pids",
        lambda root: [777] if root == prior_output else [],
    )
    result = module._dispatch(
        tmp_path / "state",
        PROJECT_ROOT,
        "mut_continuation",
        reason="fixture",
        dry_run=True,
    )
    assert result["state"] == "BLOCKED_DUPLICATE_TASK_OWNER"
    assert result["prior_output_writers"] == {prior_output: [777]}


def test_internal_launcher_environment_does_not_inherit_ambient_science_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUT_FAST_SPEC", "/ambient/must/not/leak")
    environment = module._clean_launcher_environment({"ONLY_FROM_SPEC": "yes"})
    assert environment["ONLY_FROM_SPEC"] == "yes"
    assert "MUT_FAST_SPEC" not in environment


def test_launcher_preflights_task_specs_and_slurm_pair_remains_hpc_safe() -> None:
    launcher = (PROJECT_ROOT / "scripts/autodl/launch_main_and_ablations_v1.sh").read_text(
        encoding="utf-8"
    )
    for variable in module.TASK_SPEC_ENV.values():
        assert variable in launcher
    assert '.get("controller_pid") == int(sys.argv[2])' in launcher
    slurm = (PROJECT_ROOT / "scripts/slurm/run_main_and_ablations_v1.sh").read_text(
        encoding="utf-8"
    )
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert token in slurm
