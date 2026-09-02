from __future__ import annotations

import importlib.util
import json
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
            "CHEMLLM_7B_PPO_MAIN,CHEMLLM_2B_OFF_THE_SHELF"
        ),
        "GNN_BACKBONES": "gine,gin,gcn,gatv2,gps",
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
        "t14_convergence_auditor",
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


def test_sidecar_uses_persistent_t14_auditor_relay() -> None:
    assert module.DEFAULT_LAUNCHERS["t14_convergence_auditor"].endswith(
        "launch_t14_external_convergence_auditor_relay_v1.sh"
    )
