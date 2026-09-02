from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.baselines.comrecgc.contracts import stable_json_sha256
from scripts.autodl import run_t14_external_convergence_auditor_relay_v1 as relay


def _write_one_shot_result(root: Path, *, step: int, converged: bool) -> dict[str, Any]:
    root.mkdir(parents=True)
    audit = {
        "schema_version": "tastemolnet_t14_external_convergence_audit_v1",
        "status": "CONVERGED_EARLY_STOP" if converged else "CONTINUE_T14",
        "available_steps": [5_000, 10_000, step],
        "converged": converged,
        "active_t14_root_modified": False,
        "active_sqlite_opened": False,
        "checkpoint_sqlite_opened": False,
        "signal_sent": False,
    }
    audit["audit_sha256"] = stable_json_sha256(audit)
    relay.write_json(root / "t14_external_convergence_audit.json", audit)
    if converged:
        receipt = {
            "schema_version": "tastemolnet_t14_external_convergence_receipt_v1",
            "status": "PASS",
            "audit_path": str(root / "t14_external_convergence_audit.json"),
            "audit_sha256": audit["audit_sha256"],
            "stop_action_performed": False,
            "next_safe_checkpoint_exact_pid_handover_required": True,
            "calibration_loaded": False,
            "test_loaded": False,
        }
        receipt["receipt_sha256"] = stable_json_sha256(receipt)
        relay.write_json(root / "t14_convergence_early_stop_receipt.json", receipt)
    return audit


def test_relay_waits_metadata_only_before_12500() -> None:
    action, step = relay.choose_relay_action(
        [5_000, 10_000], audited_through_step=0, converged=False
    )
    assert action == "WAITING_FOR_12500"
    assert step is None


def test_relay_reaudits_each_new_committed_checkpoint() -> None:
    assert relay.choose_relay_action(
        [5_000, 10_000, 12_500], audited_through_step=0, converged=False
    ) == ("AUDIT_NEW_COMMITTED_CHECKPOINT", 12_500)
    assert relay.choose_relay_action(
        [5_000, 10_000, 12_500], audited_through_step=12_500, converged=False
    ) == ("WAITING_FOR_NEXT_COMMITTED_CHECKPOINT", None)
    assert relay.choose_relay_action(
        [5_000, 10_000, 12_500, 15_000],
        audited_through_step=12_500,
        converged=False,
    ) == ("AUDIT_NEW_COMMITTED_CHECKPOINT", 15_000)
    assert relay.choose_relay_action(
        [5_000, 10_000, 12_500, 15_000],
        audited_through_step=0,
        converged=False,
    ) == ("AUDIT_NEW_COMMITTED_CHECKPOINT", 15_000)


def test_relay_persists_continue_then_convergence_without_signal(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    relay_root = tmp_path / "relay"
    relay_root.mkdir()
    commit = "a" * 40
    state = relay._initial_state(checkpoint_root, commit)

    attempt_12500 = relay_root / "audits" / "step-000000012500" / "attempt-a"
    first = _write_one_shot_result(attempt_12500, step=12_500, converged=False)
    state = relay._commit_audit_result(
        relay_root,
        state,
        attempt_root=attempt_12500,
        trigger_step=12_500,
        audit=first,
    )
    assert state["status"] == "WAITING_FOR_NEXT_COMMITTED_CHECKPOINT"
    assert state["audited_through_step"] == 12_500
    assert not (relay_root / "t14_convergence_relay_receipt.json").exists()

    attempt_15000 = relay_root / "audits" / "step-000000015000" / "attempt-b"
    second = _write_one_shot_result(attempt_15000, step=15_000, converged=True)
    state = relay._commit_audit_result(
        relay_root,
        state,
        attempt_root=attempt_15000,
        trigger_step=15_000,
        audit=second,
    )
    assert state["status"] == "CONVERGED_STOP_ACTION_PENDING_EXACT_PID_HANDOVER"
    assert state["stop_action_performed"] is False
    assert state["signal_sent"] is False
    receipt = json.loads(
        (relay_root / "t14_convergence_relay_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["stop_action_pending_exact_pid_handover"] is True
    assert receipt["signal_sent"] is False
    assert receipt["receipt_sha256"] == stable_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_relay_rejects_off_cadence_checkpoint() -> None:
    with pytest.raises(relay.T14ExternalConvergenceRelayError, match="off-cadence"):
        relay.choose_relay_action(
            [5_000, 11_000], audited_through_step=0, converged=False
        )


def test_relay_has_no_signal_or_sqlite_open_path() -> None:
    project = Path(__file__).resolve().parents[2]
    paths = [
        project / "scripts/autodl/run_t14_external_convergence_auditor_relay_v1.py",
        project / "scripts/autodl/launch_t14_external_convergence_auditor_relay_v1.sh",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "sqlite3" not in combined
    assert "os.kill" not in combined
    assert "SIGTERM" not in combined
    assert "SIGKILL" not in combined
    assert "tastemolnet-t14-external-convergence-relay.lock" in combined


def test_relay_python_never_opens_a_sqlite_path() -> None:
    project = Path(__file__).resolve().parents[2]
    source = (
        project / "scripts/autodl/run_t14_external_convergence_auditor_relay_v1.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    sqlite_open_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in {"connect", "execute", "executemany"}:
            sqlite_open_calls.append(name)
    assert sqlite_open_calls == []


def test_slurm_pair_has_required_hpc_contract() -> None:
    project = Path(__file__).resolve().parents[2]
    script = (
        project / "scripts/slurm/run_t14_external_convergence_auditor_relay_v1.sh"
    ).read_text(encoding="utf-8")
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
        assert token in script
