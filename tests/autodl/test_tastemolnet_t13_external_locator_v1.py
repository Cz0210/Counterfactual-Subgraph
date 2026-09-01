import hashlib
import json
from pathlib import Path
import uuid

import pytest

from scripts.autodl.run_tastemolnet_t13_external_locator_v1 import (
    TasteT13ExternalLocatorError,
    inspect_chain,
    run_follower,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/autodl/launch_tastemolnet_t13_external_locator_v1.sh"
WORKER = ROOT / "scripts/autodl/run_tastemolnet_t13_external_locator_v1.py"
SLURM = ROOT / "scripts/slurm/run_tastemolnet_t13_external_locator_v1.sh"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _live_pid(proc_root: Path, pid: int, *, start_ticks: int) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True, exist_ok=True)
    fields = ["S"] + ["1"] * 18 + [str(start_ticks)]
    (process / "stat").write_text(
        f"{pid} (bash) {' '.join(fields)}\n", encoding="utf-8"
    )


def _ready_chain(tmp_path: Path) -> dict[str, Path]:
    control = tmp_path / "control"
    output_base = tmp_path / "outputs" / "t13-full"
    dual = control / "tastemolnet-t8-dual-branch-recovery-known"
    downstream = dual / "downstream-salvage"
    t13 = control / "tastemolnet-t13-after-t8-salvage-known"
    managed_t8 = tmp_path / "outputs" / "managed-t8"
    for path in (output_base, downstream, t13, managed_t8):
        path.mkdir(parents=True, exist_ok=True)
    attempt = str(uuid.uuid4())
    terminal = output_base / f"attempt-{attempt}"
    terminal.mkdir()

    (dual / "state").write_text("PASS_AND_T13_RELAY_PERSISTED\n", encoding="utf-8")
    (dual / "completed_t8_root").write_text(f"{managed_t8}\n", encoding="utf-8")
    (downstream / "completed_t8_root").write_text(
        f"{managed_t8}\n", encoding="utf-8"
    )
    (downstream / "t13-relay-launch.txt").write_text(
        "controller_id=tastemolnet-t13-after-t8-salvage-known\n"
        "controller_pid=12345\n"
        f"controller_root={t13}\n",
        encoding="utf-8",
    )
    (t13 / "launcher.pid").write_text("12345\n", encoding="utf-8")
    (t13 / "controller.pid").write_text("12345\n", encoding="utf-8")
    (t13 / "launch.env").write_text(
        f"attempt_id={attempt}\n"
        f"output_root={terminal}\n"
        "gpu_index=1\n"
        "gpu_uuid=GPU-00000000-0000-0000-0000-000000000000\n"
        f"t8_pass_root={managed_t8}\n",
        encoding="utf-8",
    )
    (t13 / "state").write_text("PASS\n", encoding="utf-8")
    _write_json(
        t13 / "heartbeat.json",
        {"controller_pid": 12345, "phase": "PASS", "science_pid": 0},
    )
    (t13 / "completed_output_root").write_text(f"{terminal}\n", encoding="utf-8")

    (terminal / "PASS").write_bytes(b"PASS\n")
    (terminal / "SEALED").write_bytes(b"SEALED\n")
    audit = {
        "schema_version": "tastemolnet_t13_terminal_verification_v1",
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "stage": "T13_GLOBALGCE_FULL",
        "status": "PASS",
        "passed": True,
        "audit_passed": True,
        "frozen": True,
        "registry_status": "FROZEN_PASS",
        "registry_reason_codes": [],
    }
    _write_json(terminal / "final_artifact_audit.json", audit)
    audit_sha = hashlib.sha256((terminal / "final_artifact_audit.json").read_bytes()).hexdigest()
    _write_json(
        terminal / "run_manifest.json",
        {
            "schema_version": "tastemolnet_t13_run_manifest_v1",
            "dataset": "TasteMolNet",
            "method": "GlobalGCE",
            "stage": "T13_GLOBALGCE_FULL",
            "status": "PASS",
            "state": "PASS",
            "run_complete": True,
            "finalized": True,
            "frozen": True,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "independent_terminal_verification_required": False,
            "worker_wrote_pass": False,
            "terminal_verifier": "separate_verify_only_invocation",
            "final_artifact_audit_sha256": audit_sha,
        },
    )
    _write_json(
        terminal / "checkpoint.json",
        {
            "schema_version": "tastemolnet_t13_checkpoint_v1",
            "stage": "T13_GLOBALGCE_FULL",
            "phase": "PASS",
            "detail": {"final_artifact_audit_sha256": audit_sha},
        },
    )
    return {
        "control": control,
        "output_base": output_base,
        "dual": dual,
        "t13": t13,
        "terminal": terminal,
        "proc": tmp_path / "proc",
    }


def test_ready_chain_writes_exact_standard_locator(tmp_path: Path) -> None:
    paths = _ready_chain(tmp_path)
    follower = tmp_path / "locator-controller"
    result = run_follower(
        t8_dual_controller_root=paths["dual"],
        control_root=paths["control"],
        t13_output_base=paths["output_base"],
        locator_path=follower / "cell_root_locator.json",
        heartbeat_path=follower / "heartbeat.json",
        poll_seconds=5,
        once=True,
    )
    assert result["state"] == "PASS"
    assert json.loads((follower / "cell_root_locator.json").read_text()) == {
        "schema_version": "fast16_matrix_cell_root_locator_v1",
        "status": "READY",
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "terminal_root": str(paths["terminal"]),
    }
    assert json.loads((follower / "heartbeat.json").read_text())["state"] == "PASS"


def test_absent_t13_launch_is_waiting_and_writes_no_locator(tmp_path: Path) -> None:
    control = tmp_path / "control"
    output_base = tmp_path / "outputs" / "t13-full"
    dual = control / "tastemolnet-t8-dual-branch-recovery-known"
    dual.mkdir(parents=True)
    output_base.mkdir(parents=True)
    (dual / "launcher.pid").write_text("11111\n")
    (dual / "controller.pid").write_text("11111\n")
    proc = tmp_path / "proc"
    _live_pid(proc, 11111, start_ticks=777)
    (dual / "state").write_text("T8_TARGET_0_FRESH_RECOVERY_RUNNING\n")
    follower = tmp_path / "locator-controller"
    result = run_follower(
        t8_dual_controller_root=dual,
        control_root=control,
        t13_output_base=output_base,
        locator_path=follower / "cell_root_locator.json",
        heartbeat_path=follower / "heartbeat.json",
        poll_seconds=5,
        once=True,
        proc_root=proc,
    )
    assert result["state"] == "WAITING_T13_RELAY_LAUNCH"
    assert result["t8_controller_start_ticks"] == 777
    assert not (follower / "cell_root_locator.json").exists()


def test_terminal_hash_drift_fails_closed(tmp_path: Path) -> None:
    paths = _ready_chain(tmp_path)
    manifest = json.loads((paths["terminal"] / "run_manifest.json").read_text())
    manifest["final_artifact_audit_sha256"] = "0" * 64
    _write_json(paths["terminal"] / "run_manifest.json", manifest)
    with pytest.raises(TasteT13ExternalLocatorError, match="audit hash binding"):
        inspect_chain(
            t8_dual_controller_root=paths["dual"],
            control_root=paths["control"],
            t13_output_base=paths["output_base"],
        )


def test_completed_root_before_pass_state_remains_waiting(tmp_path: Path) -> None:
    paths = _ready_chain(tmp_path)
    _live_pid(paths["proc"], 12345, start_ticks=888)
    (paths["t13"] / "state").write_text("T13_FULL_RUNNING\n", encoding="utf-8")
    _write_json(
        paths["t13"] / "heartbeat.json",
        {"controller_pid": 12345, "phase": "T13_FULL_RUNNING", "science_pid": 54321},
    )
    observed = inspect_chain(
        t8_dual_controller_root=paths["dual"],
        control_root=paths["control"],
        t13_output_base=paths["output_base"],
        proc_root=paths["proc"],
    )
    assert observed["state"] == "WAITING_T13_CONTROLLER_PASS"
    assert observed["t13_controller_start_ticks"] == 888
    assert observed["completed_output_root"] == str(paths["terminal"])


def test_dead_nonterminal_controller_fails_instead_of_waiting(tmp_path: Path) -> None:
    control = tmp_path / "control"
    output_base = tmp_path / "outputs" / "t13-full"
    dual = control / "tastemolnet-t8-dual-branch-recovery-known"
    dual.mkdir(parents=True)
    output_base.mkdir(parents=True)
    (dual / "launcher.pid").write_text("11111\n")
    (dual / "controller.pid").write_text("11111\n")
    (dual / "state").write_text("T8_TARGET_0_FRESH_RECOVERY_RUNNING\n")
    with pytest.raises(TasteT13ExternalLocatorError, match="is not live"):
        inspect_chain(
            t8_dual_controller_root=dual,
            control_root=control,
            t13_output_base=output_base,
            proc_root=tmp_path / "empty-proc",
        )


def test_terminal_pass_allows_controller_to_have_exited(tmp_path: Path) -> None:
    paths = _ready_chain(tmp_path)
    observed = inspect_chain(
        t8_dual_controller_root=paths["dual"],
        control_root=paths["control"],
        t13_output_base=paths["output_base"],
        proc_root=tmp_path / "empty-proc",
    )
    assert observed["state"] == "READY"


def test_chain_and_output_roots_are_bound(tmp_path: Path) -> None:
    paths = _ready_chain(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    launch = paths["t13"] / "launch.env"
    launch.write_text(launch.read_text().replace(str(paths["terminal"]), str(outside)))
    with pytest.raises(TasteT13ExternalLocatorError, match="output escaped"):
        inspect_chain(
            t8_dual_controller_root=paths["dual"],
            control_root=paths["control"],
            t13_output_base=paths["output_base"],
        )


def test_external_follower_has_no_science_or_signal_path() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "run_tastemolnet_globalgce_full.py" not in worker
    assert "run_tastemolnet_globalgce_full.py" not in launcher
    for forbidden in ("os.kill", "pkill", "killall", "SIGKILL", "SIGTERM"):
        assert forbidden not in worker
        assert forbidden not in launcher
    assert "completed_output_root" in worker
    assert "t13-relay-launch.txt" in worker
    assert "fast16_matrix_cell_root_locator_v1" in worker
    assert "RUN_GNN_ABLATION" in launcher
    assert 'mkdir -p "$ROOT" "$OUTPUT_BASE"' in launcher


def test_paired_slurm_keeps_hpc_baseline() -> None:
    text = SLURM.read_text(encoding="utf-8")
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
        assert token in text
