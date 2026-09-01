from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
from types import SimpleNamespace

from scripts.autodl import run_tastemolnet_t8_deadline as deadline


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/autodl/run_tastemolnet_t8_dual_branch_recovery_v1.sh"
LAUNCHER = ROOT / "scripts/autodl/launch_tastemolnet_t8_dual_branch_recovery_v1.sh"
SLURM = ROOT / "scripts/slurm/run_tastemolnet_t8_dual_branch_recovery_v1.sh"
FAILED_ATTEMPT_ID = "7c8cafa6-6679-49d7-bdc6-8d6259a0fbf4"
FAILED_SALVAGE_ATTEMPT_ID = "fadc2ac6-d1e8-4ede-b526-e06d0744eb8e"


def test_branch_worker_uses_isolated_python_for_module_provenance() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert (
        '"$PY" -I -B scripts/autodl/rerun_tastemolnet_t8_single_branch_v1.py'
        in text
    )


def test_existing_fixed_recovery_contract_accepts_7c8_as_immediate_source() -> None:
    recovery, config = deadline._zero_candidate_recovery_contract(
        SimpleNamespace(
            zero_candidate_recovery=True,
            recovery_source_attempt_id=FAILED_ATTEMPT_ID,
        ),
        attempt_id="e18148d7-7d4e-40df-8aa5-f4d63361d6a3",
    )
    assert recovery["enabled"] is True
    assert recovery["source_attempt_id"] == FAILED_ATTEMPT_ID
    assert recovery["epochs"] == deadline.ZERO_CANDIDATE_RECOVERY_EPOCHS == 25
    assert config.epochs == 25


def _write_executable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(payload).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fixture(tmp_path: Path, *, valid_failure_receipt: bool) -> tuple[dict[str, str], Path]:
    runtime = tmp_path / "runtime"
    control = runtime / "control"
    repo = tmp_path / "repo"
    scripts = repo / "scripts/autodl"
    scripts.mkdir(parents=True)
    events = tmp_path / "events.log"

    source = (
        runtime
        / "outputs/autodl/tastemolnet/globalgce/t8-smoke"
        / f"state-{FAILED_ATTEMPT_ID}"
    )
    (source / "target-0").mkdir(parents=True)
    (source / "target-2").mkdir()
    failure = (
        runtime
        / "outputs/autodl/tastemolnet/globalgce/t8-salvage"
        / f"attempt-{FAILED_SALVAGE_ATTEMPT_ID}"
        / "single-branch-rerun-request.json"
    )
    failure.parent.mkdir(parents=True)
    invalid_targets = [0, 2] if valid_failure_receipt else [0]
    failure.write_text(
        json.dumps(
            {
                "schema_version": "tastemolnet_t8_single_branch_rerun_request_v1",
                "status": "RERUN_REQUIRED",
                "invalid_target_branches": invalid_targets,
                "valid_target_branches_preserved": [] if valid_failure_receipt else [2],
                "rerun_both_branches": valid_failure_receipt,
                "source_artifacts_mutated": False,
                "reasons": {
                    str(target): "T8 salvage native rule application produced no candidates for [0, 2]"
                    for target in invalid_targets
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    t3 = tmp_path / "t3"
    t4 = tmp_path / "t4"
    checkpoint = t3 / "artifacts/checkpoint"
    official = tmp_path / "official"
    split = tmp_path / "splits"
    molclr = tmp_path / "molclr"
    for directory in (checkpoint, t4, official, split, molclr):
        directory.mkdir(parents=True, exist_ok=True)
    (t3 / "verification.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    (t4 / "verification.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    (checkpoint / "model.pt").write_bytes(b"same-frozen-gine")
    (checkpoint / "feature_schema.json").write_text('{"schema":1}\n', encoding="utf-8")
    for name in ("train.csv", "calibration.csv", "test.csv"):
        (split / name).write_text("id,label\n", encoding="utf-8")
    molclr_checkpoint = molclr / "model.pth"
    molclr_checkpoint.write_bytes(b"molclr")
    threshold = tmp_path / "threshold.json"
    threshold.write_text('{"status":"PASS"}\n', encoding="utf-8")

    _write_executable(
        scripts / "rerun_tastemolnet_t8_single_branch_v1.py",
        r'''
        import hashlib
        import json
        import os
        from pathlib import Path
        import sys

        def value(flag):
            return sys.argv[sys.argv.index(flag) + 1]

        target = int(value("--target"))
        attempt = value("--attempt-id")
        source = value("--source-attempt-id")
        state = Path(value("--state-root"))
        checkpoint = Path(value("--gnn-checkpoint"))
        events = Path(os.environ["T8_TEST_EVENTS"])
        with events.open("a", encoding="utf-8") as handle:
            handle.write(f"branch_start:{target}:{attempt}\n")
        branch = state / f"target-{target}"
        branch.mkdir(parents=True)
        receipt = {
            "schema_version": "tastemolnet_t8_single_branch_recovery_v1",
            "status": "PASS",
            "attempt_id": attempt,
            "recovery_source_attempt_id": source,
            "target_label": target,
            "source_label": 1,
            "state_root": str(state),
            "branch_root": str(branch),
            "oracle_checkpoint_hash": hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest(),
            "raw_generated_count": 1,
            "branch_evidence": {
                "target_label": target,
                "source_label": 1,
                "num_classes": 3,
                "test_loaded": False,
                "calibration_loaded": False,
                "rf_oracle_used": False,
            },
            "other_target_rerun": False,
            "test_loaded": False,
            "calibration_loaded": False,
            "gnn_ablation_started": False,
        }
        (state / "single_branch_recovery.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        with events.open("a", encoding="utf-8") as handle:
            handle.write(f"branch_end:{target}:{attempt}\n")
        print("[TASTE_T8_SINGLE_BRANCH_RECOVERY_PASS]")
        ''',
    )
    _write_executable(
        scripts / "run_tastemolnet_t8_salvage_release_v1.sh",
        r'''
        #!/usr/bin/env bash
        set -euo pipefail
        printf 'downstream\n' >> "$T8_TEST_EVENTS"
        mkdir -p "$T8_SALVAGE_CONTROLLER_ROOT"
        {
          printf 'source=%s\n' "$T8_SOURCE_ATTEMPT_ID"
          printf 'target0=%s\n' "$T8_TARGET_0_ROOT"
          printf 'target2=%s\n' "$T8_TARGET_2_ROOT"
          printf 'gpu=%s\n' "$T8_SALVAGE_GPU_INDEX"
          printf 'gnn_ablation=%s\n' "$RUN_GNN_ABLATION"
        } > "$T8_SALVAGE_CONTROLLER_ROOT/test.env"
        printf '/fresh/managed/t8\n' > "$T8_SALVAGE_CONTROLLER_ROOT/completed_t8_root"
        ''',
    )

    fake_bin = tmp_path / "bin"
    _write_executable(
        fake_bin / "nvidia-smi",
        r'''
        #!/usr/bin/env bash
        case "$*" in
          *--query-compute-apps=pid*) exit 0 ;;
          *--query-gpu=uuid*) printf 'GPU-test-1\n' ;;
          *) exit 64 ;;
        esac
        ''',
    )
    _write_executable(
        fake_bin / "flock",
        r'''
        #!/usr/bin/env bash
        exit 0
        ''',
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "AUTODL_PYTHON": sys.executable,
        "AUTODL_RUNTIME_ROOT": str(runtime),
        "AUTODL_CONTROL_ROOT": str(control),
        "T8_REPO_ROOT": str(repo),
        "T8_DUAL_CONTROLLER_ROOT": str(tmp_path / "controller"),
        "T8_DUAL_BASE": str(tmp_path / "dual"),
        "T8_DUAL_GSPAN_BASE": str(tmp_path / "gspan"),
        "T8_DUAL_POLL_SECONDS": "1",
        "TASTEMOLNET_T3_OUTPUT": str(t3),
        "TASTEMOLNET_T4_OUTPUT": str(t4),
        "TASTEMOLNET_GNN_CHECKPOINT": str(checkpoint),
        "TASTEMOLNET_TRAIN_CSV": str(split / "train.csv"),
        "TASTEMOLNET_CALIBRATION_CSV": str(split / "calibration.csv"),
        "TASTEMOLNET_TEST_CSV": str(split / "test.csv"),
        "TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT": str(official),
        "MOLCLR_ROOT": str(molclr),
        "MOLCLR_CHECKPOINT": str(molclr_checkpoint),
        "TASTEMOLNET_THRESHOLD_CONTRACT": str(threshold),
        "T8_TEST_EVENTS": str(events),
        "RUN_GNN_ABLATION": "0",
    }
    return env, events


def test_relay_is_fixed_sequential_and_never_reuses_failed_terminal() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    target_0 = text.index('run_branch 0 "$TARGET_0_ATTEMPT_ID"')
    target_2 = text.index('run_branch 2 "$TARGET_2_ATTEMPT_ID"')
    downstream = text.index("run_tastemolnet_t8_salvage_release_v1.sh")
    assert target_0 < target_2 < downstream
    for token in (
        FAILED_ATTEMPT_ID,
        FAILED_SALVAGE_ATTEMPT_ID,
        '"invalid_target_branches":[0,2]',
        '"valid_candidate_counts":{"0":0,"2":0}',
        'needle="T8 salvage native rule application produced no candidates for [0, 2]"',
        "TARGET_0_ATTEMPT_ID=",
        "TARGET_2_ATTEMPT_ID=",
        "T8_TARGET_0_ROOT=$TARGET_0_STATE/target-0",
        "T8_TARGET_2_ROOT=$TARGET_2_STATE/target-2",
        "T8_SALVAGE_GPU_INDEX=1",
        "PASS_AND_T13_RELAY_PERSISTED",
    ):
        assert token in text
    assert "run_tastemolnet_t8_deadline.py" not in text
    assert "kill -TERM" not in text
    assert "pkill" not in text
    assert "killall" not in text


def test_relay_waits_for_natural_gpu1_and_keeps_split_boundary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        'GPU_INDEX=${T8_DUAL_GPU_INDEX:-1}',
        '[[ "$GPU_INDEX" == "1" ]]',
        "WAITING_FOR_GPU1",
        "--query-compute-apps=pid",
        'gpu-$GPU_UUID.coordination.lock',
        "flock -n 8",
        "flock -u 8",
        "TASTEMOLNET_GNN_CHECKPOINT",
        "TASTEMOLNET_TRAIN_CSV",
        'value.get("test_loaded") is not False',
        'value.get("calibration_loaded") is not False',
        '"gnn_ablation_started":False',
        'RUN_GNN_ABLATION:-0',
    ):
        assert token in text


def test_launcher_persists_pid_heartbeat_and_disables_ablation() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "run_tastemolnet_t8_dual_branch_recovery_v1.sh",
        "T8_DUAL_CONTROLLER_ROOT",
        "T8_DUAL_GPU_INDEX=1",
        "RUN_GNN_ABLATION=0",
        "nohup bash",
        "launcher.pid",
        "controller_id=",
        "controller_pid=",
    ):
        assert token in text


def test_paired_slurm_is_thin_and_refuses_a_non_gpu1_allocation() -> None:
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
        "export T8_DUAL_GPU_INDEX=1",
        "export RUN_GNN_ABLATION=0",
        'CUDA_VISIBLE_DEVICES:-',
        "run_tastemolnet_t8_dual_branch_recovery_v1.sh",
    ):
        assert token in text


def test_relay_executes_target_zero_then_two_then_existing_chain(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path, valid_failure_receipt=True)
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    rows = events.read_text(encoding="utf-8").splitlines()
    assert [row.split(":", 1)[0] for row in rows] == [
        "branch_start",
        "branch_end",
        "branch_start",
        "branch_end",
        "downstream",
    ]
    assert rows[0].split(":")[1] == "0"
    assert rows[1].split(":")[1] == "0"
    assert rows[2].split(":")[1] == "2"
    assert rows[3].split(":")[1] == "2"
    assert rows[0].split(":")[2] != rows[2].split(":")[2]
    assert FAILED_ATTEMPT_ID not in {rows[0].split(":")[2], rows[2].split(":")[2]}

    controller = Path(env["T8_DUAL_CONTROLLER_ROOT"])
    heartbeat = json.loads((controller / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["phase"] == "PASS_AND_T13_RELAY_PERSISTED"
    assert heartbeat["gnn_ablation_started"] is False
    assert (controller / "completed_t8_root").read_text(encoding="utf-8").strip() == "/fresh/managed/t8"
    downstream = (controller / "downstream-salvage/test.env").read_text(encoding="utf-8")
    assert f"source={FAILED_ATTEMPT_ID}" in downstream
    assert "gpu=1" in downstream
    assert "gnn_ablation=0" in downstream
    assert f"state-{FAILED_ATTEMPT_ID}/target-0" not in downstream
    assert f"state-{FAILED_ATTEMPT_ID}/target-2" not in downstream


def test_relay_rejects_non_dual_failure_receipt_before_gpu_or_science(tmp_path: Path) -> None:
    env, events = _fixture(tmp_path, valid_failure_receipt=False)
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "T8_DUAL_FAILURE_RECEIPT_FIELD_MISMATCH" in result.stderr
    assert not events.exists()
