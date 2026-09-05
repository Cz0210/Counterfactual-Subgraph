import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from src.utils import gnn_seed7_mac_relay as relay


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(relay, "MAC_PARENT", tmp_path / "mac")
    p = relay.RelayPlan(str(uuid.uuid4()))
    p.control.mkdir(parents=True)
    return p


def receipt(data=b"verified-package"):
    return {"state": "PASS", "main_matrix_write": False, "path": str(relay.HPC_ROOT / relay.ARCHIVE_NAME), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "scientific_engine_commit": "532e83733971701b0709086469d2ed8955a96e25",
        "publication_driver_commit": "fd98c5f23bf835f2b68799d03b7a2fd8b8b713f7"}


def producer_backbones():
    """Read the real producer contract, without importing GPU/science modules."""
    source = Path(__file__).resolve().parents[1] / "src/ablations/gnn/cpu_evaluation.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "BACKBONES" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("Real producer BACKBONES contract missing")


class Pulse:
    def update(self, **_):
        pass


class FakeRunner:
    def __init__(self, plan, data=b"verified-package", fail=None):
        self.plan, self.data, self.fail, self.calls = plan, data, fail, []

    def run(self, argv, stage):
        self.calls.append((stage, argv))
        if stage == self.fail:
            raise relay.RelayError("simulated network interruption")
        r = receipt()
        if stage == "HPC_TO_MAC":
            self.plan.local_partial.write_bytes(self.data)
        if stage == "VERIFY_HPC_ARCHIVE":
            return json.dumps(r)
        if stage == "VERIFY_AUTODL_ARCHIVE":
            return json.dumps({**r, "path": str(self.plan.remote_final)})
        if stage == "AUTODL_INDEPENDENT_IMPORT":
            return json.dumps({"schema_version": "bace_gnn_portable_location_overlay_v1", "state": "PASS", "archive_sha256": r["sha256"], "main_matrix_write": False,
                "evaluation_root": str(self.plan.import_root / "evaluation"),
                "model_roots": {name: str(self.plan.import_root / "classifiers" / name) for name in producer_backbones()},
                "scientific_engine_commit": r["scientific_engine_commit"], "publication_driver_commit": r["publication_driver_commit"],
                "classifier_inference_rerun": False, "ot_recomputed": False,
                "historical_hpc_paths_opened": False, "original_manifest_paths_preserved": True})
        return "{}"


def test_fixed_campaign_fresh_roots_and_rsync269(plan):
    assert relay.HPC_RECEIPT.name == "result_package.json"
    assert "exact-parent-closeout-fd98c5f2/verified" in str(relay.HPC_RECEIPT)
    assert plan.incoming != plan.import_root and plan.attempt_id in str(plan.incoming)
    argv = relay.transfer_command("tongji-hpc:/fixed/archive", str(plan.local_partial))
    assert "--partial" in argv and "BatchMode=yes" in " ".join(argv)
    assert not any(arg.startswith(("--append-verify", "--info", "--delete", "--inplace", "--protect-args")) for arg in argv)


@pytest.mark.parametrize("change", ({"state": "FAILED"}, {"path": "/share/home/u20526/other.tar.gz"}, {"bytes": 0}, {"bytes": True}, {"sha256": "not-a-hash"}, {"main_matrix_write": True}, {"scientific_engine_commit": "0" * 40}, {"publication_driver_commit": "0" * 40}))
def test_receipt_fails_closed(change):
    with pytest.raises(relay.RelayError):
        relay.validate_receipt({**receipt(), **change})


def test_attempt_id_rejects_traversal():
    with pytest.raises(ValueError):
        relay.RelayPlan("../../other")


def test_success_binds_every_transport_and_calls_only_import(plan):
    runner = FakeRunner(plan)
    result = relay.transfer_and_import(plan, receipt(), runner, Pulse())
    assert result["state"] == "VERIFIED_PACKAGE_IMPORTED"
    assert not plan.local_partial.exists() and plan.local_final.read_bytes() == b"verified-package"
    assert [s for s, _ in runner.calls] == ["VERIFY_HPC_ARCHIVE", "HPC_TO_MAC", "PREPARE_AUTODL_FRESH_ROOTS", "MAC_TO_AUTODL", "VERIFY_AUTODL_ARCHIVE", "AUTODL_INDEPENDENT_IMPORT"]
    assert (plan.control / "mac_transport_receipt.json").is_file()
    assert (plan.control / "autodl_import_receipt.json").is_file()
    command = " ".join(runner.calls[-1][1])
    assert "import_bace_gnn_verified.py" in command and "CUDA_VISIBLE_DEVICES=" in command
    assert "--archive-path" in command and "--expected-sha256" in command
    assert "fast16_matrix_authority" not in command and "run_bace_native_llm" not in command


def test_real_five_backbone_importer_schema_is_preserved(plan):
    assert tuple(relay.EXPECTED_BACKBONES) == tuple(producer_backbones())
    result = relay.transfer_and_import(plan, receipt(), FakeRunner(plan), Pulse())
    imported = json.loads((plan.control / "autodl_import_receipt.json").read_text())
    assert result["state"] == "VERIFIED_PACKAGE_IMPORTED"
    assert "gatedgcn_plus" in imported["model_roots"] and len(imported["model_roots"]) == 5
    assert imported["scientific_engine_commit"] == receipt()["scientific_engine_commit"]
    assert imported["publication_driver_commit"] == receipt()["publication_driver_commit"]


def test_corrupt_local_transfer_keeps_partial_and_never_uploads(plan):
    runner = FakeRunner(plan, data=b"corrupt")
    with pytest.raises(relay.RelayError, match="Mac archive bytes/SHA"):
        relay.transfer_and_import(plan, receipt(), runner, Pulse())
    assert plan.local_partial.read_bytes() == b"corrupt" and not plan.local_final.exists()
    assert len(runner.calls) == 2


def test_network_failure_no_retries_and_local_evidence_preserved(plan):
    runner = FakeRunner(plan, fail="MAC_TO_AUTODL")
    with pytest.raises(relay.RelayError, match="network"):
        relay.transfer_and_import(plan, receipt(), runner, Pulse())
    assert plan.local_final.is_file()
    assert sum(stage == "MAC_TO_AUTODL" for stage, _ in runner.calls) == 1
    assert all(stage != "AUTODL_INDEPENDENT_IMPORT" for stage, _ in runner.calls)


@pytest.mark.parametrize("change", ({"evaluation_root": "/unexpected/evaluation"}, {"main_matrix_write": True}, {"classifier_inference_rerun": True}, {"ot_recomputed": True}, {"scientific_engine_commit": "0" * 40}, {"publication_driver_commit": "0" * 40}, {"model_roots": {}}))
def test_importer_must_bind_fresh_root_without_rerunning_science(plan, change):
    runner = FakeRunner(plan)
    original = runner.run
    def changed(argv, stage):
        output = original(argv, stage)
        return json.dumps({**json.loads(output), **change}) if stage == "AUTODL_INDEPENDENT_IMPORT" else output
    runner.run = changed
    with pytest.raises(relay.RelayError, match="independent import"):
        relay.transfer_and_import(plan, receipt(), runner, Pulse())
    assert plan.local_final.is_file()
    assert not (plan.control / "autodl_import_receipt.json").exists()


def test_local_file_identity_rejects_symlink(plan):
    actual = plan.mac_root / "original"
    actual.write_bytes(b"x")
    plan.local_partial.symlink_to(actual)
    with pytest.raises(relay.RelayError, match="regular"):
        relay.file_identity(plan.local_partial)


def test_heartbeat_continues_during_transport_subprocess(plan, monkeypatch):
    monkeypatch.setattr(relay, "HEARTBEAT_SECONDS", .02)
    seen = []
    original = relay.atomic_json
    def observed(path, value):
        if path.name == "heartbeat.json":
            seen.append(dict(value))
        original(path, value)
    monkeypatch.setattr(relay, "atomic_json", observed)
    with relay.Heartbeat(plan) as pulse:
        relay.CommandRunner(plan.control, pulse).run([sys.executable, "-c", "import time;time.sleep(.15)"], "TINY_TEST_CHILD")
    assert sum(row.get("active_child_pid") is not None for row in seen) >= 2


def test_remote_verify_partial_then_atomic_rename(tmp_path):
    data = b"safe round trip"
    source, final = tmp_path / "archive.partial", tmp_path / "archive.tar.gz"
    source.write_bytes(data)
    r = subprocess.run([sys.executable, "-c", relay.VERIFY_ARCHIVE, str(source), str(len(data)), hashlib.sha256(data).hexdigest(), str(final)], text=True, capture_output=True)
    assert r.returncode == 0 and final.read_bytes() == data and not source.exists()
    source.write_bytes(data)
    repeat = subprocess.run([sys.executable, "-c", relay.VERIFY_ARCHIVE, str(source), str(len(data)), hashlib.sha256(data).hexdigest(), str(final)], capture_output=True)
    assert repeat.returncode != 0 and source.exists()


def test_remote_fresh_root_guard(tmp_path):
    incoming, output = tmp_path / "incoming", tmp_path / "import"
    output.mkdir()
    r = subprocess.run([sys.executable, "-c", relay.PREPARE_AUTODL, str(tmp_path), str(incoming), str(output)], capture_output=True)
    assert r.returncode != 0 and not incoming.exists()


def test_no_unmounted_external_disk_fallback(monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda _: False)
    with pytest.raises(relay.RelayError, match="not mounted"):
        relay.run_relay(relay.RelayPlan(str(uuid.uuid4())))
