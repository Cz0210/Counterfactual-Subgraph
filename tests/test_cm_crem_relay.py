"""Bounded local/transport interface checks; no scheduler or remote writes."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cm_relay_fixture", REPO/"scripts/run_cm_crem_relay.py")
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def test_fixed_stage_dag_and_no_duplicate_science_stages():
    assert len(relay.STAGES) == len(set(relay.STAGES))
    assert relay.STAGES.index("pilot-closeout") < relay.STAGES.index("attribution")
    assert relay.STAGES.index("select") < relay.STAGES.index("test") < relay.STAGES.index("audit")
    assert relay.STAGES[-2:] == ["export", "package"]


@pytest.mark.parametrize("root", ["/tmp/not_cm", "/share/home/u20526/czx/foo;echo_bad"])
def test_remote_scope_rejected_before_network(tmp_path, root):
    process = subprocess.run([sys.executable, str(REPO/"scripts/run_cm_crem_relay.py"),
        "--hpc-run-root", root, "--hpc-execution-root", "/share/home/u20526/czx/worktrees/cm-test",
        "--local-root", str(tmp_path), "--start-time-utc", "2026-09-09T16:27:44Z", "--once"],
        capture_output=True, text=True)
    assert process.returncode != 0
    assert "ValueError" in process.stderr


def test_transport_compatible_with_mac_rsync_269_and_scoped_lifetime():
    text = (REPO/"scripts/run_cm_crem_relay.py").read_text()
    assert '"--partial"' in text
    assert "--delete" not in text and "--append-verify" not in text and "--info" not in text
    assert "168*3600" in text and "time.sleep(300)" in text
    assert 'failures >= 2' in text
    assert "verify_import(local_package, local_manifest, imported)" in text
    assert "cm_import_receipt.json" in text


def test_record_only_import_help_and_no_model_load():
    process = subprocess.run([sys.executable, "-I", "-B", str(REPO/"scripts/import_cm_crem.py"), "--help"],
                             capture_output=True, text=True)
    assert process.returncode == 0
    assert all(x in process.stdout for x in ("--package", "--manifest", "--destination"))


@pytest.mark.parametrize("field,value", [("science_hash", "other"), ("package_sha256", "other"),
                                        ("package_bytes", 100), ("status", "PENDING")])
def test_reuse_requires_exact_package_not_just_same_science(field, value):
    manifest = {"science_hash": "s", "package_sha256": "p", "package_bytes": 101}
    receipt = {**manifest, "status": "CM_RESULT_IMPORT_VERIFIED"}
    relay.require_import_identity(receipt, manifest)
    receipt[field] = value
    with pytest.raises(ValueError, match="exact CM package"):
        relay.require_import_identity(receipt, manifest)


def test_external_mount_must_exist_and_remote_root_is_package_bound():
    text = (REPO/"scripts/run_cm_crem_relay.py").read_text()
    assert 'os.path.ismount("/Volumes/DireRaven")' in text
    assert 'Existing transfer root is not bound to this package' in text
