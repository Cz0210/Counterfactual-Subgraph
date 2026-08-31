from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import pytest

from src.baselines.comrecgc.contracts import sha256_file
from src.utils import autodl_aids_comrecgc_exact_postprocess_v1 as postprocess


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class _ContinuationFixture:
    external_dbscan_source_manifest: Path | None = None
    external_dbscan_source_receipt: Path | None = None
    common_recourse_resume: bool = False


def test_fresh_postprocess_binds_exact_terminal_and_publishes_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _json(tmp_path / "controller.json", {"controller": "aids"})
    dbscan = _json(tmp_path / "exact/dbscan/run_manifest.json", {"complete": True})
    receipt = _json(
        tmp_path / "exact/exact_recovery_receipt.json",
        {
            "run_complete": True,
            "dbscan_partition_proven": True,
            "dbscan_manifest_path": str(dbscan.resolve()),
            "dbscan_manifest_sha256": sha256_file(dbscan),
        },
    )
    manifest = {
        "stages": [
            {
                "stage_id": postprocess.EXACT_STAGE,
                "terminal_path": str(receipt),
            }
        ]
    }
    monkeypatch.setattr(
        postprocess, "load_bound_controller_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(
        postprocess,
        "validate_stage_terminal",
        lambda _manifest, *, stage_id: {
            "stage_receipt": json.loads(receipt.read_text())
        },
    )
    monkeypatch.setattr(
        postprocess,
        "_continuation_inputs",
        lambda _manifest, _output: _ContinuationFixture(),
    )
    output = tmp_path / "fresh-output"

    def _run(values: _ContinuationFixture) -> dict:
        assert values.external_dbscan_source_manifest == dbscan.resolve()
        assert values.external_dbscan_source_receipt == receipt.resolve()
        assert values.common_recourse_resume is True
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["OMP_NUM_THREADS"] == "4"
        _json(output / "common_recourse/_RUN_COMPLETE.json", {"run_complete": True})
        _json(output / "_RUN_COMPLETE.json", {"status": "PASS", "run_complete": True})
        (output / "PASS").write_bytes(b"PASS\n")
        return {"status": "PASS", "run_complete": True}

    monkeypatch.setattr(postprocess, "run_continuation", _run)
    monkeypatch.setattr(
        postprocess,
        "_validate_common_recourse_completion",
        lambda **_kwargs: None,
    )
    heartbeat = tmp_path / "control/heartbeat.json"
    result = postprocess.run_aids_exact_postprocess(
        controller_manifest_path=controller,
        exact_receipt_path=receipt,
        output_root=output,
        heartbeat_path=heartbeat,
        resume=False,
        max_workers=4,
        heartbeat_interval_seconds=5,
    )
    assert result["status"] == "PASS"
    assert result["dbscan_rerun"] is False
    assert result["max_workers"] == 4
    live = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert live["state"] == "PASS"
    assert live["dbscan_rerun"] is False
    assert live["max_workers"] == 4
    assert live["output_root"] == str(output.resolve())


def test_rejects_receipt_not_bound_to_controller_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _json(tmp_path / "controller.json", {"controller": "aids"})
    expected = _json(tmp_path / "exact/expected.json", {"status": "PASS"})
    supplied = _json(tmp_path / "exact/supplied.json", {"status": "PASS"})
    monkeypatch.setattr(
        postprocess,
        "load_bound_controller_manifest",
        lambda _path: {
            "stages": [
                {
                    "stage_id": postprocess.EXACT_STAGE,
                    "terminal_path": str(expected),
                }
            ]
        },
    )
    with pytest.raises(
        postprocess.AIDSExactPostprocessError,
        match="controller-bound terminal",
    ):
        postprocess.run_aids_exact_postprocess(
            controller_manifest_path=controller,
            exact_receipt_path=supplied,
            output_root=tmp_path / "fresh-output",
            heartbeat_path=tmp_path / "control/heartbeat.json",
            resume=False,
            max_workers=8,
            heartbeat_interval_seconds=5,
        )
    live = json.loads(
        (tmp_path / "control/heartbeat.json").read_text(encoding="utf-8")
    )
    assert live["state"] == "FAILED"


def test_rejects_worker_count_above_authorized_ceiling(tmp_path: Path) -> None:
    controller = _json(tmp_path / "controller.json", {"controller": "aids"})
    receipt = _json(tmp_path / "receipt.json", {"status": "PASS"})
    with pytest.raises(
        postprocess.AIDSExactPostprocessError,
        match="max_workers",
    ):
        postprocess.run_aids_exact_postprocess(
            controller_manifest_path=controller,
            exact_receipt_path=receipt,
            output_root=tmp_path / "fresh-output",
            heartbeat_path=tmp_path / "control/heartbeat.json",
            resume=False,
            max_workers=9,
            heartbeat_interval_seconds=5,
        )
    assert not (tmp_path / "fresh-output").exists()
