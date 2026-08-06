from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.baselines.comrecgc.freeze_recovery_result import REQUIRED, freeze
from src.baselines.comrecgc.contracts import sha256_file


def _source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    gate = tmp_path / "gate"
    source.mkdir()
    gate.mkdir()
    for name in REQUIRED:
        (source / name).write_text(f"artifact:{name}\n", encoding="utf-8")
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_complete": True,
                "method": "COMRECGC-Adapted-DeterministicChemRepair",
                "project_commit": "p",
                "upstream_commit": "u",
                "repair_policy_sha256": "r",
                "teacher_sha256": "t",
                "molclr_checkpoint_sha256": "m",
            }
        ),
        encoding="utf-8",
    )
    (gate / "gate_result.json").write_text(
        json.dumps({"audit_passed": True}), encoding="utf-8"
    )
    return source, gate


def test_freeze_is_atomic_and_preserves_source(tmp_path: Path) -> None:
    source, gate = _source(tmp_path)
    before = {name: sha256_file(source / name) for name in REQUIRED}
    output = tmp_path / "paper/comrecgc"

    manifest = freeze(source_dir=source, gate_dir=gate, output_dir=output)

    assert (output / "_FINALIZED.json").is_file()
    assert manifest["method"] == "COMRECGC-Adapted-DeterministicChemRepair"
    assert set(manifest["files"]) == set(REQUIRED)
    assert {name: sha256_file(source / name) for name in REQUIRED} == before
    for name in REQUIRED:
        assert sha256_file(output / name) == before[name]
        assert manifest["files"][name]["materialization_mode"] in {
            "hardlink",
            "atomic_copy",
        }


def test_freeze_refuses_to_overwrite(tmp_path: Path) -> None:
    source, gate = _source(tmp_path)
    output = tmp_path / "paper/comrecgc"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        freeze(source_dir=source, gate_dir=gate, output_dir=output)


def test_freeze_requires_passing_gate(tmp_path: Path) -> None:
    source, gate = _source(tmp_path)
    (gate / "gate_result.json").write_text(
        json.dumps({"audit_passed": False}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="passing full gate"):
        freeze(source_dir=source, gate_dir=gate, output_dir=tmp_path / "paper/comrecgc")
