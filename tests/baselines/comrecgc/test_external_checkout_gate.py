from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.verify_comrecgc_checkout as gate
from src.baselines.comrecgc.contracts import UPSTREAM_COMMIT
from src.baselines.comrecgc import upstream


def test_gate_fails_closed_for_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gate.verify_checkout(tmp_path / "missing")


def test_gate_records_commit_files_and_offline_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "COMRECGC"
    root.mkdir()
    for name in gate.REQUIRED_FILES:
        (root / name).write_text(f"# {name}\n", encoding="utf-8")

    monkeypatch.setattr(gate, "validate_upstream_checkout", lambda value: Path(value))
    monkeypatch.setattr(gate, "read_upstream_commit", lambda *_args: UPSTREAM_COMMIT)

    @contextmanager
    def imported(_root: Path):
        yield {
            "comrecgc": SimpleNamespace(),
            "common_recourse": SimpleNamespace(),
        }

    monkeypatch.setattr(gate, "imported_upstream", imported)
    result = gate.verify_checkout(root, validate_imports=True)
    assert result["passed"] is True
    assert result["commit_match"] is True
    assert result["import_pass"] is True
    assert result["network_required"] is False


def test_gate_reuses_process_scoped_safe_commit_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "migrated-read-only-checkout"
    root.mkdir()
    for name in gate.REQUIRED_FILES:
        (root / name).write_text(f"# {name}\n", encoding="utf-8")

    monkeypatch.setattr(gate, "validate_upstream_checkout", lambda value: Path(value))
    observed: list[list[str]] = []
    temporary_configs: list[Path] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        observed.append(list(argv))
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        safe_config = Path(str(environment["GIT_CONFIG_GLOBAL"]))
        temporary_configs.append(safe_config)
        assert safe_config.read_text(encoding="utf-8") == (
            f'[safe]\n\tdirectory = "{root.resolve()}"\n'
        )
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert kwargs["timeout"] == 30
        return SimpleNamespace(stdout=f"{UPSTREAM_COMMIT}\n")

    monkeypatch.setattr(upstream.subprocess, "run", fake_run)
    result = gate.verify_checkout(root)

    assert result["passed"] is True
    assert result["actual_commit"] == UPSTREAM_COMMIT
    assert observed == [["git", "-C", str(root.resolve()), "rev-parse", "HEAD"]]
    assert all(not path.exists() for path in temporary_configs)


def test_gate_rejects_non_contract_commit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen project contract"):
        gate.verify_checkout(tmp_path, expected_commit="0" * 40)
